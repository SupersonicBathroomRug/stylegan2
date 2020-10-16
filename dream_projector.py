# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

#----------------------------------------------------------------------------

class DreamProjector:
    def __init__(self):
        self.num_steps                  = 300 # was 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.01 # was 0.1
        self.initial_noise_factor       = 0.05
        self.initial_noise_strength     =0.1
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5 # was 1e5, 1e4 makes sense, too
        self.verbose                    = False
        self.clone_net                  = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None
        self._celeba_classifier     = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, Gs, network_protobuf_path, layer_name, neuron_index):
        self._minibatch_size = 1
        self._Gs = Gs
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Image output graph.
        self._info('Building image output graph...')
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
        self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, self._Gs.components.synthesis.input_shape[1], 1])
        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        proc_images_expr = (self._images_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

        # Loss graph.
        self._info('Building loss graph...')

        if self._celeba_classifier is None:
            from lucid.modelzoo.vision_base import Model

            class FrozenNetwork(Model):
                model_path = network_protobuf_path
                image_shape = [256, 256, 3]
                image_value_range = (0, 1)
                input_name = 'input_1'

            network = FrozenNetwork()
            network.load_graphdef()
            # proc_images_expr.shape = (1, 3, 256, 256), range = (0, 255)
            # input_image.shape = (1, 256, 256, 3), range = (0, 1)
            input_image = tf.transpose(proc_images_expr, perm=(0, 2, 3, 1)) / 255

            # TODO memory leak
            ins = str(np.random.randint(10000))
            network.import_graph(t_input=input_image, scope=ins)

            # layer_name, neuron_index = "Mixed_5c_Branch_3_b_1x1_act/Relu", 16
            g = tf.get_default_graph()
            layer = g.get_tensor_by_name(ins + "/" + layer_name + ":0")
            neuron = layer[:, :, :, neuron_index]
            # reduce_max would make sense too.
            mean_activation = tf.reduce_mean(neuron, axis=(1, 2))
            self._loss = - mean_activation
            self._dist = self._loss

        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()

    def run(self):
        # Run to completion.
        self.start()
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.images = self.get_images()
        return pres

    def start(self):
        assert self._Gs is not None

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        tflib.set_vars({self._dlatents_var: (np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])+tf.random.normal([self._minibatch_size,1,1,512],0,1,tf.float32).numpy()*np.tile(self._dlatent_std, [self._minibatch_size,1,1])*self.initial_noise_strength)})
        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})

#----------------------------------------------------------------------------
