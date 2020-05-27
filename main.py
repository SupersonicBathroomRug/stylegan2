# Download the model of choice
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from io import BytesIO
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio

import pretrained_networks

parser = argparse.ArgumentParser()
parser.add_argument("--network_protobuf_path", type=str)
parser.add_argument("--layer_name", type=str)
parser.add_argument("--neuron_index", type=int)
parser.add_argument("--weight_editor", help='invert_top, invert_one, prune_top, prune_one', type=str)
parser.add_argument("--weight_edit_param", help='parameter of weight editor', type=int)

FLAGS = parser.parse_args()

network_protobuf_path = FLAGS.network_protobuf_path
layer_name = FLAGS.layer_name
fancy_layer_name = layer_name.replace("_act/Relu", "")
neuron_index = FLAGS.neuron_index
weight_editor = FLAGS.weight_editor
weight_edit_param = FLAGS.weight_edit_param


print(network_protobuf_path, layer_name, neuron_index, weight_editor, weight_edit_param)


# Choose between these pretrained models - I think 'f' is the best choice:

# 1024×1024 faces
# stylegan2-ffhq-config-a.pkl
# stylegan2-ffhq-config-b.pkl
# stylegan2-ffhq-config-c.pkl
# stylegan2-ffhq-config-d.pkl
# stylegan2-ffhq-config-e.pkl
# stylegan2-ffhq-config-f.pkl

# 512×384 cars
# stylegan2-car-config-a.pkl
# stylegan2-car-config-b.pkl
# stylegan2-car-config-c.pkl
# stylegan2-car-config-d.pkl
# stylegan2-car-config-e.pkl
# stylegan2-car-config-f.pkl

# 256x256 horses
# stylegan2-horse-config-a.pkl
# stylegan2-horse-config-f.pkl

# 256x256 churches
# stylegan2-church-config-a.pkl
# stylegan2-church-config-f.pkl

# 256x256 cats
# stylegan2-cat-config-f.pkl
# stylegan2-cat-config-a.pkl
network_pkl = "gdrive:networks/stylegan2-ffhq-config-f.pkl"

# If downloads fails, due to 'Google Drive download quota exceeded' you can try downloading manually from your own Google Drive account
# network_pkl = "/content/drive/My Drive/GAN/stylegan2-ffhq-config-f.pkl"

print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Useful utility functions...

# Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_in_w_space(dlatents, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    imgs = []
    for row, dlatent in log_progress(enumerate(dlatents), name = "Generating images"):
        #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
    return imgs       

def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
        
    imgs = []
    for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
    return imgs

def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs

# Generates a list of images, based on a list of seed for latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)

def saveImgs(imgs, location):
  for idx, img in log_progress(enumerate(imgs), size = len(imgs), name="Saving images"):
    file = location+ str(idx) + ".png"
    img.save(file)

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  str_file = BytesIO()
  PIL.Image.fromarray(a).save(str_file, format)
  im_data = str_file.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

        
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))
    
def drawLatent(image,latents,x,y,x2,y2, color=(255,0,0,100)):
  buffer = PIL.Image.new('RGBA', image.size, (0,0,0,0))
   
  draw = ImageDraw.Draw(buffer)
  cy = (y+y2)/2
  draw.rectangle([x,y,x2,y2],fill=(255,255,255,180), outline=(0,0,0,180))
  for i in range(len(latents)):
    mx = x + (x2-x)*(float(i)/len(latents))
    h = (y2-y)*latents[i]*0.1
    h = clamp(h,cy-y2,y2-cy)
    draw.line((mx,cy,mx,cy+h),fill=color)
  return PIL.Image.alpha_composite(image,buffer)
             
  
def createImageGrid(images, scale=0.25, rows=1):
   w,h = images[0].size
   w = int(w*scale)
   h = int(h*scale)
   height = rows*h
   cols = ceil(len(images) / rows)
   width = cols*w
   canvas = PIL.Image.new('RGBA', (width,height), 'white')
   for i,img in enumerate(images):
     img = img.resize((w,h), PIL.Image.ANTIALIAS)
     canvas.paste(img, (w*(i % cols), h*(i // cols))) 
   return canvas

def convertZtoW(latent, truncation_psi=0.7, truncation_cutoff=9):
  dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
  dlatent_avg = Gs.get_var('dlatent_avg') # [component]
  for i in range(truncation_cutoff):
    dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
    
  return dlatent

def interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

# Taken from https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=1, size=None, name='Items'):
    print()

# Circuit editing

class ParameterEditor():
  """Conveniently edit the parameters of a lucid model.
  Example usage:
    model = models.InceptionV1()
    param = ParameterEditor(model.graph_def)
    # Flip weights of first channel of conv2d0
    param["conv2d0_w", :, :, :, 0] *= -1
  """

  def __init__(self, graph_def):
    self.nodes = {}
    for node in graph_def.node:
      if "value" in node.attr:
        self.nodes[str(node.name)] = node
    # Set a flag to mark the fact that this is an edited model
    '''
    if not "lucid_is_edited" in self.nodes:
      with tf.Graph().as_default() as temp_graph:
        const = tf.constant(True, name="lucid_is_edited")
        const_node = temp_graph.as_graph_def().node[0]
        graph_def.node.extend([const_node])
    '''


  def __getitem__(self, key):
    name = key[0] if isinstance(key, tuple) else key
    tensor = self.nodes[name].attr["value"].tensor
    shape = [int(d.size) for d in tensor.tensor_shape.dim]
    array = np.frombuffer(tensor.tensor_content, dtype="float32").reshape(shape).copy()
    return array[key[1:]] if isinstance(key, tuple) else array

  def __setitem__(self, key, new_value):
    name = key[0] if isinstance(key, tuple) else key
    tensor = self.nodes[name].attr["value"].tensor
    node_shape = [int(d.size) for d in tensor.tensor_shape.dim]
    if isinstance(key, tuple):
      array = np.frombuffer(tensor.tensor_content, dtype="float32")
      array = array.reshape(node_shape).copy()
      array[key[1:]] = new_value
      tensor.tensor_content = array.tostring()
    else:
      assert new_value.shape == tuple(node_shape), "mismatch: new_value.shape %s != node_shape %s" % (str(new_value.shape), str((node_shape)))
      tensor.tensor_content = new_value.tostring()


import tensorflow as tf
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param

from lucid.modelzoo.vision_base import Model


class FrozenNetwork(Model):
    model_path = network_protobuf_path
    image_shape = [256, 256, 3]
    image_value_range = (0, 1)
    input_name = 'input_1'

network = FrozenNetwork()
network.load_graphdef()

params = ParameterEditor(network.graph_def)

kernel_name = 'Mixed_5c_Branch_3_b_1x1_conv/kernel'

w = params[kernel_name]
print(w.shape)
neuron = w[..., neuron_index]

print(np.histogram(neuron.flatten(), bins=10))

p = weight_edit_param

if weight_editor == "invert_top":
    neuron *= (1 - 2 * (neuron >= np.sort(neuron, axis=None)[-p]).astype(float))

elif weight_editor == "invert_one":
    neuron *= (1 - 2 * (neuron == np.sort(neuron, axis=None)[-p]).astype(float))

elif weight_editor == "prune_top":
    neuron -= neuron * (neuron >= np.sort(neuron, axis=None)[-p]).astype(float)

elif weight_editor == "prune_one":
    neuron -= neuron * (neuron == np.sort(neuron, axis=None)[-p]).astype(float)

else: print("no weight edit was made - invalid editor name")

w[..., neuron_index] = neuron
print(np.histogram(w[..., neuron_index].flatten(), bins=10))

params[kernel_name] = w

obj = objectives.channel(layer_name, neuron_index)
param_f = lambda: param.image(256, fft=True, decorrelate=True)
renders = render.render_vis(network, obj, param_f, thresholds=(2024,))


# Convert uploaded images to TFRecords
import dataset_tool
print(dataset_tool.__file__)
dataset_tool.create_from_images("./projection/records/", "./projection/imgs/", True)

# Run the projector
import run_projector
import projector
import dream_projector

import importlib
importlib.reload(dream_projector)

import training.dataset
import training.misc
import os 



def project_real_images(dataset_name, data_dir, num_images, num_snapshots):
    proj = projector.Projector()
    proj.set_network(Gs, layer_name, neuron_index)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = training.dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, verbose=True, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = training.misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        run_projector.project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('projection/out/image%04d-' % image_idx), num_snapshots=num_snapshots)


def dream_project(Gs, network, layer_name, neuron_index, png_prefix, num_snapshots):
    proj = dream_projector.DreamProjector()
    proj.set_network(Gs, network, layer_name, neuron_index)
    run_projector.dream_project(proj, png_prefix, num_snapshots)


# project_real_images("records","./projection", 1, 100)

dream_project(Gs, network, layer_name, neuron_index, png_prefix='projection/out/image-', num_snapshots=100)

# TODO 300 hardwired already in dream_projector
import shutil
shutil.copyfile('projection/out/image-step%04d.png' % 300, "projection/%s-%03d.png" % (fancy_layer_name, neuron_index))

# Create video 

import glob

imgs = sorted(glob.glob("projection/out/*step*.png"))

movieName = "projection/%s-%03d.mp4" % (fancy_layer_name, neuron_index)
with imageio.get_writer(movieName, mode='I') as writer:
    for filename in imgs:
        image = imageio.imread(filename)

        # Concatenate images with original target image
        w,h = image.shape[0:2]
        canvas = PIL.Image.new('RGBA', (w,h), 'white')
        canvas.paste(Image.fromarray(image), (0, 0))

        writer.append_data(np.array(canvas))


import tensorflow as tf
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param

from lucid.modelzoo.vision_base import Model

last_image_file = sorted(glob.glob("projection/out/*step*.png"))[-1]
stylegan_render = imageio.imread(last_image_file)
lucid_render = renders[0][0]
lucid_render = (np.clip(lucid_render, 0, 1) * 255).astype(np.uint8)

h, w = lucid_render.shape[:2]
canvas = PIL.Image.new('RGB', (w * 2, h), 'white')
canvas.paste(Image.fromarray(lucid_render), (0, 0))
canvas.paste(Image.fromarray(stylegan_render).resize((w, h), PIL.Image.LANCZOS), (w, 0))
canvas.save("projection/combined_%s_%03d.png" % (layer_name.split("/")[0], neuron_index))
