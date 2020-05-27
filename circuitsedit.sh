# wget https://www.renyi.hu/~daniel/tmp/lucid/200.pb

python main.py --network_protobuf_path=200.pb \
	--layer_name=Mixed_5c_Branch_3_b_1x1_act/Relu \
	--neuron_index=1 \
	--weight_editor=invert_top \
	--weight_edit_param=10
