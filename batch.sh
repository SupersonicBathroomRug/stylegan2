
for (( i=0; i<128; ++i ))
do
    rm -f projection/out/*.png
    time CUDA_VISIBLE_DEVICES=1 python main.py 200.pb Mixed_5c_Branch_3_b_1x1_act/Relu $i
done