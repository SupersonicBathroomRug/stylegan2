# inceptionv1 network trained on the 40 binary outputs CelebA task.
wget https://www.renyi.hu/~daniel/tmp/lucid/200.pb

rm -f projection/out/*.png
time CUDA_VISIBLE_DEVICES=1 python main.py 200.pb Mixed_5c_Branch_3_b_1x1_act/Relu 100

nohup bash batch.sh > cout 2> cerr &

ssh renyi.hu mkdir www/tmp/lucid-stylegan2-full
scp -q projection/*.png renyi.hu:./www/tmp/lucid-stylegan2-full
ssh renyi.hu mkdir www/lucid-stylegan2
scp vis.html renyi.hu:./www/lucid-stylegan2/index.html
