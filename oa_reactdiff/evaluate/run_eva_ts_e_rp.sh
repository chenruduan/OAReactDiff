export CUDA_VISIBLE_DEVICES=1
export timesteps=150
export resamplings=10
export jump_length=10
export partition="valid"
export single_frag_only=0
export model="leftnet_2304"
export power="2.5"

save_path=nohupout/$model-$partition-timesteps-$timesteps-resamplings-$resamplings-single_frag_only-$single_frag_only-power-$power.out

nohup python -u evaluate_ts_w_rp.py \
    --timesteps $timesteps \
    --resamplings $resamplings \
    --jump_length $jump_length \
    --partition $partition \
    --single_frag_only $single_frag_only \
    --model $model \
    --power $power \
    > $save_path 2>&1 &