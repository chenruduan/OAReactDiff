export CUDA_VISIBLE_DEVICES=$1
export timesteps=150
export resamplings=15
export jump_length=15
export dataset="transition1x"
export partition="train_addprop"
export single_frag_only=0
export model="leftnet_2074"
export power="2"
export position_key="positions"

save_path=nohupout/conf-$model-$dataset-$partition-timesteps-$timesteps-resamplings-$resamplings-single_frag_only-$single_frag_only-power-$power.out
echo "save path: " $save_path

nohup python -u generate_confidence_sample.py \
    --timesteps $timesteps \
    --resamplings $resamplings \
    --jump_length $jump_length \
    --partition $partition \
    --dataset $dataset \
    --single_frag_only $single_frag_only \
    --model $model \
    --power $power \
    --position_key $position_key \
    > $save_path 2>&1 &
