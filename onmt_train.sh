data_dir="./data/*"

export CUDA_VISIBLE_DEVICES=0

steps=$3
batch_size=$4
save_at=$5

if [ -n $steps ]
then
    steps=200000
fi

if [ -n $batch_size ]
then
    batch_size=4096
fi

if [ -n $save_at ]
then
    save_at=10000
fi

onmt_train -data $1 -save_model $2 \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps $steps  -max_generator_batches 2 -dropout 0.1 \
        -batch_size $batch_size -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps $save_at -save_checkpoint_steps $save_at \
        -world_size 1 -gpu_ranks 0
