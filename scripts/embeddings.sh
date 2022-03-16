python src/embeddings.py --output_dir ./protoruns/run7 --batch_size 64 --num_proto_steps 100000 --wandb

# CUDA_VISIBLE_DEVICES=1 python src/embeddings_old.py --output_dir ./protoruns/run5 --batch_size 32 --hparams '{"mixup":1}' --num_proto_steps 1000000 --tensorboard --resume 400100 --checkpoint_freq 10000
