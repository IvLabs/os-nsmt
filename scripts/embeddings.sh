# CUDA_VISIBLE_DEVICES=0 python src/embeddings_old.py --output_dir ./protoruns/run4 --batch_size 48 --num_proto_steps 100000 --tensorboard --resume 60000

CUDA_VISIBLE_DEVICES=1 python src/embeddings_old.py --output_dir ./protoruns/run5 --batch_size 32 --hparams '{"mixup":1}' --num_proto_steps 1000000 --tensorboard --resume 400100 --checkpoint_freq 10000
