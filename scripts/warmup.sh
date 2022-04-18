# CUDA_VISIBLE_DEVICES=2 python3 warmup.py --root data/ --dataset OfficeHome -s 'Ar,Pr' -t 'Cl,Rw' --trained_wt weights/uda/OfficeHome --wandb 0 --batch_size 16 --net resnet50

CUDA_VISIBLE_DEVICES=2 python warmup.py --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome