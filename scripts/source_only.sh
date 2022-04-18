#Office-31
# python image_source_final.py --output san --gpu_id 0 --dset office --max_epoch 50 --s 0 --net vit

#office-home
# CUDA_VISIBLE_DEVICES=2 python image_source_final.py --root data --output weights --batch_size 128 --dataset OfficeHome --max_epoch 150 --source Ar,Pr --target Cl,Rw --wandb 0

# Source: Ar,Pr
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 500 --source Ar,Pr --target Cl,Rw --wandb 0

# Source Cl,Rw
# CUDA_VISIBLE_DEVICES=1 python image_source_final.py --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 250 --source Cl,Rw --target Ar,Pr --wandb 1

