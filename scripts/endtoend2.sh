# Pr->Rw
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --source Pr --target Rw --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=2 python warmup.py --source Pr --target Rw --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=2 python src/degaa_new.py --source Pr --target Rw --output_dir ./adapt/Pr_Rw --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Rw->Ar
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --source Rw --target Ar --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=2 python warmup.py --source Rw --target Ar --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=2 python src/degaa_new.py --source Rw --target Ar --output_dir ./adapt/Rw_Ar --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Rw->Pr
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --source Rw --target Pr --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=2 python warmup.py --source Rw --target Pr --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=2 python src/degaa_new.py --source Rw --target Pr --output_dir ./adapt/Rw_Pr --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Rw->Cl
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --source Rw --target Cl --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=2 python warmup.py --source Rw --target Cl --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=2 python src/degaa_new.py --source Rw --target Cl --output_dir ./adapt/Rw_Cl --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb