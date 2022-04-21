# Cl->Pr
CUDA_VISIBLE_DEVICES=1 python image_source_final.py --source Cl --target Pr --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=1 python warmup.py --source Cl --target Pr --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=1 python src/degaa_new.py --source Cl --target Pr --output_dir ./adapt/Cl_Pr --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Cl->Rw
CUDA_VISIBLE_DEVICES=1 python image_source_final.py --source Cl --target Rw --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=1 python warmup.py --source Cl --target Rw --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=1 python src/degaa_new.py --source Cl --target Rw --output_dir ./adapt/Cl_Rw --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Pr->Ar
CUDA_VISIBLE_DEVICES=1 python image_source_final.py --source Pr --target Ar --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=1 python warmup.py --source Pr --target Ar --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=1 python src/degaa_new.py --source Pr --target Ar --output_dir ./adapt/Pr_Ar --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Pr->Cl
CUDA_VISIBLE_DEVICES=1 python image_source_final.py --source Pr --target Cl --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=1 python warmup.py --source Pr --target Cl --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=1 python src/degaa_new.py --source Pr --target Cl --output_dir ./adapt/Pr_Cl --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb