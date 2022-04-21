# for prototypes
# CUDA_VISIBLE_DEVICE=0 python src/embeddings.py --wandb --output_dir ./run2
# for source only training (will save 3 weights netF, netB, netC)
# CUDA_VISIBLE_DEVICES=0 python image_source_final.py --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 100 --source Ar --target Cl --wandb 0
# for computing centroids and saving TSNE plots
CUDA_VISIBLE_DEVICES=0 python warmup.py --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome  --source Ar --target Cl
# for adaptation
CUDA_VISIBLE_DEVICES=0 python src/degaa_new.py --output_dir ./adapt/Ar_CL --dataset OfficeHome --source Ar --target Cl --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Ar->Pr
CUDA_VISIBLE_DEVICES=0 python image_source_final.py --source Ar --target Pr --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=0 python warmup.py --source Ar --target Pr --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=0 python src/degaa_new.py --source Ar --target Pr --output_dir ./adapt/Ar_Pr --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Ar->Rw
CUDA_VISIBLE_DEVICES=0 python image_source_final.py --source Ar --target Rw --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=0 python warmup.py --source Ar --target Rw --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=0 python src/degaa_new.py --source Ar --target Rw --output_dir ./adapt/Ar_Rw --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb

# Cl->Ar
CUDA_VISIBLE_DEVICES=0 python image_source_final.py --source Cl --target Ar --root data --output weights --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --max_epoch 50 --wandb 1
CUDA_VISIBLE_DEVICES=0 python warmup.py --source Cl --target Ar --root data --batch_size 64 --dataset OfficeHome --proto_path ./protoruns/run7/prototypes.pth --trained_wt ./weights/oda/OfficeHome
CUDA_VISIBLE_DEVICES=0 python src/degaa_new.py --source Cl --target Ar --output_dir ./adapt/Cl_Ar --dataset OfficeHome --batch_size 48 --epochs 100 --episodes 5 --trained_wt weights/oda --wandb