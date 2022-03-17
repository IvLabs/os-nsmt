CUDA_VISIBLE_DEVICES=2 python src/degaa.py --output_dir ./adapt/run1 --dataset OfficeHome --source Ar,Pr --target Cl,Rw --batch_size 64 --trained_wt weights/oda --wandb
