CUDA_VISIBLE_DEVICES=1 python src/degaa_new.py --output_dir ./adapt/run3 --dataset OfficeHome --source Ar,Pr --target Cl,Rw --batch_size 32 --epochs 100 --trained_wt weights/oda --wandb
