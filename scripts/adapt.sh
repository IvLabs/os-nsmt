CUDA_VISIBLE_DEVICES=1 python src/degaa.py --output_dir ./adapt/run1 --tensorboard --dataset OfficeHome --source Ar,Pr --target Cl,Rw --batch_size 32 --trained_wt weights/uda
