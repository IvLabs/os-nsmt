CUDA_VISIBLE_DEVICES=0 python src/degaa_new.py --output_dir ./adapt/run5 --dataset OfficeHome --source Ar,Pr --target Cl,Rw --batch_size 24 --epochs 100 --episodes 1 --trained_wt weights_final/oda
