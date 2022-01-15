# for prototypes
CUDA_VISIBLE_DEVICE=0 python src/embeddings.py --wandb --output_dir ./run2
# for source only training (will save 3 weights netF, netB, netC)
CUDA_VISIBLE_DEVICE=0 python image_source_final.py --dataset OfficeHome --output weights --batch_size 128 --max_epoch 50 --source Ar,Pr --target Cl,Rw --wandb 0
# for computing centroids and saving TSNE plots
CUDA_VISIBLE_DEVICE=0 python warmup.py --dataset OfficeHome --source Ar,Pr --target Cl,Rw --trained_wt weights/uda/OfficeHome
# for adaptation
