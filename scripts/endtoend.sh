# for prototypes
CUDA_VISIBLE_DEVICE=0 python src/embeddings.py --wandb --output_dir ./run2
# for warmup (will save 3 weights netF, netB, netC)
python image_source_final.py --gpu_id 2  --dataset OfficeHome --output weights --batch_size 128 --max_epoch 50 --source Ar,Pr --target Cl,Rw --wandb 0

# for adaptation