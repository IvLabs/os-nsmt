#Office-31
# python image_source_final.py --output san --gpu_id 0 --dset office --max_epoch 50 --s 0 --net vit

#office-home
CUDA_VISIBLE_DEVICES=2 python image_source_final.py --root data --output weights --batch_size 128 --dataset OfficeHome --max_epoch 150 --source Ar,Pr --target Cl,Rw --wandb 0
# python image_source_final.py --output weights --gpu_id 0 --dataset OfficeHome --max_epoch 50 --source "Ar,Pr" --target "Cl,Rw"
# python image_source_final.py --output weights --gpu_id 1 --dataset OfficeHome --max_epoch 50 --source "Ar,Pr" --target "Cl,Rw"
# python image_source_final.py --output weights --gpu_id 0 --dataset OfficeHome --max_epoch 50 --source "Ar,Pr" --target "Cl,Rw"

#pacs
# python image_source_final.py --output san --gpu_id 0 --dset pacs --max_epoch 50 --s 0 --net vit

#domain_net
# python image_source_final.py --output san --gpu_id 0 --dset domain_net --max_epoch 50 --s 3 --net vit

#VisDA 2017
# python image_source_final.py --output san --gpu_id 0 --dset visda-2017 --max_epoch 50 --s 0 --net vit --batch_size 128
