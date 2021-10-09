#Office-31
# python image_source_final.py --output san --gpu_id 0 --dset office --max_epoch 50 --s 0 --net vit

#office-home
python image_source_final.py --output weights --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --net resnet50
python image_source_final.py --output weights --gpu_id 0 --dset office-home --max_epoch 50 --s 1 --net resnet50
python image_source_final.py --output weights --gpu_id 1 --dset office-home --max_epoch 50 --s 2 --net resnet50
python image_source_final.py --output weights --gpu_id 0 --dset office-home --max_epoch 50 --s 3 --net resnet50

#pacs
# python image_source_final.py --output san --gpu_id 0 --dset pacs --max_epoch 50 --s 0 --net vit

#domain_net
# python image_source_final.py --output san --gpu_id 0 --dset domain_net --max_epoch 50 --s 3 --net vit

#VisDA 2017
# python image_source_final.py --output san --gpu_id 0 --dset visda-2017 --max_epoch 50 --s 0 --net vit --batch_size 128
