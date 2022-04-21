# Open-Set Multi-Source Multi-Target Domain Adaptation
## How to run

### Dataset directory
<details>
  <summary>Click to see full directory tree</summary>

```
   data
    ├── domain_net
    │   ├── clipart
    │   ├── clipart.txt
    │   ├── infograph
    │   ├── infograph.txt
    │   ├── painting
    │   ├── painting.txt
    │   ├── quickdraw
    │   ├── quickdraw.txt
    │   ├── real
    │   ├── real.txt
    │   ├── sketch
    │   └── sketch.txt
    ├── office
    │   ├── amazon
    │   ├── amazon.txt
    │   ├── dslr
    │   ├── dslr.txt
    │   ├── webcam
    │   └── webcam.txt
    ├── office-home
    │   ├── Art
    │   ├── Art.txt
    │   ├── Clipart
    │   ├── Clipart.txt
    │   ├── Product
    │   ├── Product.txt
    │   ├── Real_World
    │   └── RealWorld.txt
    ├── office_home_mixed
    │   ├── Art_Clipart_Product
    │   ├── Art_Clipart_Product.txt
    │   ├── Art_Clipart_Real_World
    │   ├── Art_Clipart_Real_World.txt
    │   ├── Art_Product_Real_World
    │   ├── Art_Product_Real_World.txt
    │   ├── Clipart_Product_Real_World
    │   └── Clipart_Product_Real_World.txt
    └── pacs
        ├── art_painting
        ├── art_painting.txt
        ├── cartoon
        ├── cartoon.txt
        ├── __MACOSX
        ├── photo
        ├── photo.txt
        ├── sketch
        └── sketch.txt
```
</details>

Install the dependencies and run scripts.
### Prerequisites:

- See [requirements.txt](requirements.txt)
- Install dependencies using `pip3 install -r requirements.txt`

### Prepare pretrain model
We choose R50-ViT-B_16 as our backbone.
```sh class:"lineNo"
# Download pretrained R50-ViT-B_16
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir -p ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

### Download the dataset

- Run `sh source_only.sh` 
- Suppose you want to donwload Office-Home, just uncomment any line of Office-Home source training and it will download

### CODE
- Run `sh source_only.sh` for multi-src training. 
- Run `scripts/warmup.sh` for calculating centroids
- Run `scripts/adapt.sh` for adaptation

### Codes

1. [degaa_new.py](./src/degaa_new.py)

    * Passes whole data through LOF not batchwise
    * The LOF is applied every kth epoch of training (k=20)
    * Output Dir: [./adapt/run2](./adapt/run2/)
    * Pseudo Code:
      ```py
      for epoch in epochs:

        if epoch % num_episodes = 0:
          for i in iters_per_epoch:
            Feat_S = concat(Feat_S, netB(netF(batch_s) || DE))
            Feat_T = concat(Feat_T, netB(netF(batch_t) || DE))
          
          Y_preds = LOF(Feat_T) # Array of [1 and -1]
          Label_T.where( Y_preds == -1 ) = 25 

          known = Feat_T.where(Y_preds == 1)
          Label_T.where( Y_preds == 1) = KNN(known, centroids)

          TempData = [Feat_s, Label_S, Feat_T, Label_T]

        for batch in TempData:
          F_s, F_t = GAA(Feat_S, Feat_T)
          Y_s, Y_t = softmax(netC(F_s)), softmax(netC(F_t))

          loss = CE(Y_s, Label_s) + CE(Y_t, Label_t)

          loss.backward()
      ```

This code is taken from [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library). This will be boiler-plate code. Do checkout some of the [examples](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation).