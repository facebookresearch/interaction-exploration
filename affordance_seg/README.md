## Affordance segmentation model training

**Train an RGB policy** following the instructions in the root directory. 

**Collect a dataset of rollouts** using a pretrained interaction exploration agent. This will generate and store rollouts to `affordance_seg/data/rgb`.
```
bash affordance_seg/tools/collect_dataset.sh
```

**Train the affordance segmentation UNet** on the dataset of rollouts. This will create a checkpoint in `affordance_seg/cv/rgb_unet/`
```
python -m affordance_seg.train_unet \
       --data_dir affordance_seg/data/rgb/ \
       --cv_dir affordance_seg/cv/rgb_unet \
       --train
```

Note: Pretrained models and precomputed datasets are provided with the repo. See `data/download_data.sh`. These can be used to train interaction exploration policies with affordance maps.
