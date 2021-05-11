# Learning Affordance Landscapes for Interaction Exploration in 3D Environments

![Teaser](http://vision.cs.utexas.edu/projects/interaction-exploration/media/concept.png)

This repo contains code to train and evaluate *interaction exploration* agents that can discover and explore new interactions with object in their environment, while simultaneously building visual affordance models for their environment via exploration. See our [project page](http://vision.cs.utexas.edu/projects/interaction-exploration/) for more details and our spotlight video.

This is the code accompanying our NeurIPS20 (spotlight) work:  
**Learning Affordance Landscapes for Interaction Exploration in 3D Environments**  
*Tushar Nagarajan and Kristen Grauman.*  
[[arxiv]](https://arxiv.org/pdf/2008.09241.pdf) [[project page]](http://vision.cs.utexas.edu/projects/interaction-exploration/)

## Requirements
Install required packages:
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

Install AI2-Thor
```
pip install ai2thor==2.3.8
```

Download the simulator files (one-time download when THOR is first run) and test the simulator with a simple keyboard controlled agent.
```
python kb_agent.py --x_display 0 --env-name ThorObjs-v0
```

Note: an X server is required to run the simulator on each GPU used for training. 
Code tested with Python 3.7, Pytorch 1.6, cuda 10.1

## Data

**Download precomputed data and model checkpoints** 
```
bash interaction_exploration/tools/download_data.sh
```
This will download model checkpoints for all policies trained for interaction exploration and a precomputed copy of the affordance dataset and model that is be generated with a trained policy as part of interaction exploration. The dataset/models may be generated/trained using our code, but is provided here for convenience as well.


## Training interaction exploration agents

To train a simple RGB agent on GPU 0, Display :0
```
python -m interaction_exploration.run \
    --config interaction_exploration/config/rgb.yaml \
    --mode train \
    SEED 0 TORCH_GPU_ID 0 X_DISPLAY :0 \
    ENV.NUM_STEPS 256 \
    NUM_PROCESSES 16 \
    CHECKPOINT_FOLDER interaction_exploration/cv/rgb/run0/ \

```

To train our interaction exploration agents that use a trained affordance module
```
python -m interaction_exploration.run \
    --config interaction_exploration/config/intexp.yaml \
    --mode train \
    SEED 0 TORCH_GPU_ID 0 X_DISPLAY :0 \
    ENV.NUM_STEPS 256 \
    NUM_PROCESSES 16 \
    CHECKPOINT_FOLDER interaction_exploration/cv/intexp/run0/ \
    MODEL.BEACON_MODEL affordance_seg/cv/rgb_unet/epoch=39-val_loss=0.6737.ckpt
```
Note: The segmentation model must be trained first. See the [README](affordance_seg/README.md) in `affordance_segmentation/`. tl;dr First train the baseline RGB policy, then extract a dataset using rollouts from the policy, then train the segmentation model.

To train some or all models 3 times with different seeds please see and edit the `train.sh` script. Then run:
```
bash interaction_exploration/tools/train.sh
```
This will save checkpoints to `cv/{method}/run{idx}` for each method and for 3 uniquely seeded training runs.

## Evaluation

To evaluate an interaction exploration model 
```
export UNETCKPT=`ls interaction_exploration/cv/intexp/run0/unet/*.ckpt`
python -m interaction_exploration.run \
    --config interaction_exploration/config/intexp.yaml \
    --mode eval \
    ENV.NUM_STEPS 1024 \
    NUM_PROCESSES 32 \
    EVAL.DATASET interaction_exploration/data/test_episodes_K_16.json \
    TORCH_GPU_ID 0 X_DISPLAY :0 \
    CHECKPOINT_FOLDER interaction_exploration/cv/intexp/run0/ \
    LOAD interaction_exploration/cv/intexp/run0/ckpt.24.pth \
    MODEL.BEACON_MODEL $UNETCKPT
```

To evaluate some or all trained models please see and edit the `eval.sh` script. Then run:
```
bash interaction_exploration/tools/eval.sh
```

Once the policy rollouts and rewards are generated, some or all model results can be visualized using 
```
python -m interaction_exploration.tools.plot_results --cv_dir interaction_exploration/cv/ --models random rgb nav-novelty obj-coverage intexp
```

This should result in a curve similar to the one below, equivalent to Fig. 3 in the paper.
<img src="http://vision.cs.utexas.edu/projects/interaction-exploration/media/results_github.png" height="256">

## Policy visualizations

To see a trained policy in action
```
bash interaction_exploration/tools/enjoy.sh <config> <checkpoint_dir>

# e.g. for interaction exploration
bash interaction_exploration/tools/enjoy.sh \
    interaction_exploration/config/intexp.yaml \
    interaction_exploration/cv/intexp/run0
```

Policy visualizations will show the current egocentric view (left), and a topdown view (right) showing successful and unsuccessful interaction attempts as green and yellow dots respectively.



## License

This project is released under the CC-BY-NC 4.0 license, as found in the LICENSE file.

## Cite

If you find this repository useful in your own research, please consider citing:
```
@inproceedings{interaction-exploration,
    author = {Nagarajan, Tushar and Grauman, Kristen},
    title = {Learning Affordance Landscapes for Interaction Exploration in 3D Environments},
    booktitle = {NeurIPS},
    year = {2020}
}
```
