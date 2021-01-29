# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

train () {

    CONFIG=$1
    RUN=$2
    CHECKPOINT_FOLDER=$3
    shift 3

    python -m interaction_exploration.run \
        --config $CONFIG \
        --mode train \
        TORCH_GPU_ID $RUN X_DISPLAY :$RUN SEED $RUN \
        ENV.NUM_STEPS 256 \
        NUM_PROCESSES 16 \
        CHECKPOINT_FOLDER ${CHECKPOINT_FOLDER} \
        $@ &
}

# Train 3 runs in parallel (on GPU 0,1,2) with different seeds

# IntExp (RGB)
CVDIR=interaction_exploration/cv/rgb/
for i in 0 1 2; do
    train interaction_exploration/config/rgb.yaml \
            $i \
            $CVDIR/run$i
done

# Navigation Novelty
CVDIR=interaction_exploration/cv/nav-novelty/
for i in 0 1 2; do
    train interaction_exploration/config/nav-novelty.yaml \
            $i \
            $CVDIR/run$i
done

# Object Coverage
CVDIR=interaction_exploration/cv/obj-coverage/
for i in 0 1 2; do
    train interaction_exploration/config/obj-coverage.yaml \
            $i \
            $CVDIR/run$i
done

# IntExp (Fixed Scale)
CVDIR=interaction_exploration/cv/intexp/
for i in 0 1 2; do
    train interaction_exploration/config/intexp.yaml \
            $i \
            $CVDIR/run$i \
            MODEL.BEACON_MODEL affordance_seg/cv/rgb_unet/epoch=24-val_loss=0.6995.ckpt
done