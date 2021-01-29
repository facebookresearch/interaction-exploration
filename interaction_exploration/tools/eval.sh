# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

evaluate () {

    CONFIG=$1
    RUN=$2
    CHECKPOINT_FOLDER=$3
    shift 3

    python -m interaction_exploration.run \
        --config $CONFIG \
        --mode eval \
        ENV.ENV_NAME ThorInteractionCount-v0 \
        TORCH_GPU_ID $RUN X_DISPLAY :$RUN \
        ENV.NUM_STEPS 1024 \
        NUM_PROCESSES 32 \
        CHECKPOINT_FOLDER ${CHECKPOINT_FOLDER} \
        EVAL.DATASET interaction_exploration/data/test_episodes_K_16.json \
        $@ &

}

# Evaluate 3 runs in parallel (on GPU 0,1,2)

# Random
CVDIR=interaction_exploration/cv/random/
for i in 0 1 2; do
    evaluate interaction_exploration/config/random.yaml \
            $i \
            $CVDIR/run$i
done

# IntExp (RGB)
CVDIR=interaction_exploration/cv/rgb/
for i in 0 1 2; do
    evaluate interaction_exploration/config/rgb.yaml \
            $i \
            $CVDIR/run$i \
            LOAD ${CVDIR}/run$i/ckpt.24.pth
done

# Navigation Novelty
CVDIR=interaction_exploration/cv/nav-novelty/
for i in 0 1 2; do
    evaluate interaction_exploration/config/nav-novelty.yaml \
            $i \
            $CVDIR/run$i \
            LOAD ${CVDIR}/run$i/ckpt.24.pth \
            ENV.ENV_NAME ThorInteractionCyclerFixedView-v0
done

# Object Coverage
CVDIR=interaction_exploration/cv/obj-coverage/
for i in 0 1 2; do
    evaluate interaction_exploration/config/obj-coverage.yaml \
            $i \
            $CVDIR/run$i \
            LOAD ${CVDIR}/run$i/ckpt.24.pth \
            ENV.ENV_NAME ThorInteractionCycler-v0
done

# IntExp (Fixed Scale)
CVDIR=interaction_exploration/cv/intexp/
for i in 0 1 2; do
    UNETCKPT=`ls $CVDIR/run$i/unet/*.ckpt`
    evaluate interaction_exploration/config/intexp_fixedscale.yaml \
            $i \
            $CVDIR/run$i \
            LOAD ${CVDIR}/run$i/ckpt.24.pth \
            MODEL.BEACON_MODEL $UNETCKPT
done
