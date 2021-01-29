# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

for GPU in 0 1 2 3; do
    python -m affordance_seg.collect_dset \
         --out-dir affordance_seg/data/rgb/ \
         NUM_PROCESSES 16 \
         LOAD interaction_exploration/cv/rgb/run2/ckpt.24.pth \
         EVAL.DATASET affordance_seg/data/episode_splits/episodes_K_256_split_$GPU.json \
         ENV.NUM_STEPS 256 \
         TORCH_GPU_ID $GPU \
         X_DISPLAY :$GPU \
         ENV.ENV_NAME ThorBeaconsFixedScale-v0 \
         MODE eval &
done

