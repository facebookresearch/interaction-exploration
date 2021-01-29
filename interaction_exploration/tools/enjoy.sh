# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

CONFIG=$1
CVDIR=$2
shift 2

UNETCKPT="N/A"
if [ -d "$CVDIR/unet/" ]; then
	UNETCKPT=`ls $CVDIR/unet/*.ckpt`
fi
python -m interaction_exploration.run \
    --config $CONFIG \
    --mode enjoy \
    CHECKPOINT_FOLDER interaction_exploration/cv/tmp \
    TORCH_GPU_ID 0 \
    X_DISPLAY :0 \
    LOAD ${CVDIR}/ckpt.24.pth \
    MODEL.BEACON_MODEL $UNETCKPT \
    $@