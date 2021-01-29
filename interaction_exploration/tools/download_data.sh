# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Download all precomputed data and models from the public S3 bucket
BUCKET=https://dl.fbaipublicfiles.com/interaction-exploration
function dls3_unzip {
	mkdir -p `dirname $1`
	wget -O $1 $BUCKET/$1
	unzip $1 -d `dirname $1`
	rm $1
}

# download pretrained affordance model
dls3_unzip affordance_seg/cv.zip
echo 'Downloaded pretrained affordance models'

# # [OPTIONAL] download precomputed affordance segmentation data to train the affordance model
# dls3_unzip affordance_seg/data.zip
# dls3_unzip affordance_seg/rollouts.zip
# echo 'Downloaded precomputed affordance data'

# download all pretrained interaction exploration models
dls3_unzip interaction_exploration/cv.zip
echo 'Downloaded pretrained policy checkpoints'


echo '[NOTE] Uncomment lines in script to optionally download affordance training data'