import subprocess
import numpy as np


ckpt_prefix = "droppedInception3/droppedInception3-conv2d_4a_3x3-"
layer = "vgg_16/conv3/conv3_3/weights"
percentages = np.arange(0.50, 1.01, 0.02)

for perc in percentages:
    command = "CUDA_VISIBLE_DEVICES=0 python /data/ramyadML/scratch/tensorflow-models/research/slim/eval_image_classifier.py --checkpoint_path=/data/ramyadML/TF-slim-checkpoints/"
    command = command + ckpt_prefix + ("%.2f" % perc)
    command = command + " --dataset_dir=/data/ramyadML/TF-slim-data/imageNet/processed --dataset_name=imagenet --dataset_split_name=validation"

    print (command)

#python eval_image_classifier.py     --alsologtostderr     --checkpoint_path=/data/ramyadML/TF-slim-checkpoints/droppedInception3/Dropped-Inception-0.25     --dataset_dir=/data/ramyadML/TF-slim-data/imageNet/processed     --dataset_name=imagenet     --dataset_split_name=validation

#CUDA_VISIBLE_DEVICES=0 python drop.py --file_name=VGG16/vgg_16.ckpt --out_file=droppedVGG/droppedVGG--0.05 --drop_layer=InceptionV3/Conv2d_4a_3x3/weights --drop_percentage=0.05
