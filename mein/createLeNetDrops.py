import subprocess
import numpy as np


clean_ckpt = "LeNet/model.ckpt-20000" 
ckpt_prefix = "droppedLeNet/droppedLeNet"
layer = "LeNet/fc4/weights"
percentages = np.arange(0, 1.01, 0.02)

for perc in percentages:
    command = "CUDA_VISIBLE_DEVICES= python drop.py"
    command = command + " --file_name=" + clean_ckpt
    command = command + " --out_file=" + ckpt_prefix + "-fc4" + ("-%.2f" % perc)
    command = command + " --drop_layer=" + layer
    command = command + " --drop_percentage=" + ("%.2f" % perc)

    print (command)
    #output = subprocess.call(command)

#CUDA_VISIBLE_DEVICES=0 python drop.py --file_name=VGG16/vgg_16.ckpt --out_file=droppedVGG/droppedVGG--0.05 --drop_layer=InceptionV3/Conv2d_4a_3x3/weights --drop_percentage=0.05
