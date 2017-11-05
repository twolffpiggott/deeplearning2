# deeplearning2
Fast AI Deep Learning 2 Implementations

## Style Transfer
```python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', type=bool, default=False,
                    help='use GPU for tf model')
parser.add_argument('--limit_mem', type=bool, default=False,
                    help='limit tensorflow memory usage to only necessary')
parser.add_argument('--show_original', type=bool, default=False,
                    help='show original image')

args = parser.parse_args()

import os
if args.use_gpu == False:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_DEVICE_ORDER']= 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg
import neural_style

# restrict tf to use only the necessary GPU RAM
if args.limit_mem:
    limit_mem()

# set paths to imagenet samples and grab the image paths
path = '/home/tim/Github/deeplearning2/sample/'
fnames = glob.glob(path+'**/*.JPEG', recursive=True)

# choose an image
img=Image.open(fnames[1000])
if args.show_original: img.show()

# construct style transfer object for given image
simple_style = neural_style.StyleTransfer(img)

# reconstruct image from noise
simple_style.recreate_image_from_noise()

# reconstruct style from noise
simple_style.recreate_style_from_noise('data/wave.jpg')

# transfer style
simple_style.transfer_style('data/wave.jpg')

```
