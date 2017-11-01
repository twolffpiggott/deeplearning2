# deeplearning2
Fast AI Deep Learning 2 Implementations

## Style Transfer
```python
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg
import neural_style

# restrict tf to use only the necessary GPU RAM
limit_mem()

# set paths to imagenet samples and grab the image paths
path = '/home/tim/Github/deeplearning2/sample/'
fnames = glob.glob(path+'**/*.JPEG', recursive=True)

# choose an image
img=Image.open(fnames[1000]); img.show()

# construct style transfer object for given image
simple_style = neural_style.StyleTransfer(img)
# reconstruct image from noise
simple_style.recreate_from_noise()
```
