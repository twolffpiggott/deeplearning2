import importlib
import utils2; importlib.reload(utils2)
from utils2 import *

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg

# restrict tf to use only the necessary GPU RAM
limit_mem()

# set paths to imagenet samples
path = '/home/tim/Github/deeplearning2/sample/'

fnames = glob.glob(path+'**/*.JPEG', recursive=True)
n = len(fnames)
fn = fnames[74]
print(n, fn)

img=Image.open(fnames[400]); img.show()

# Imagenet preprocessing: subtract the mean of each channel and reverse the RGB order
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
deproc = lambda x, s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)
img_arr = preproc(np.expand_dims(np.array(img), 0))
shp = img_arr.shape

# include_top includes the final classification block
model = VGG16_Avg(include_top=False) 

# get activations from near the end of the convolutional model
layer = model.get_layer('block5_conv1').output

# calculate target activations for chosen layer
layer_model = Model(model.input, layer)
targ = K.variable(layer_model.predict(img_arr))

# define an object for separate access of loss functions and gradients
class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

# define loss to calculate MSE between the two outputs at the specified convolutional layer
loss = K.sum(metrics.mse(layer, targ))
grads = K.gradients(loss, model.input)
fn = K.function([model.input], [loss]+grads)
evaluator = Evaluator(fn, shp)

# deterministic optimisation using a line search via sklean fmin_l_bfgs_b
if not os.path.exists('results'):
    os.makedirs('results')
def solve_image(eval_obj, niter, x):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127, 127)
        print('Current loss value: ', min_val)
        imsave(f'results/res_at_iteration_{i}.png', deproc(x.copy(), shp)[0])
    return x

# generate a random image
rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape) / 100
x = rand_img(shp)
plt.imshow(x[0])

# run deterministic optimisation approach
x = solve_image(evaluator, niter=10, x=x)

# check image
Image.open('results/res_at_iteration_9.png')

                    
