import importlib
import utils2; importlib.reload(utils2)
from utils2 import *

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics

from vgg16_avg import VGG16_Avg

# restrict tf to use only the necessary GPU RAM
limit_mem()

# define an object for separate access of loss functions and gradients
class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

# define a gram matrix function for highly effective style loss
def gram_matrix(x):
    # each row is a channel and the columns are flattened x, y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # the dot product of this vector with its transpose gives the correlation
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

def style_loss(x, targ):
    return K.sum(metrics.mse(gram_matrix(x), gram_matrix(targ)))


class StyleTransfer:
    
    def __init__(self, image): 
        self.image = image
        # Imagenet preprocessing: subtract the mean of each channel and reverse the RGB order
        self.rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        self.preproc = lambda x: (x - self.rn_mean)[:, :, :, ::-1]
        self.deproc = lambda x, s: np.clip(x.reshape(s)[:, :, :, ::-1] + self.rn_mean, 0, 255)
        self.img_arr = self.preproc(np.expand_dims(np.array(self.image), 0))
        self.shp = self.img_arr.shape
        
    def recreate_from_noise(self, niter=10):
        # include_top includes the final classification block
        model = VGG16_Avg(include_top=False) 
        # get activations from near the end of the convolutional model
        layer = model.get_layer('block5_conv1').output
        # calculate target activations for chosen layer
        layer_model = Model(model.input, layer)
        targ = K.variable(layer_model.predict(self.img_arr))
        # define loss to calculate MSE between the two outputs at the specified convolutional layer
        loss = K.sum(metrics.mse(layer, targ))
        grads = K.gradients(loss, model.input)
        fn = K.function([model.input], [loss]+grads)
        evaluator = Evaluator(fn, self.shp)
        # deterministic optimisation using a line search via sklean fmin_l_bfgs_b
        if not os.path.exists('results'):
            os.makedirs('results')
        def solve_image(eval_obj, niter, x):
            for i in range(niter):
                x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                                 fprime=eval_obj.grads, maxfun=20)
                x = np.clip(x, -127, 127)
                print('Current loss value: ', min_val)
                imsave(f'results/res_at_iteration_{i}.png', self.deproc(x.copy(), self.shp)[0])
            return x
        # generate a random image
        rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape) / 100
        x = rand_img(self.shp)
        # run deterministic optimisation approach
        x = solve_image(evaluator, niter=niter, x=x)

    def recreate_style(self, style_img, niter=10):
        self.style = Image.open(style_img)
        self.style_arr = self.preproc(np.expand_dims(self.style, 0)[:, :, :, :3])
        self.style_shp = self.style_arr.shape
        model = VGG16_Avg(include_top=False, input_shape=self.style_shp[1:])
        outputs = {l.name: l.output for l in model.layers}
        layers = [outputs[f'block{o}_conv1'] for o in range(1, 3)]
        layers_model = Model(model.input, layers)
        targs = [K.variable(o) for o in layers_model.predict(self.style_arr)]
        loss = sum(style_loss(l1[0], l2[0]) for l1, l2 in zip(layers, targs))
        grads = K.gradients(loss, model.input)
        style_fn = K.function([model.input], [loss] + grads)
        evaluator = Evaluator(style_fn, self.style_shp)
        rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)
        x = rand_img(self.style_shp)
        x = scipy.ndimage.filters.gaussian_filter(x, [0, 2, 2, 0])
        x = rand_img(self.style_shp)
        def solve_image(eval_obj, niter, x):
            for i in range(niter):
                x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                                 fprime=eval_obj.grads, maxfun=20)
                x = np.clip(x, -127, 127)
                print('Current loss value: ', min_val)
                imsave(f'results/res_at_iteration_{i}.png', self.deproc(x.copy(), self.style_shp)[0])
            return x
        x = solve_image(evaluator, niter, x)
        
        
        
        
        
