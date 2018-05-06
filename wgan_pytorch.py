from dcgan import DCGAN_D, DCGAN_G
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from decorators import timeit
from tqdm import tqdm
from torch import optim, FloatTensor as FT
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import dcgan
import torch


bs,sz,nz = 64, 64, 100

PATH = 'data/cifar10/'
data = datasets.CIFAR10(root=PATH, download=True,
                        transform=transforms.Compose([
                            transforms.Scale(sz),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)), ]))

dataloader = DataLoader(data, bs, True, num_workers=8)
n = len(dataloader)
print(n)


def save(img, fname, dpi, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imsave(fname, arr=np.transpose((img/2+0.5).clamp(0, 1).numpy(), (1, 2, 0)),
               interpolation='nearest', dpi=dpi)

def save(img, fname, dpi, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose((img/2+0.5).clamp(0,1).numpy(), (1,2,0)), interpolation='nearest')
    plt.savefig(fname, dpi=dpi)

def weights_init(mod):
    """
    Initialize all the weights in a pytorch model.

    :param mod: pytorch module for application of weight initialization
    """
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        mod.weight.data.normal_(0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


generator = DCGAN_G(sz, nz, 3, 64, 1, 1).cuda()
generator.apply(weights_init)
discriminator = DCGAN_D(sz, 3, 64, 1, 1).cuda()
discriminator.apply(weights_init)


def Var(*params):
    return Variable(FT(*params).cuda())


def create_noise(b):
    return Variable(FT(b, nz, 1, 1).cuda().normal_(0, 1))

# input placeholder
inpt = Var(bs, 3, sz, nz)
# fixed noise used to visualize images after training
fixed_noise = create_noise(bs)
# define the numbers 0 and -1
one = torch.FloatTensor([1]).cuda()
neg_one = one * -1

# define the generator and discriminator optimizers
generator_optim = optim.RMSprop(generator.parameters(), lr=1e-4)
discriminator_optim = optim.RMSprop(discriminator.parameters(), lr=1e-4)


def discriminator_step(v, init_grad):
    """
    Feed a candidate image (real or fake) to the discriminator and compute the
    gradients with respect to the loss.

    :param v: candidate image for discriminator
    :param init_grad:
    """
    err = discriminator(v)
    err.backward(init_grad)
    return err


def make_trainable(net, val):
    """
    Toggle trainanility for part of the network (either generator or
    discriminator).

    :param net: model to toggle
    :param val: bool trainability
    """
    for p in net.parameters():
        p.requires_grad = val


tqdm.monitor_interval = 0
@timeit
def train(niter, first=True):
    """
    Training loop for the GAN.

    :param niter: number of epochs
    :param first: enable increased training frequency for discriminator during
                  the early part of training
    """
    gen_iterations = 0
    for epoch in range(niter):
        data_iter = iter(dataloader)
        i = 0
        with tqdm(range(n), desc=f'[epoch: {epoch}/{niter}, '
                  f'generator iters: {gen_iterations}]', leave=False) as pbar:
            while i < n:
                make_trainable(discriminator, True)
                d_iters = (100 if first and (gen_iterations < 25) or
                        (gen_iterations % 500) == 0 else 5)
                j = 0
                while j < d_iters and i < n:
                    j += 1
                    i += 1
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    real = Variable(next(data_iter)[0].cuda())
                    discriminator.zero_grad()
                    d_err_real = discriminator_step(real, one)

                    fake = generator(create_noise(real.size()[0]))
                    inpt.data.resize_(real.size()).copy_(fake.data)
                    d_err_fake = discriminator_step(inpt, neg_one)
                    d_err = d_err_real - d_err_fake
                    discriminator_optim.step()
                    pbar.update()
                make_trainable(discriminator, False)
                generator.zero_grad()
                g_err = discriminator_step(generator(create_noise(bs)), one)
                generator_optim.step()
                gen_iterations += 1

train(10, True)

# visualising results
fake = generator(fixed_noise).data.cpu()
save(img=utils.make_grid(fake), fname='cifar_grid.png', dpi=600)


