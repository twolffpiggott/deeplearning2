import torch
import torch.nn as nn


class MLP_G(nn.Module):
    """
    Multilayer perceptron generator
    """
    def __init__(self, isize, nz, nc, ngf, ngpu):
        """
        Initialisation of MLP_G.

        :param isize: number of pixels in output image
        :param nz: size of random noise input to generator
        :param nc: number of channels in outut image
        :param ngf: hidden layer(s) size
        :param ngpu: number of gpus to train on
        """
        super().__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # input Z goes into a linear layer of size ngf
            nn.Linear(in_features=nz, out_features=ngf),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ngf, out_features=ngf),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ngf, out_features=ngf),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ngf, out_features=nc*isize*isize)
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        """
        Forward pass of MLP_G

        :param input: input to generator
        """
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(module=self.main,
                                               inputs=input,
                                               device_ids=range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    """
    Multilayer perceptron discriminator
    """
    def __init__(self, isize, nz, nc, ndf, ngpu):
        """
        Initialisation of MLP_D.

        :param isize: number of pixels in input image
        :param nz: size of random noise input to generator
        :param nc: number of channels in input image
        :param ndf: hidden layer(s) size
        :param ngpu: number of gpus to train on
        """
        super().__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # input Z goes to a linear layer of size ndf
            nn.Linear(in_features=nc*isize*isize, out_features=ndf),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=nfd, out_features=ndf),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=nfd, out_features=ndf),
            nn.ReLU(inplace=True),
            nn.Linear(ndf, 1)
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        """
        Forward pass of MLP_D.

        :param input: input to discriminator
        """
        input = input.view(input.size(0),
                           input.size(1)*input.size(2)*input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(module=self.main,
                                               inputs=input,
                                               device_ids=range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)
