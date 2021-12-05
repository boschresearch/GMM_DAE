''' model
    This file contains VAE model definition for MNIST, FASHIONMNIST, SVHN and CELEBA dataset.
'''
import torch
import torch.utils.data
from torch import nn


class MNISTEncoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(MNISTEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(True),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 2),
            nn.ReLU(True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 4),
            nn.ReLU(True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 8),
            nn.ReLU(True)
        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        # Reshape
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class MNISTDecoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(MNISTDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 2 * 2 * 1024),
            nn.ReLU(True),

        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 2, 2)
        output = self.decoder_conv(input)
        return output


class SVHNEncoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(SVHNEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(True),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 2),
            nn.ReLU(True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 4),
            nn.ReLU(True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 8),
            nn.ReLU(True)
        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class SVHNDecoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(SVHNDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 2 * 2 * 1024),
            nn.ReLU(True)
        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            self.conv2,
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            self.conv3,
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            self.conv4,
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 2, 2)
        output = self.decoder_conv(input)
        return output


class CELEBEncoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(CELEBEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 2),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 4),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 8),
        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class CELEBDecoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(CELEBDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 4 * 4 * 1024),
            nn.ReLU(True)
        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf * 4),
            self.conv2,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf * 2),
            self.conv3,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf),
            self.conv4,
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 4, 4)
        output = self.decoder_conv(input)
        return output


class VAE(nn.Module):
    def __init__(self, dataset="MNIST", nc=1, ndf=32, nef=32, nz=16, isize=128, latent_noise_scale=1e-3,
                 device=torch.device("cuda:0"), is_train=True):
        super(VAE, self).__init__()
        self.nz = nz
        self.is_train = is_train
        self.latent_noise_scale = latent_noise_scale

        if dataset == "MNIST" or dataset == "FASHIONMNIST" :
            # Encoder
            self.encoder = MNISTEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = MNISTDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "SVHN":
            # Encoder
            self.encoder = SVHNEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = SVHNDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "CELEB":
            # Encoder
            self.encoder = CELEBEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = CELEBDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)

    def forward(self, images):
        z = self.encode(images)
        if self.is_train:
            z_noise = self.latent_noise_scale * torch.randn((images.size(0), self.nz), device=z.device)
        else:
            z_noise = 0.0
        return self.decode(z + z_noise), z

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z):
        return self.decoder(z)
