''' train
    This file contains training script.
'''
import os
import sys

import matplotlib
import torch
import torch.nn as nn
from torch.autograd import Variable

from loss import (get_vaeloss, estimate_loss_coefficients)
from model import VAE
from plots import save_sample
from prepareinput import prepare_data_loader
from utils import set_gmm_centers, exp_lr_scheduler

matplotlib.use('Agg')


if __name__ == "__main__":
    try:
        from configparser import ConfigParser
    except ImportError:
        from configparser import ConfigParser  # ver. < 3.0
    major_idx = str(sys.argv[1])
    # instantiate
    config = ConfigParser()

    # parse existing file
    config.read('config.ini')
    dataset = config.get(major_idx, 'dataset')
    experiment_name = config.get(major_idx, 'experiment_name')
    img_size = config.getint(major_idx, 'image_size')
    batch_size = config.getint(major_idx, 'batch_size')
    num_cluster = config.getint(major_idx, 'num_clusters')
    epochs = config.getint(major_idx, 'epochs')
    latent_dim = config.getint(major_idx, 'latent_dim')
    image_num_channels = config.getint(major_idx, 'image_num_channels')
    nef = config.getint(major_idx, 'nef')
    ndf = config.getint(major_idx, 'ndf')
    lr = config.getfloat(major_idx, 'lr')
    exp_lr = config.getboolean(major_idx, 'exp_lr')
    latent_noise_scale = config.getfloat(major_idx, 'latent_noise_scale')
    use_L2 = config.getboolean(major_idx, 'use_L2')
    save_dir = config.get(major_idx, 'save_dir') + "/" + dataset + "/" + experiment_name
    data_dir = config.get(major_idx, 'data_dir')
    image_loss_weight = config.getfloat(major_idx, 'image_loss_weight')
    # create the directories if not exist
    os.makedirs(save_dir, exist_ok=True)
    # get train and test dataloder
    trainloader, valloader, testloader = prepare_data_loader(data_dir, batch_size, dataset)
    # set prior means and std
    gmm_centers, gmm_std = set_gmm_centers(latent_dim, num_cluster)

    # get weights of the loss functions used
    ks_weight, cv_weight = estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=64)

    # Initialize the model.
    model = VAE(dataset=dataset, nc=image_num_channels, ndf=ndf, nef=nef, nz=latent_dim, isize=img_size,
                latent_noise_scale=latent_noise_scale)
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        if exp_lr:
            optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=7)
        for i, data in enumerate(trainloader, 0):
            model.train()
            if dataset == "CELEB":
                inputs = Variable(data[0].type(torch.cuda.FloatTensor))
            else:
                inputs = Variable(data.type(torch.cuda.FloatTensor))

            recon_images, latent_vectors = model(inputs)
            loss_mean, weighted_ksloss, weighted_cov_loss, weighted_imageloss = \
                get_vaeloss(recon_images, latent_vectors, inputs, ks_weight, cv_weight, image_loss_weight, gmm_centers,
                            gmm_std)

            # zero the parameter gradients
            optimizer.zero_grad()
            # Apply L2 regularization to the decoder
            if use_L2:
                l2_reg = None
                for W in model.decoder.parameters():
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
                loss_mean = loss_mean + 1e-7 * l2_reg

            loss_mean.backward()
            optimizer.step()

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\treconloss: {:.6f} cdf: {:.6f} var: {:.6f} totalloss: '
                '{:.6f}'.format(
                    epoch, i * len(data), len(trainloader.dataset),
                           100. * i / len(trainloader), weighted_imageloss, weighted_ksloss, weighted_cov_loss, loss_mean.item()))
        # save images every 5 epochs
        if epoch % 5 == 0:
            save_sample(model, valloader, epoch, save_dir, dataset, num_cluster,
                        gmm_centers, gmm_std)

    print('Finished Training')
    # Save the model
    torch.save(model.state_dict(), '%s/vae_epoch_final_%d.pth' % (save_dir, epoch))
    print('The trained model was stored in %s' % save_dir)
