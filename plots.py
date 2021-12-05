''' train
    This file contains scripts to visualize and save images during training.
'''
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import cm
from torch.autograd import Variable
from torchvision import transforms, utils
from dataloder import ToNumpy
from utils import sample_gmm_per_cluster
matplotlib.use('Agg')


def save_set_of_images(path, image_tensor, dataset):
    if not os.path.exists(path):
        os.mkdir(path)
    transform_inv = transforms.Compose([ToNumpy()])
    output = transform_inv(image_tensor)
    for j in range(output.shape[0]):
        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            plt.imsave(os.path.join(path, "predicted%s.png" % str(j)), output[j, :, :, :].squeeze(), cmap=cm.gray)
        else:
            plt.imsave(os.path.join(path, "predicted%s.png" % str(j)), output[j, :, :, :].squeeze())


def save_sample(model, valloader, current_epoch, save_directory, dataset, num_cluster, gmm_centers,
                gmm_std):
    os.makedirs(save_directory, exist_ok=True)
    with torch.no_grad():
        model.eval()
        valbatch = next(iter(valloader))
        if dataset == "CELEB":
            validationimages = valbatch[0]
        else:
            validationimages = valbatch

        val_inputs = Variable(validationimages.type(torch.cuda.FloatTensor))
        new_val_inputs = val_inputs[:20]
        val_recon_images, val_latent_vectors = model(new_val_inputs)
        new_samples = []
        for i in range(num_cluster):
            new_samples.append(
                model.module.decode((sample_gmm_per_cluster(i, gmm_centers, gmm_std, nb_samples=20).cuda())))

        result_sample = torch.cat([new_val_inputs, val_recon_images], dim=0)
        cluster_samples = torch.cat(new_samples, dim=0)
        result_sample = result_sample.cpu()
        cluster_samples = cluster_samples.cpu()
        f = os.path.join(save_directory,
                         'sample_%d.jpg' % (
                                 current_epoch + 1)
                         )
        f_clustersamples = os.path.join(save_directory,
                                        'clustersample_%d.jpg' % (
                                                current_epoch + 1)
                                        )
        utils.save_image(result_sample, f, nrow=20)
        #utils.save_image(cluster_samples, f_clustersamples, nrow=20)
