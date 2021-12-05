''' loss
    This file contains the total training loss of our model.
'''
import numpy as np
from torch.nn import functional as F

from latent_regularizer import (
    mean_squared_covariance_gmm,
    mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting
)
from utils import draw_gmm_samples


def estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100):
    """Estimate the weights of our multi-modal loss."""
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses = [], []
    # Estimate weights with gmm samples:
    for i in range(num_samples):
        z, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(
            embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std
        )
        ks_loss = ks_loss.cpu().detach().numpy()
        cv_loss = mean_squared_covariance_gmm(
            embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std
        )
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ks_loss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, cv_weight


def get_vaeloss(predicted_images, latent_vectors, true_images, ks_weight, cv_weight, image_loss_weight, gmm_centers,
                gmm_std):
    """Total loss function."""
    # Determine the loss on images:
    image_loss = F.mse_loss(predicted_images, true_images)
    ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(latent_vectors, gmm_centers, gmm_std)
    cs_loss = mean_squared_covariance_gmm(latent_vectors, gmm_centers, gmm_std)
    weighted_ksloss = ks_weight * ks_loss
    weighted_cov_loss = cv_weight * cs_loss
    weighted_imageloss = 1 / image_loss_weight * image_loss
    losses = weighted_ksloss + weighted_cov_loss + weighted_imageloss
    loss_mean = losses.mean().cuda()
    return loss_mean, weighted_ksloss, weighted_cov_loss, weighted_imageloss
