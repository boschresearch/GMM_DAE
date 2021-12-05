''' latent regularizer - this file contains our proposed regularization loss.
// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import matplotlib
import torch
from utils import compute_empirical_covariance, compute_gmm_covariance
matplotlib.use('Agg')


def mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(embedding_matrix, gmm_centers, gmm_std):
    """Return the kolmogorov distance for each dimension.

    embedding_matrix:
        The latent representation of the batch.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        All components of the GMM are assumed to have share the same covariance matrix: C = gmm_std**2 * Identity.

    Note that the returned distances are NOT in the same order as embedding matrix.
    Thus, this is useful for means/max, but not for visual inspection.
    """

    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    emb_num, emb_dim = sorted_embeddings.shape[-2:]
    num_gmm_centers, _ = gmm_centers.shape
    # For the sorted embeddings, the empirical CDF depends to the "index" of each
    # embedding (the number of embeddings before it).
    # Unsqueeze enables broadcasting
    empirical_cdf = torch.linspace(
        start=1 / emb_num,
        end=1.0,
        steps=emb_num,
        device=embedding_matrix.device,
        dtype=embedding_matrix.dtype,
    ).unsqueeze(-1)

    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std

    # compute CDF values for the embeddings using the Error Function
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)

    return torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)


def mean_squared_covariance_gmm(embedding_matrix, gmm_centers, gmm_std):
    """Compute mean squared distance between the empirical covariance matrix of a embedding matrix and the covariance of
    a GMM prior with given centers and per center standard deviation under
    the assumption that different dimensions are uncorrelated on a per center level and equal weighing of modes.

    Parameters
    ----------
    embedding_matrix: torch.Tensor
        Latent Vectors.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        All components of the GMM are assumed to have share the same covariance matrix: C = gmm_std**2 * Identity.
    Returns
    -------
    mean_cov: float
        Mean squared distance between empirical and prior covariance.

    """
    # Compute empirical covariances:
    sigma = compute_empirical_covariance(embedding_matrix)
    comp_covariance, gmm_covariance = compute_gmm_covariance(gmm_centers, gmm_std)
    comp_covariance.to(embedding_matrix.device)
    gmm_covariance.to(embedding_matrix.device)

    diff = torch.pow(sigma - gmm_covariance, 2)

    mean_cov = torch.mean(diff)
    return mean_cov


def mean_squared_individual_covariance_gmm(embedding_matrix, gmm_centers, gmm_std):
    """Compute mean squared distance between the empirical covariance matrix of a embedding matrix and the covariance of
    a GMM prior with given centers and per center standard deviation under
    the assumption that different dimensions are uncorrelated on a per center level and equal weighing of modes.

    Parameters
    ----------
    embedding_matrix: torch.Tensor
        Latent Vectors.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        All components of the GMM are assumed to have share the same covariance matrix: C = gmm_std**2 * Identity.
    Returns
    -------
    mean_cov: float
        Mean squared distance between empirical and prior covariance.

    """
    # Compute empirical covariances:
    sigma = compute_empirical_covariance(embedding_matrix)
    # Compare it with GMM covariance matrix.
    comp_covariance, gmm_covariance = compute_gmm_covariance(gmm_centers, gmm_std)
    comp_covariance.to(embedding_matrix.device)
    gmm_covariance.to(embedding_matrix.device)

    diff = torch.pow(sigma - gmm_covariance, 2)

    mean_cov = torch.mean(diff)
    return mean_cov
