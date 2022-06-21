#!/usr/bin/env python3
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
from imageio import imread
from scipy import linalg
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
from torch.utils import data
import img_data as img_data
import sent_data as sent_data
import clip
import torch.nn as nn


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=512,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--path1', type=str, default=64)
parser.add_argument('--path2', type=str, default=64)
parser.add_argument('--path3', type=str, default=64)


def get_activations(data_loader, model, batch_size=64, dims=2048, cuda=False, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    # d0 = images.shape[0]
    num_batches = len(data_loader)
    n_used_imgs = num_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))
    # for i in range(n_batches):
    for i, batch in enumerate(data_loader):
        start = i * batch_size
        end = start + batch_size
        if cuda:
            batch = batch.cuda()
        # exchaged_image = nn.functional.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
        cnn_code, region_features = model.encode_image(batch)
        cnn_code /= cnn_code.norm(dim=-1, keepdim=True)
        cnn_code *= 10
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if cnn_code.shape[2] != 1 or cnn_code.shape[3] != 1:
        #     cnn_code = adaptive_avg_pool2d(cnn_code, output_size=(1, 1))
        #     print(cnn_code.size())
        pred_arr[start:end] = cnn_code.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_new_distance(mu_f, sigma_f, mu_r, sigma_r, mu_l, sigma_l, eps=1e-6):
    muf = np.atleast_1d(mu_f)
    mur = np.atleast_1d(mu_r)
    mul = np.atleast_1d(mu_l)

    sigmaf = np.atleast_2d(sigma_f)
    sigmar = np.atleast_2d(sigma_r)
    sigmal = np.atleast_2d(sigma_l)

    diff1 = muf - mur
    diff2 = muf - mul
    # Product might be almost singular
    covf, _ = linalg.sqrtm(sigmaf, disp=False)
    covr, _ = linalg.sqrtm(sigmar, disp=False)
    covl, _ = linalg.sqrtm(sigmal, disp=False)
    covfl, _ = linalg.sqrtm(sigmaf.dot(sigmal), disp=False)
    covrl, _ = linalg.sqrtm(sigmar.dot(sigmal), disp=False)
    covfr, _ = linalg.sqrtm(sigmaf.dot(sigmar), disp=False)
    tr_2 = np.trace(covf.dot(covf)) - np.trace(covfl) + np.trace(covrl) - np.trace(covfr)

    mean = 2 * (diff1.dot(diff2))
    tr_covmean = np.trace((covf - covl).dot(covf - covr))
    # print(mean, 2 * tr_covmean, 2 * tr_2)
    return mean + 2 * tr_covmean


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    # print(diff.dot(diff), np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=False, verbose=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        dataset = img_data.Dataset(path, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        print(dataset.__len__())
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
        m, s = calculate_activation_statistics(dataloader, model, batch_size, dims, cuda)
        # np.savez('image_CUB_feature_VIT.npz', mu=m, sigma=s)
        # raise KeyError
    return m, s


def compute_language_statistics(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        mu, sigma = f['mu'][:], f['sigma'][:]
        f.close()

    else:
        split = 'test'
        dataset = sent_data.Dataset(path, split)
        print(dataset.__len__())
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        num_batches = len(dataloader)
        n_used_imgs = num_batches * batch_size
        pred_arr = np.empty((n_used_imgs, dims))
        # for i in range(n_batches):
        for i, batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size
            if cuda:
                batch = batch.cuda()
            # exchaged_image = nn.functional.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
            text_features, word_embs = model.encode_text(batch)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            # if cnn_code.shape[2] != 1 or cnn_code.shape[3] != 1:
            #     cnn_code = adaptive_avg_pool2d(cnn_code, output_size=(1, 1))
            #     print(cnn_code.size())
            pred_arr[start:end] = text_features.cpu().data.numpy().reshape(batch_size, -1)

        print(' done')
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)

        # np.savez('sent_CUB_feature_VIT.npz', mu=mu, sigma=sigma)
        # raise KeyError
    return mu, sigma


def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    #
    device = "cuda"
    # clip_model, _ = clip.load("RN101", device=device)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    print('Load Oral clip encoder')
    clip_model.float()
    clip_model.eval()

    m1, s1 = _compute_statistics_of_path(paths[0], clip_model, batch_size, dims, cuda)
    m2, s2 = compute_language_statistics(paths[1], clip_model, batch_size, dims, cuda)
    m3, s3 = _compute_statistics_of_path(paths[2], clip_model, batch_size, dims, cuda)

    fid_value1 = calculate_frechet_distance(m1, s1, m2, s2)
    print('Distance between real image and real sent: ', fid_value1)
    fid_value2 = calculate_frechet_distance(m3, s3, m2, s2)
    print('Distance between fake image and real sent: ', fid_value2)
    fid_value3 = calculate_frechet_distance(m1, s1, m3, s3)
    print('Distance between real image and fake image: ', fid_value3)
    print('CMD score is: ', fid_value3 + fid_value2 - fid_value1)
    return fid_value1


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = ["", "", ""]
    paths[0] = args.path1
    paths[1] = args.path2
    paths[2] = args.path3
    print(paths)
    fid_value = calculate_fid_given_paths(paths, args.batch_size, args.gpu, args.dims)
