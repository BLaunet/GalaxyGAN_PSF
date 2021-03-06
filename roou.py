#!/usr/bin/python
import argparse
import glob
import os
import random
import numpy as np
import pandas
from astropy.io import fits
from scipy.stats import norm

import photometry
from config import Config as conf

# mode : 0 training : 1 testing

parser = argparse.ArgumentParser()


def crop(img, size):
    cropped = img.copy()
    center = cropped.shape[0] / 2
    min = center - size / 2
    max = center + size / 2
    cropped = cropped[min:max, min:max, :]
    return cropped


def roou():
    random.seed(42)

    parser.add_argument("--input",
                        default="%s/fits_test" % conf.run_case)
    parser.add_argument("--output", default=conf.data_path)
    parser.add_argument("--mode", default="1")
    parser.add_argument("--crop", default="0")
    parser.add_argument('--psf', default='sdss')
    args = parser.parse_args()

    input = args.input
    output = args.output
    mode = int(args.mode)
    cropsize = int(args.crop)
    psf_type = args.psf

    #Conf params
    psf_noise = conf.noise  # sigma
    ratio_max = conf.max_contrast_ratio

    if mode == 1: #Test set
        input = '%s/fits_test' % conf.run_case
        catalog_path = glob.glob('%s/catalog_test*' % conf.run_case)[0]
    elif mode == 0: #Train set
        input = '%s/fits_train' % conf.run_case
        catalog_path = glob.glob('%s/catalog_train*' % conf.run_case)[0]
    print('Input files : %s' % input)
    catalog = pandas.read_csv(catalog_path)

    train_folder = '%s/train' % output
    test_folder = '%s/test' % output
    raw_test_folder = '%s/fits_input%s' % (conf.run_case, conf.ext)
    # GAN_input_path = '%s/%s_%s/npy_input'%(conf.run_case, conf.stretch_type, conf.scale_factor)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(raw_test_folder):
        os.makedirs(raw_test_folder)

    # fits_path = '%s/*-r.fits.bz2'%(input)
    fits_path = '%s/*-r.fits' % (input)
    files = glob.glob(fits_path)

    not_found = 0
    for i in files:
        # image_id = os.path.basename(i).replace("-r.fits.bz2", '')
        image_id = os.path.basename(i).replace('-r.fits', '')
        print('\n')
        print(image_id)

        obj_line = catalog.loc[catalog['dr7ObjID'] == int(image_id)]
        if obj_line.empty:
            not_found = not_found + 1
            print('Not found')
            continue

        # f = bz2.BZ2File(i)
        f = i

        rfits = fits.open(f)
        data_r = rfits[0].data
        rfits.close()

        flux = obj_line['cModelFlux_r'].item()
        
        fwhm = 1.4
        fwhm_use = fwhm / 0.396
        gaussian_sigma = fwhm_use / 2.355

        # add white noise
        if psf_noise != 0:
            data_r_nz = data_r[data_r < 0.1]
            data_r_nearzero = data_r_nz[data_r_nz > -0.1]
            _, s = norm.fit(data_r_nearzero)

            whitenoise_var = (psf_noise * s) ** 2

            if whitenoise_var < 0:
                whitenoise_var = 0.00000001
        else:
            whitenoise_var = None

        r = random.uniform(0.1, ratio_max)
        print("ratio = %s" % r)

        if psf_type == 'step':
            data_PSF = photometry.add_step_PSF(data_r, r * flux, fwhm_use)

        elif psf_type == 'gaussian':
            data_PSF = photometry.add_gaussian_PSF(data_r, r * flux, gaussian_sigma)

        elif psf_type == 'sdss':
            data_PSF = photometry.add_sdss_PSF(data_r, r * flux, obj_line, whitenoise_var=whitenoise_var)
            if data_PSF is None:
                print('Ignoring file %s' % i)
                continue

        else:
            print('Unknown psf type : %s' % psf_type)
            raise ValueError(psf_type)

        print('data_r centroid : %s' % photometry.find_centroid(data_r))
        print('data_PSF centroid : %s' % photometry.find_centroid(data_PSF))
        figure_original = np.ones((data_r.shape[0], data_r.shape[1], conf.img_channel))
        figure_original[:, :, 0] = data_r

        figure_with_PSF = np.ones((data_r.shape[0], data_r.shape[1], conf.img_channel))

        # Renormalization
        figure_with_PSF[:, :, 0] = data_PSF  # *data_r.sum()/data_PSF.sum()

        # Saving the "raw" data+PSF before stretching
        saving_orig = True
        if mode and saving_orig:
            raw_name = '%s/%s-r.fits' % (raw_test_folder, image_id)
            if os.path.exists(raw_name):
                os.remove(raw_name)
            hdu = fits.PrimaryHDU(data_PSF)
            hdu.writeto(raw_name)
        # Crop
        if (cropsize > 0):
            figure_original = crop(figure_original, cropsize)
            figure_with_PSF = crop(figure_with_PSF, cropsize)

        # threshold
        MAX = conf.pixel_max_value
        MIN = conf.pixel_min_value

        figure_original[figure_original < MIN] = MIN
        figure_original[figure_original > MAX] = MAX

        figure_with_PSF[figure_with_PSF < MIN] = MIN
        figure_with_PSF[figure_with_PSF > MAX] = MAX

        # Scaling
        figure_original = conf.stretch(figure_original)
        figure_with_PSF = conf.stretch(figure_with_PSF)

        # print(figure_with_PSF)
        # output result to pix2pix format
        figure_combined = np.zeros((figure_original.shape[0], figure_original.shape[1] * 2, 1))
        figure_combined[:, : figure_original.shape[1], :] = figure_original[:, :, :]
        figure_combined[:, figure_original.shape[1]:2 * figure_original.shape[1], :] = figure_with_PSF[:, :, :]

        if mode:
            mat_path = '%s/test/%s-r.npy' % (output, image_id)
            # np.save('%s/%s-r.npy'%(GAN_input_path,image_id), figure_combined)
        else:
            mat_path = '%s/train/%s-r.npy' % (output, image_id)
        np.save(mat_path, figure_combined)
    print("%s images have not been used because there were no corresponding objects in the catalog" % not_found)


roou()
