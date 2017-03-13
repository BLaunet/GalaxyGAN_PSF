#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import cv2
import math
import random
from scipy.stats import norm
import os
from astropy.io import fits
import glob
import bz2
import csv
import photutils
from IPython import embed
from config import Config as conf
# mode : 0 training : 1 testing

parser = argparse.ArgumentParser()

def gaussian_PSF(center_coord, size, sigma):
    '''
    :param center_coord: [x0,y0] coordinates of the center of the galaxy on which the PSF has to be applied
    :param size: 1D size in pixels of the generated image
    :param sigma: sigma of the gaussian distribution
    :return: normalised PSF of size size*size centered on [x0, y0]
    '''
    x0, y0 = center_coord[0]
    x, y = np.mgrid[0:size,0:size]
    g = np.exp(-(((x-y0)**2 + (y-x0)**2)/(2.0*sigma**2)))
    return g/g.sum()

def adjust(origin):
    img = origin.copy()
    img[img>4] = 4
    img[img < -0.1] = -0.1
    MIN = np.min(img)
    MAX = np.max(img)
    img = np.arcsinh(10*(img - MIN)/(MAX-MIN))/3
    return img

def crop(img, size):
    cropped = img.copy()
    center = cropped.shape[0]/2
    min = center - size/2
    max = center + size/2
    cropped = cropped[min:max,min:max,:]
    return cropped

def find_centroid(im, guesslist=np.zeros((1,2)), b_size=0):
    '''
    :param im: image data array
    :param guesslist: array of coordinate tuples (as np.ndarray) in pixel coordinates, can only be provided, if b_size
    does not vanish. The coordinates in guesslist are taken as centeres of the cutouts of size b_size.
    :param b_size: size (in pixel coordinates) of image cutout used for calculation. Is set as length of im by
    default.
    :return: array of centroid tuples
    '''
    l = guesslist.shape[0]
    b_size = int(round(b_size))
    centroid_pos = []
    if b_size == 0:
        threshold = np.max(im) / 10.
        centroid_pos.append(photutils.find_peaks(im, threshold, box_size=15, subpixel = True, npeaks = 1))
    else:
        for k in range(0,l):
            crd = map(int, guesslist[k])
            region = im[ crd[1]-b_size : crd[1]+b_size , crd[0]-b_size : crd[0]+b_size ]
            threshold = np.max(region) / 10.
            centroid_pos.append(photutils.find_peaks(region, threshold, box_size=15, subpixel = True , npeaks = 1))

    x_ctr = []
    y_ctr = []
    for i in range(0, l):
        x_ctr.append(float(centroid_pos[i]['x_centroid']) - b_size + int(guesslist[i, 0]))
        y_ctr.append(float(centroid_pos[i]['y_centroid']) - b_size + int(guesslist[i, 1]))

    centroid_pos = zip(x_ctr, y_ctr)
    return centroid_pos

def get_fluxes(filename):
    cModelFlux_r = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            cModelFlux_r[row['id']]=float(row['cModelFlux_r'])
    return cModelFlux_r

def roou():
    is_demo = 0

    parser.add_argument("--fwhm", default="1.4")
    parser.add_argument("--ratio", default="-1")
    parser.add_argument("--input", default="./fits_train")    #"/mnt/ds3lab/galaxian/source/sdss/dr12/images/fits")
    parser.add_argument("--catalog", default = "z_005_catalog.csv")
    parser.add_argument("--figure", default="figures")
    parser.add_argument("--mode", default="0")
    parser.add_argument("--crop", default = "0")
    args = parser.parse_args()

    fwhm = float(args.fwhm)
    ratio  = float(args.ratio)
    input =  args.input
    fluxes = get_fluxes(args.catalog)
    figure = args.figure
    mode = int(args.mode)
    cropsize = int(args.crop)

    train_folder = '%s/train'%(args.figure)
    test_folder = '%s/test'%(args.figure)

    if not os.path.exists('./' + args.figure):
        os.makedirs("./" + args.figure)
    if not os.path.exists("./" + train_folder):
        os.makedirs("./" + train_folder)
    if not os.path.exists("./" + test_folder):
        os.makedirs("./" + test_folder)

    fits_path = '%s/*-r.fits.bz2'%(input)
    files = glob.glob(fits_path)
    #files = [f for f in files if os.path.basename(f).replace("-r.fits.bz2", '') in fluxes.keys() ]  To be used later

    not_found_fluxes = 0
    for i in files:
        image_id = os.path.basename(i).replace("-r.fits.bz2", '')
        print(image_id)
        try:
            flux = fluxes[image_id]
        except KeyError:
            not_found_fluxes = not_found_fluxes + 1
            continue

        f = bz2.BZ2File(i)

        rfits = fits.open(f)
        data_r = rfits[0].data
        rfits.close()

        size = data_r.shape[0]
        center_coord = [size//2, size//2]

        centroid_coord = find_centroid(data_r, np.array([center_coord]), ) 
        #centroid_coord = center_coord

        figure_original = np.ones((data_r.shape[0],data_r.shape[1],1))
        figure_original[:,:,0] = data_r


        # PSF
        fwhm_use = fwhm/0.396
        gaussian_sigma = fwhm_use / 2.355
        psf = gaussian_PSF(centroid_coord, size, gaussian_sigma)

        figure_with_PSF = np.ones((data_r.shape[0],data_r.shape[1],1))

        if(ratio == -1):
            r = random.uniform(0.1, 10)
        else:
            r = ratio
        print("ratio = %s" %r)

        data_PSF = psf*r*flux + data_r
        #Renormalization
        figure_with_PSF[:, :, 0] = data_PSF*data_r.sum()/data_PSF.sum()

        

        #Crop
        if(cropsize > 0):
            figure_original = crop(figure_original,cropsize)
            figure_with_PSF = crop(figure_with_PSF, cropsize)

        print("Max pixel = figure_with_PSF.max()")
        # threshold
        MAX = 4
        MIN = -0.1


#        figure_original[figure_original<MIN]=MIN
#        figure_original[figure_original>MAX]=MAX

#        figure_with_PSF[figure_with_PSF<MIN]=MIN
#        figure_with_PSF[figure_with_PSF>MAX]=MAX

        # normalize figures
#        figure_original = (figure_original-MIN)/(MAX-MIN)
#        figure_with_PSF = (figure_with_PSF-MIN)/(MAX-MIN)


        # asinh scaling
        figure_original = np.arcsinh(10*figure_original)/3
        figure_with_PSF = np.arcsinh(10*figure_with_PSF)/3

        # output result to pix2pix format
        figure_combined = np.zeros((figure_original.shape[0], figure_original.shape[1]*2,1))
        figure_combined[:,: figure_original.shape[1],:] = figure_original[:,:,:]
        figure_combined[:, figure_original.shape[1]:2*figure_original.shape[1],:] = figure_with_PSF[:,:,:]

        if mode:
            mat_path = '%s/test/%s.npy'% (figure,image_id)
        else:
            mat_path = '%s/train/%s.npy'% (figure,image_id)

        np.save(mat_path, figure_combined)
    print("%s images have not been used because there were no corresponding flux" % not_found_fluxes)
roou()
