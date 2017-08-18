#!/usr/bin/python
import argparse
import numpy as np
import os
import glob
from astropy.io import fits
from config import Config as conf
# mode : 0 training : 1 testing
parser = argparse.ArgumentParser()


def format_pix2pix():

    parser.add_argument("--in", default='%s/fits_test'%conf.run_case)
    parser.add_argument("--out", default="%s/npy_input/test"%conf.run_case)
    args = parser.parse_args()
    filter_string = conf.filter_
    fits_test = args.in
    npy_dir= args.out
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    for f in glob.glob('%s/*-%s.fits'%(fits_test, filter_string)):
        image_id = os.path.basename(f).replace('-%s.fits','') % (filter_string)

        rfits = fits.open(f)
        data_r = rfits[0].data
        rfits.close()

        figure_original = np.ones((data_r.shape[0],data_r.shape[1],1))
        figure_original[:,:,0] = data_r
        figure_original = conf.stretch(figure_original)

        # output result to pix2pix format
        figure_combined = np.zeros((figure_original.shape[0], figure_original.shape[1]*2,1))
        figure_combined[:,: figure_original.shape[1],:] = figure_original[:,:,:]
        figure_combined[:, figure_original.shape[1]:2*figure_original.shape[1],:] = figure_original[:,:,:]

        mat_path = '%s/%s-%s.npy'% (npy_dir,image_id, filter_string)

        np.save(mat_path, figure_combined)

format_pix2pix()
