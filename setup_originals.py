#!/usr/bin/python

import os
import pandas as pd
import glob
import argparse
import numpy as np
from config import Config as conf

parser = argparse.ArgumentParser()


def download_psfField(catalog):
    download_main_dir = 'https://dr13.sdss.org/sas/dr13/env/PHOTO_REDUX'
    psfFields_dir = '/mnt/ds3lab/galaxian/source/sdss/dr12/psf-data'
    save_dir = '/mnt/ds3lab/blaunet/psfFields'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    im_ids = catalog['dr7ObjID'].tolist()


    psfs = []
    for i in im_ids:
        obj_line = catalog.loc[catalog['dr7ObjID'] == i]
        if obj_line.empty:
            continue

        run = obj_line['run'].item()
        rerun = obj_line['rerun'].item()
        camcol = obj_line['camcol'].item()
        field = obj_line['field'].item()
        psfFilename = 'psField-%06d-%d-%04d.fit'%(run, camcol, field)
        psfField_path = '%d/%d/objcs/%d/%s'%(rerun, run, camcol, psfFilename)
        if os.path.exists('%s/%s'%(save_dir, psfFilename)) or os.path.exists('%s/%s'%(save_dir, psfField_path)) or  os.path.exists('%s/%s'%(psfFields_dir, psfField_path)):
            continue
        psField_dl = '%s/%d/%d/objcs/%d/%s'%(download_main_dir, rerun, run, camcol, psfFilename)
        psf_save_dir = '%s/%d/%d/objcs/%d'%(save_dir,rerun, run, camcol)
        if not os.path.exists(psf_save_dir):
            os.makedirs(psf_save_dir)
        print('Download %s in %s'%(psField_dl,psf_save_dir))
        os.system('cd %s; wget %s'%(psf_save_dir, psField_dl))

def clean_dir(dir, catalog):
    objids = catalog['dr7ObjID'].tolist()
    for f in glob.glob('%s/*'%dir):
        if int(os.path.basename(f)[:-7]) not in objids:
            print('Removing %s'%f)
            os.remove(f)

    for i in objids:
        already_here = [int(os.path.basename(f)[:-7]) for f in glob.glob('%s/*'%dir)]
        if i in already_here:
            continue
        obj_line = catalog.loc[catalog['dr7ObjID'] == i]
        try:
            path = obj_line['fits_path'].item()
        except ValueError:
            print(obj_line)
            continue
        print('Copying %s'%i)
        os.system('cp %s %s'%(path, dir))



def setup():
    parser.add_argument("--train", default='')
    parser.add_argument("--test", default='')

    args = parser.parse_args()

    if not args.train:
        try:
            train_catalog = pd.read_csv(glob.glob("%s/catalog_train*"%conf.run_case)[0])
        except:
            train_catalog = pd.DataFrame()
            print('No train catalog found')
    else:
        train_catalog = pd.read_csv(args.train)

    if not args.test:
        try:
            test_catalog = pd.read_csv(glob.glob("%s/catalog_test*"%conf.run_case)[0])
        except:
            test_catalog = pd.DataFrame()
            print('No test catalog found')
    else:
        test_catalog = pd.read_csv(args.test)

    if not train_catalog.empty:
        fits_train = '%s/fits_train'%conf.run_case
        if not os.path.exists(fits_train):
            os.makedirs(fits_train)
        clean_dir(fits_train, train_catalog)
        download_psfField(train_catalog)
    if not test_catalog.empty:
        fits_test = '%s/fits_test'%conf.run_case
        if not os.path.exists(fits_test):
            os.makedirs(fits_test)
        clean_dir(fits_test, test_catalog)
        download_psfField(test_catalog)

setup()
