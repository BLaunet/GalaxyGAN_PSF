#!/usr/bin/python
import argparse
import os
import glob
import pandas as pd
import numpy as np
from config import Config as conf
import sys

parser = argparse.ArgumentParser()

def extract_catalog():
    parser.add_argument("--zmin", default="0.194")
    parser.add_argument("--zmax", default="0.206")
    parser.add_argument("--train_size", default="5000")
    parser.add_argument("--test_size", default = "100")

    args = parser.parse_args()

    zmin = float(args.zmin)
    zmax  = float(args.zmax)
    train_size = int(args.train_size)
    test_size = int(args.test_size)

    wanted_fields = ['dr7ObjID', 'Z','field', 'run', 'rerun', 'camcol', 'colc', 'rowc', 'cModelFlux_r']


    train_csv = '%s/catalog_train_z_%s_%s.csv'%(conf.run_case,zmin,zmax)
    test_csv = '%s/catalog_test_z_%s_%s.csv'%(conf.run_case,zmin,zmax)

    if not os.path.exists(conf.run_case):
        os.makedirs(conf.run_case)

    source_dir = '/mnt/ds3lab/galaxian/source/sdss'
    fits_dir = '%s/dr12/images/fits'%source_dir

    new_catalog = pd.read_csv('%s/dr13/catalogue/SDSS_Master_Table.csv'%source_dir).filter(items=wanted_fields)
    new_catalog=new_catalog[(new_catalog['Z']>zmin) & (new_catalog['Z']<zmax)]
    new_catalog['fits_path'] = fits_dir \
                               +'/'+new_catalog['run'].apply(str) \
                               +'/'+new_catalog['camcol'].apply(str) \
                               +'/'+new_catalog['field'].apply(str) \
                               +'/'+new_catalog['dr7ObjID'].apply(str) \
                               +'-r.fits'
    new_catalog = new_catalog[new_catalog['fits_path'].apply(os.path.exists)]

    if new_catalog.shape[0] < (train_size+test_size):
        print('Not enough objects to match criteria : %s \n' % new_catalog.shape[0])
        sys.exit()

    train_df = new_catalog.iloc[:train_size]
    test_df = new_catalog.iloc[train_size:train_size+test_size]

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
extract_catalog()
