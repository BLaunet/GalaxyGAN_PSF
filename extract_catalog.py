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
    parser.add_argument("--zmin", default="0.095")
    parser.add_argument("--zmax", default="0.105")
    parser.add_argument("--train_size", default="1000")
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

    old_catalog = pd.read_csv('%s/dr12/catalogue/SDSS_objects_specsample.csv'%source_dir)
    new_catalog = pd.read_csv('%s/dr13/catalogue/SDSS_Master_Table.csv'%source_dir).filter(items=wanted_fields)

    redshift_specified=new_catalog[(new_catalog['Z']>zmin) & (new_catalog['Z']<zmax)]
    if redshift_specified.shape[0] < (train_size+test_size):
        print('Not enough objects to match criteria : %s \n' % redshift_specified.shape[0])
        sys.exit()



    objids = iter(redshift_specified['dr7ObjID'].tolist())
    obj_id = objids.next()

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    print('Building training catalog')
    while train_df.shape[0] < train_size:
        line = new_catalog[new_catalog['dr7ObjID'] == obj_id]
        run = line['run'].item()
        camcol = line['camcol'].item()
        field = line['field'].item()
        fits_path = '%s/%s/%s/%s/%s-r.fits'%(fits_dir, run, camcol, field, obj_id)
        if not os.path.exists(fits_path):
            print'Object %s not found'%obj_id
            try:
                obj_id = objids.next()
                continue
            except StopIteration:
                print('We reached the end of the catalog in building the training catalog')
                break
        line['fits_path'] = fits_path
        train_df = train_df.append(line)
        try:
            obj_id = objids.next()
        except StopIteration:
            print('We reached the end of the catalog in building the training catalog')
            break

    print('Building testing catalog')
    while test_df.shape[0] < test_size:
        line = new_catalog[new_catalog['dr7ObjID'] == obj_id]
        try:
            run = line['run'].item()
        except ValueError:
            print(line)
            print(obj_id)
            try:
                obj_id = objids.next()
                continue
            except StopIteration:
                print('We reached the end of the catalog in building the test catalog')
                break
        camcol = line['camcol'].item()
        field = line['field'].item()
        fits_path = '%s/%s/%s/%s/%s-r.fits'%(fits_dir, run, camcol, field, obj_id)
        if not os.path.exists(fits_path):
            print'Object %s not found'%obj_id
            try:
                obj_id = objids.next()
                continue
            except StopIteration:
                print('We reached the end of the catalog in building the test catalog')
                break
        line['fits_path'] = fits_path
        test_df = test_df.append(line)
        try:
            obj_id = objids.next()
        except StopIteration:
            print('We reached the end of the catalog in building the test catalog')
            break

    print('Train Catalog size = %s'%train_df.shape[0])
    print('Test Catalog size = %s'%test_df.shape[0])

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
extract_catalog()
