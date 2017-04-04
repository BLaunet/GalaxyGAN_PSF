import pandas as pd
import glob
import os

def generate_catalog():
    
    catalog_path = '/mnt/ds3lab/galaxian/source/sdss/dr13/catalogue/SDSS_Master_Table.csv'
    training_set = './fits_train'
    test_set = './fits_test'
    save_path = './catalog.csv'
    
    wanted_fields = ['dr7ObjID', 'field', 'run', 'rerun', 'camcol', 'colc', 'rowc', 'cModelFlux_r']
    
    catalog = pd.read_csv(catalog_path)
    
    train_path = glob.glob('%s/*-r.fits.bz2'%training_set)
    test_path = glob.glob('%s/*-r.fits.bz2'%test_set)
    fitspath = train_path+test_path
    im_ids = [int(os.path.basename(f).replace('-r.fits.bz2', '')) for f in fitspath]
    
    selection = catalog[catalog['dr7ObjID'].isin(im_ids)].filter(items=wanted_fields)
    print('%s objects have not been found in the master catalog'%(len(fitspath) - selection.shape[0]))    
    selection.to_csv(save_path, index=False)

if __name__ == "__main__":
    generate_catalog()