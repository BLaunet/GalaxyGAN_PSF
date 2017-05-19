
import numpy as np
from scipy.stats import norm
import os
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import pandas
import photutils
import galfit
from config import Config as conf

def generate_sdss_psf(obj_line, psf_filename):
    home_dir = '/mnt/ds3lab/blaunet'
    psfTool_path = '%s/readAtlasImages-v5_4_11/read_PSF'%home_dir
    psfFields_dir = '%s/psfFields'%home_dir
    #psfFields_dir = '/mnt/ds3lab/galaxian/source/sdss/dr12/psf-data'

    run = obj_line['run'].item()
    rerun = obj_line['rerun'].item()
    camcol = obj_line['camcol'].item()
    field = obj_line['field'].item()
    #psfField = '%s/%d/%d/objcs/%d/psField-%06d-%d-%04d.fit'%(psfFields_dir, rerun, run, camcol, run, camcol, field)

    psfField = '%s/psField-%06d-%d-%04d.fit'%(psfFields_dir, run, camcol, field)

    colc = obj_line['colc'].item()
    rowc = obj_line['rowc'].item()
    os.system('%s %s 3 %s %s %s'%(psfTool_path, psfField, rowc, colc, psf_filename ))
    try:
        hdu = fits.open(psf_filename)
        psf_data = np.array(hdu[0].data, dtype = float)/1000 - 1
        hdu.close()
        os.remove(psf_filename)
        hdu = fits.PrimaryHDU(psf_data)
        hdu.writeto(psf_filename)
    except:
        print('no psf %s'%psf_filename)

def add_sdss_PSF(original, psf_flux, obj_line, multiple=False):

    SDSS_psf_dir = '%s/psf/SDSS'%conf.run_case
    GALFIT_psf_dir = '%s/psf/GALFIT'%conf.run_case
    if not os.path.exists(SDSS_psf_dir):
        os.makedirs(SDSS_psf_dir)
    if not os.path.exists(GALFIT_psf_dir):
        os.makedirs(GALFIT_psf_dir)

    obj_id = obj_line['dr7ObjID'].item()
    SDSS_psf_filename = '%s/%s-r.fits'%(SDSS_psf_dir, obj_id)
    GALFIT_psf_filename = '%s/%s-r.fits'%(GALFIT_psf_dir, obj_id)
    if not os.path.exists(GALFIT_psf_filename):
        print('No Galfit PSF')
        if not os.path.exists(SDSS_psf_filename):
            generate_sdss_psf(obj_line, SDSS_psf_filename)
        psf = galfit.fit_PSF_GALFIT(SDSS_psf_filename, GALFIT_psf_dir)
        if psf is None:
            return None
    else:
        psf = galfit.open_GALFIT_results(GALFIT_psf_filename, 'model')

    center = [original.shape[1]//2, original.shape[0]//2]
    centroid_galaxy = find_centroid(original, guesslist=center, b_size=20)
    #centroid_galaxy = [211,211]
    centroid_PSF = find_centroid(psf)
    #centroid_PSF = [25.0,25.0]

    composite_image = np.copy(original)

    k = 3 if multiple else 1
    
    gal_x = int(round(centroid_galaxy[0]))
    gal_y = int(round(centroid_galaxy[1]))
    ps_x = int(round(centroid_PSF[0]))
    ps_y = int(round(centroid_PSF[1]))

    psf = psf/psf.sum()
    for x in range(0, psf.shape[1]):
        for y in range(0, psf.shape[0]):
            x_rel = gal_x - ps_x + x
            y_rel = gal_y - ps_y + y
            if x_rel>=0 and y_rel>=0 and x_rel<original.shape[1] and y_rel<original.shape[0]:
                composite_image[y_rel,x_rel] += psf_flux*psf[y, x]

    return composite_image

def add_gaussian_PSF(original, psf_flux, sigma):
    '''
    :param center_coord: [x0,y0] coordinates of the center of the galaxy on which the PSF has to be applied
    :param size: 1D size in pixels of the generated image
    :param sigma: sigma of the gaussian distribution
    :return: normalised PSF of size size*size centered on [x0, y0]
    '''
    size = original.shape[0]
    x0, y0 = find_centroid(original)
    x, y = np.mgrid[0:size,0:size]
    psf = np.exp(-(((x-y0)**2 + (y-x0)**2)/(2.0*sigma**2)))

    return original+psf_flux*psf/psf.sum()

def add_step_PSF(original, psf_flux, sigma):
    size = original.shape[0]
    x0, y0 = find_centroid(original)
    psf = np.zeros((size,size))
    #psf[int(y0-sigma/2):int(y0+sigma/2)+1,int(x0-sigma/2):int(x0+sigma/2)+1]=1
    for i in range(size):
        for j in range(size):
            if (x0-i)**2+(y0-j)**2 <= sigma**2:
                psf[j,i]=1
    return original+psf_flux*psf/psf.sum()

def starphot(imdata, position, radius, r_in, r_out, plotting=False, plotfilename=None, plotpath=None):
    '''
    sources: http://spiff.rit.edu/classes/phys373/lectures/signal/signal_illus.html, http://photutils.readthedocs.io/en/stable/photutils/aperture.html
    :param imdata:
    :param position:
    :param radius:
    :param plotting:
    :param plotfilename:
    :param plotpath:
    :return:
    '''
    statmask = photutils.make_source_mask(imdata, snr=5, npixels=5, dilate_size=10)
    bkg_annulus = photutils.CircularAnnulus(position, r_in, r_out)
    bkg_phot_table = photutils.aperture_photometry(imdata, bkg_annulus, method='subpixel', mask=statmask)
    bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
    src_aperture = photutils.CircularAperture(position, radius)
    src_phot_table = photutils.aperture_photometry(imdata, src_aperture, method='subpixel')
    signal = src_phot_table['aperture_sum'] - bkg_mean_per_pixel*src_aperture.area()
    noise_squared = signal + bkg_mean_per_pixel*src_aperture.area()
    if plotting == True:
        fig, ax = plt.subplots()
        ax.imshow(imdata, cmap='gray_r', origin='lower', norm=LogNorm())
        src_aperture.plot(ax=ax)
        bkg_annulus.plot(ax=ax)
        plt.title(plotfilename)
        plt.savefig(plotpath+plotfilename)
        plt.close()
    return float(str(signal.data[0])), float(str(noise_squared.data[0])), float(str(bkg_mean_per_pixel.data[0]))

def find_centroid(im, guesslist=np.zeros(2), b_size=0):
    '''
    :param im: image data array
    :param guesslist: array of coordinate tuples (as np.ndarray) in pixel coordinates, can only be provided, if b_size
    does not vanish. The coordinates in guesslist are taken as centeres of the cutouts of size b_size.
    :param b_size: size (in pixel coordinates) of image cutout used for calculation. Is set as length of im by
    default.
    :return: array of centroid tuples
    '''
    b_size = int(round(b_size))
    if b_size == 0:

        mean, median, std = sigma_clipped_stats(im, sigma = 3.0, iters=5)
        #threshold = np.max(im) / 10.
        threshold = 3*std
        centroid_pos = (photutils.find_peaks(im, threshold, box_size=20, subpixel = True, npeaks = 1))


    else:
        crd = (map(int, guesslist))
        region = im[ crd[1]-b_size : crd[1]+b_size , crd[0]-b_size : crd[0]+b_size ]
        mean, median, std = sigma_clipped_stats(region, sigma = 3.0, iters=5)
        threshold = 3*std
        centroid_pos = photutils.find_peaks(region, threshold, box_size=b_size, subpixel = True , npeaks = 1)
    x_ctr = (float(centroid_pos['x_centroid']) - b_size + int(guesslist[0]))
    y_ctr = (float(centroid_pos['y_centroid']) - b_size + int(guesslist[1]))

    return [x_ctr, y_ctr]
