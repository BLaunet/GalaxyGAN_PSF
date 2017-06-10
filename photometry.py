import os

import numpy as np
from astropy.io import fits
from photutils import centroid_com

import galfit
from config import Config as conf

from shutil import copyfile
import glob
sex = 'sextractor'


def calc_zeropoint(exposure_time, calibration_factor):
    return 22.5 + 2.5 * np.log10(1. / exposure_time / calibration_factor)


def SExtractor_get_stars(path, filename, magzero, threshold, saturation_level, gain, pixel_scale, fwhm, imageshape):
    file_res = open(path + 'sex_stars.conf', "w")
    file_res.write('#-------------------------------- Catalog ------------------------------------\n\n')
    file_res.write('CATALOG_NAME     sex_stars.cat        # name of the output catalog\n')
    file_res.write('CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n')
    file_res.write('                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC \n')
    file_res.write('PARAMETERS_NAME  {}{}           # name of the file containing catalog contents \n\n'.format(path, 'sex_stars.param'))
    file_res.write('#------------------------------- Extraction ----------------------------------\n\n')
    file_res.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)\n')
    file_res.write('DETECT_MINAREA   5              # min. # of pixels above threshold\n')
    file_res.write('DETECT_THRESH    5              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n')
    file_res.write('ANALYSIS_THRESH  {}             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n\n'.format(threshold))
    file_res.write('FILTER           Y              # apply filter for detection (Y or N)?\n')
    file_res.write('FILTER_NAME      /mnt/ds3lab/dostark/sextractor_defaultfiles/default.conv   # name of the file containing the filter\n\n')
    file_res.write('DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds \n')
    file_res.write('DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending\n\n')
    file_res.write('CLEAN            Y              # Clean spurious detections? (Y or N)?\n')
    file_res.write('CLEAN_PARAM      1.0            # Cleaning efficiency)\n\n')
    file_res.write('MASK_TYPE        CORRECT        # type of detection MASKing: can be one of\n\n')
    file_res.write('                                # NONE, BLANK or CORRECT\n\n')
    file_res.write('#------------------------------ Photometry -----------------------------------\n\n')
    file_res.write('PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels\n')
    file_res.write('PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>\n')
    file_res.write('PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,\n')
    file_res.write('                                # <min_radius>\n\n')
    file_res.write('SATUR_LEVEL      {}             # level (in ADUs) at which arises saturation\n'.format(saturation_level))
    file_res.write('SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)\n\n')
    file_res.write('MAG_ZEROPOINT    {}             # magnitude zero-point\n'.format(magzero))
    file_res.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)\n')
    file_res.write('GAIN             {}             # detector gain in e-/ADU\n'.format(gain))
    file_res.write('GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU\n')
    file_res.write('PIXEL_SCALE      {}             # size of pixel in arcsec (0=use FITS WCS info)\n\n'.format(pixel_scale))
    file_res.write('# ------------------------- Star/Galaxy Separation ----------------------------\n\n')
    file_res.write('SEEING_FWHM      {}             # stellar FWHM in arcsec\n'.format(fwhm))
    file_res.write('STARNNW_NAME     /mnt/ds3lab/dostark/sextractor_defaultfiles/default.nnw  # Neural-Network_Weight table filename\n\n')
    file_res.write('# ------------------------------ Background -----------------------------------\n\n')
    file_res.write('BACK_SIZE        64              # Background mesh: <size> or <width>,<height>\n')
    file_res.write('BACK_FILTERSIZE  3               # Background filter: <size> or <width>,<height>\n\n')
    file_res.write('BACKPHOTO_TYPE   GLOBAL          # can be GLOBAL or LOCAL\n\n')
    file_res.write('#------------------------------ Check Image ----------------------------------\n\n')
    file_res.write('CHECKIMAGE_TYPE  NONE            # can be NONE, BACKGROUND, BACKGROUND_RMS,\n')
    file_res.write('                                 # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,\n')
    file_res.write('                                 # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,\n')
    file_res.write('                                 # or APERTURES\n')
    file_res.write('CHECKIMAGE_NAME  /mnt/ds3lab/dostark/sextractor_defaultfiles/check.fits     # Filename for the check-image\n\n')
    file_res.write('#--------------------- Memory (change with caution!) -------------------------\n\n')
    file_res.write('MEMORY_OBJSTACK  3000            # number of objects in stack\n')
    file_res.write('MEMORY_PIXSTACK  300000          # number of pixels in stack\n')
    file_res.write('MEMORY_BUFSIZE   1024            # number of lines in buffer\n\n')
    file_res.write('#----------------------------- Miscellaneous ---------------------------------\n\n')
    file_res.write('VERBOSE_TYPE     NORMAL          # can be QUIET, NORMAL or FULL\n')
    file_res.write('HEADER_SUFFIX    .head           # Filename extension for additional headers\n')
    file_res.write('WRITE_XML        N               # Write XML file (Y/N)?\n')
    file_res.write('XML_NAME         sex.xml         # Filename for XML output\n')
    file_res.close()

    file_param = open(path+'sex_stars.param', "w")
    file_param.write('NUMBER\n')
    file_param.write('X_IMAGE\n')
    file_param.write('Y_IMAGE\n')
    file_param.write('FLUX_AUTO\n')
    file_param.write('CLASS_STAR\n')
    file_param.close()
    os.system('cd '+path+ ' ; '+sex+' -c sex_stars.conf '+filename)
    data = np.genfromtxt(path+'sex_stars.cat',dtype=None,comments='#', names=['number', 'x', 'y', 'flux', 'classifier'])
    x_data = np.array(data['x'])
    y_data = np.array(data['y'])
    fluxes = np.array(data['flux'])
    star_class_data = np.array(data['classifier'])
    mask = (x_data >= 5 * fwhm / pixel_scale) & (y_data >= 5 * fwhm / pixel_scale) & (
        x_data <= imageshape[1] - 5 * fwhm / pixel_scale) & (y_data <= imageshape[0] - 5 * fwhm / pixel_scale)
    mask_negative = np.invert(mask)
    star_class_data[mask_negative] *= 0.0
    starmask = star_class_data >= 0.8
    stars_xcoords = x_data[starmask]
    stars_ycoords = y_data[starmask]
    fluxes = fluxes[starmask]
    return stars_xcoords, stars_ycoords, fluxes, np.max(star_class_data)>=0.8


def generate_sdss_psf(obj_line, psf_filename):
    home_dir = '/mnt/ds3lab/blaunet'
    psfTool_path = '%s/readAtlasImages-v5_4_11/read_PSF' % home_dir
    psfFields_dir_1 = '%s/psfFields' % home_dir
    psfFields_dir_2 = '/mnt/ds3lab/galaxian/source/sdss/dr12/psf-data'

    run = obj_line['run'].item()
    rerun = obj_line['rerun'].item()
    camcol = obj_line['camcol'].item()
    field = obj_line['field'].item()
    # psfField = '%s/%d/%d/objcs/%d/psField-%06d-%d-%04d.fit'%(psfFields_dir, rerun, run, camcol, run, camcol, field)

    psfField = '%s/psField-%06d-%d-%04d.fit' % (psfFields_dir_1, run, camcol, field)
    if not os.path.exists(psfField):
        psfField = '%s/%d/%d/objcs/%d/psField-%06d-%d-%04d.fit' % (
            psfFields_dir_1, rerun, run, camcol, run, camcol, field)
    if not os.path.exists(psfField):
        psfField = '%s/%d/%d/objcs/%d/psField-%06d-%d-%04d.fit' % (
            psfFields_dir_2, rerun, run, camcol, run, camcol, field)
    if not os.path.exists(psfField):
        raise FileNotFoundError('No psfField fit found')

    colc = obj_line['colc'].item()
    rowc = obj_line['rowc'].item()
    os.system('%s %s 3 %s %s %s' % (psfTool_path, psfField, rowc, colc, psf_filename))
    try:
        hdu = fits.open(psf_filename)
        psf_data = np.array(hdu[0].data, dtype=float) / 1000 - 1
        hdu.close()
        os.remove(psf_filename)
        hdu = fits.PrimaryHDU(psf_data)
        hdu.writeto(psf_filename)
    except:
        print('no psf %s' % psf_filename)
        print('psfField = %s' % psfField)
        print('rowc = %s' % rowc)
        print('colc = %s' % colc)


def add_sdss_PSF(origpath, original, psf_flux, obj_line, whitenoise_var = None, multiple=False, sexdir = None):
    SDSS_psf_dir = '%s/psf/SDSS' % conf.run_case
    GALFIT_psf_dir = '%s/psf/GALFIT' % conf.run_case
    if not os.path.exists(SDSS_psf_dir):
        os.makedirs(SDSS_psf_dir)
    if not os.path.exists(GALFIT_psf_dir):
        os.makedirs(GALFIT_psf_dir)

    obj_id = obj_line['dr7ObjID'].item()
    SDSS_psf_filename = '%s/%s-r.fits' % (SDSS_psf_dir, obj_id)
    GALFIT_psf_filename = '%s/%s-r.fits' % (GALFIT_psf_dir, obj_id)
    if not os.path.exists(GALFIT_psf_filename):
        # print('No Existing Galfit PSF')
        if not os.path.exists(SDSS_psf_filename):
            generate_sdss_psf(obj_line, SDSS_psf_filename)
        psf = galfit.fit_PSF_GALFIT(SDSS_psf_filename, GALFIT_psf_dir)
        if psf is None:
            print('Error in Galfit fit')
            return None
    else:
        psf = galfit.open_GALFIT_results(GALFIT_psf_filename, 'model')

    if sexdir:
        if not os.path.isdir(sexdir):
            os.makedirs(sexdir)
        try:
            nmgy_per_count = fits.getheader(origpath)['NMGY']
        except KeyError:
            nmgy_per_count = 0.0238446
        #ideally field_data would be the field but we don't have the (huge) data on the sgs machines...
        field_data = original
        hdu_output = fits.PrimaryHDU(field_data/nmgy_per_count)
        hdulist_output = fits.HDUList([hdu_output])
        hdulist_output.writeto(sexdir+'field_ADU.fits', overwrite=True)
        exptime = 53.9
        threshold = 5
        saturation_limit = 1500
        gain = 4.73
        pixel_scale = 0.396
        fwhm = 1.4
        zeropoint = calc_zeropoint(exptime, nmgy_per_count)
        x_coordinates, y_coordinates, fluxes, starboolean = SExtractor_get_stars(sexdir, 'field_ADU.fits', zeropoint, threshold, saturation_limit, gain, pixel_scale, fwhm, original.shape)
        if not starboolean:
            return None
        xcrds = np.array(x_coordinates)
        ycrds = np.array(y_coordinates)
        fluxes = np.array(fluxes)*nmgy_per_count
        #psf_flux is the flux the fake AGN must have (galaxy flux*cratio)
        fluxdistances = fluxes - psf_flux
        star_index = np.argmin(np.abs(fluxdistances))
        scale_factor = psf_flux / fluxes[star_index]
        crop_center = find_centroid(original, center=[xcrds[star_index], ycrds[star_index]])
        psf = scale_factor * crop_star(original, 2 * fwhm / pixel_scale, crop_center)
        files_to_delete = glob.glob(sexdir+'*')
        for f in files_to_delete:
            os.remove(f)
    else:
        # Scaling up
        psf = psf / psf.sum()
        psf = psf * psf_flux
    
    center = [original.shape[1] // 2, original.shape[0] // 2]
    centroid_galaxy = find_centroid(original)
    centroid_PSF = find_centroid(psf)
    
    # Whitenoise
    if whitenoise_var:
        whitenoise = np.random.normal(0, np.sqrt(whitenoise_var), (psf.shape[0], psf.shape[1]))
        print(whitenoise_var)
        psf = psf + whitenoise
     
    
    composite_image = np.copy(original)

    k = 3 if multiple else 1

    gal_x = int(round(centroid_galaxy[0]))
    gal_y = int(round(centroid_galaxy[1]))
    ps_x = int(round(centroid_PSF[0]))
    ps_y = int(round(centroid_PSF[1]))

    for x in range(0, psf.shape[1]):
        for y in range(0, psf.shape[0]):
            x_rel = gal_x - ps_x + x
            y_rel = gal_y - ps_y + y
            if x_rel >= 0 and y_rel >= 0 and x_rel < original.shape[1] and y_rel < original.shape[0]:
                composite_image[y_rel, x_rel] += psf[y, x]

    return composite_image


def add_gaussian_PSF(original, psf_flux, sigma):
    '''
    :param sigma: sigma of the gaussian distribution
    :return: normalised PSF of size size*size centered on [x0, y0]
    '''
    size = original.shape[0]
    x0, y0 = find_centroid(original)
    x, y = np.mgrid[0:size, 0:size]
    psf = np.exp(-(((x - y0) ** 2 + (y - x0) ** 2) / (2.0 * sigma ** 2)))

    return original + psf_flux * psf / psf.sum()


def add_step_PSF(original, psf_flux, sigma):
    size = original.shape[0]
    x0, y0 = find_centroid(original)
    psf = np.zeros((size, size))
    # psf[int(y0-sigma/2):int(y0+sigma/2)+1,int(x0-sigma/2):int(x0+sigma/2)+1]=1
    for i in range(size):
        for j in range(size):
            if (x0 - i) ** 2 + (y0 - j) ** 2 <= sigma ** 2:
                psf[j, i] = 1
    return original + psf_flux * psf / psf.sum()


def crop(img, cut):
    '''Crop the image to a cut*cut image centered on the center'''
    size = img.shape[0]
    return img[size / 2 - cut:size / 2 + cut, size / 2 - cut:size / 2 + cut]


def crop_star(img, cut, cc):
    return img[int(cc[1] - cut):int(cc[1] + cut + 1), int(cc[0] - cut):int(cc[0] + cut + 1)]



def find_centroid(img, cut=5, center=None):
    if not center:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        x_tmp, y_tmp = centroid_com(crop(img, cut))
    else:
        x_tmp, y_tmp = centroid_com(crop_star(img, cut, center))
    return [center[0] - cut + x_tmp, center[1] - cut + y_tmp]




# def find_centroid(im, guesslist=np.zeros(2), b_size=0):
#     '''
#     :param im: image data array
#     :param guesslist: array of coordinate tuples (as np.ndarray) in pixel coordinates, can only be provided, if b_size
#     does not vanish. The coordinates in guesslist are taken as centeres of the cutouts of size b_size.
#     :param b_size: size (in pixel coordinates) of image cutout used for calculation. Is set as length of im by
#     default.
#     :return: array of centroid tuples
#     '''
#     b_size = int(round(b_size))
#     if b_size == 0:
#
#         mean, median, std = sigma_clipped_stats(im, sigma = 3.0, iters=5)
#         #threshold = np.max(im) / 10.
#         threshold = 3*std
#         centroid_pos = (photutils.find_peaks(im, threshold, box_size=20, subpixel = True, npeaks = 1))
#
#
#     else:
#         crd = (map(int, guesslist))
#         region = im[ crd[1]-b_size : crd[1]+b_size , crd[0]-b_size : crd[0]+b_size ]
#         mean, median, std = sigma_clipped_stats(region, sigma = 3.0, iters=5)
#         threshold = 3*std
#         centroid_pos = photutils.find_peaks(region, threshold, box_size=b_size, subpixel = True , npeaks = 1)
#     x_ctr = (float(centroid_pos['x_centroid']) - b_size + int(guesslist[0]))
#     y_ctr = (float(centroid_pos['y_centroid']) - b_size + int(guesslist[1]))
#
#     return [x_ctr, y_ctr]
