import os

import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm
from photutils import centroid_com
import photutils
from astropy.stats import sigma_clipped_stats
import galfit
from config import Config as conf
import matplotlib.pyplot as plt

from shutil import copyfile
import glob

sex = 'sextractor'


def calc_zeropoint(exposure_time, calibration_factor):
    return 22.5 + 2.5 * np.log10(1. / exposure_time / calibration_factor)


def SExtractor_get_stars(path, filename, magzero, threshold, saturation_level, gain, pixel_scale, fwhm, imageshape,
                         edge, mindist):
    file_res = open(path + 'sex_stars.conf', "w")
    file_res.write('#-------------------------------- Catalog ------------------------------------\n\n')
    file_res.write('CATALOG_NAME     sex_stars.cat        # name of the output catalog\n')
    file_res.write('CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n')
    file_res.write('                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC \n')
    file_res.write('PARAMETERS_NAME  {}{}           # name of the file containing catalog contents \n\n'.format(path,
                                                                                                                'sex_stars.param'))
    file_res.write('#------------------------------- Extraction ----------------------------------\n\n')
    file_res.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)\n')
    file_res.write('DETECT_MINAREA   5              # min. # of pixels above threshold\n')
    file_res.write('DETECT_THRESH    5              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n')
    file_res.write(
        'ANALYSIS_THRESH  {}             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n\n'.format(threshold))
    file_res.write('FILTER           Y              # apply filter for detection (Y or N)?\n')
    file_res.write(
        'FILTER_NAME      /mnt/ds3lab/dostark/sextractor_defaultfiles/default.conv   # name of the file containing the filter\n\n')
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
    file_res.write(
        'SATUR_LEVEL      {}             # level (in ADUs) at which arises saturation\n'.format(saturation_level))
    file_res.write('SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)\n\n')
    file_res.write('MAG_ZEROPOINT    {}             # magnitude zero-point\n'.format(magzero))
    file_res.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)\n')
    file_res.write('GAIN             {}             # detector gain in e-/ADU\n'.format(gain))
    file_res.write('GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU\n')
    file_res.write(
        'PIXEL_SCALE      {}             # size of pixel in arcsec (0=use FITS WCS info)\n\n'.format(pixel_scale))
    file_res.write('# ------------------------- Star/Galaxy Separation ----------------------------\n\n')
    file_res.write('SEEING_FWHM      {}             # stellar FWHM in arcsec\n'.format(fwhm))
    file_res.write(
        'STARNNW_NAME     /mnt/ds3lab/dostark/sextractor_defaultfiles/default.nnw  # Neural-Network_Weight table filename\n\n')
    file_res.write('# ------------------------------ Background -----------------------------------\n\n')
    file_res.write('BACK_SIZE        64              # Background mesh: <size> or <width>,<height>\n')
    file_res.write('BACK_FILTERSIZE  3               # Background filter: <size> or <width>,<height>\n\n')
    file_res.write('BACKPHOTO_TYPE   GLOBAL          # can be GLOBAL or LOCAL\n\n')
    file_res.write('#------------------------------ Check Image ----------------------------------\n\n')
    file_res.write('CHECKIMAGE_TYPE  NONE            # can be NONE, BACKGROUND, BACKGROUND_RMS,\n')
    file_res.write('                                 # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,\n')
    file_res.write('                                 # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,\n')
    file_res.write('                                 # or APERTURES\n')
    file_res.write(
        'CHECKIMAGE_NAME  /mnt/ds3lab/dostark/sextractor_defaultfiles/check.fits     # Filename for the check-image\n\n')
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

    file_param = open(path + 'sex_stars.param', "w")
    file_param.write('NUMBER\n')
    file_param.write('X_IMAGE\n')
    file_param.write('Y_IMAGE\n')
    file_param.write('FLUX_AUTO\n')
    file_param.write('CLASS_STAR\n')
    file_param.close()
    os.system('cd ' + path + ' ; ' + sex + ' -c sex_stars.conf ' + filename)
    try:
        data = np.genfromtxt(path + 'sex_stars.cat', dtype=None, comments='#',
                             names=['number', 'x', 'y', 'flux', 'classifier'])
    except IOError:
        return [], [], [], False
    x_data = np.array(data['x'])
    y_data = np.array(data['y'])
    fluxes = np.array(data['flux'])
    star_class_data = np.array(data['classifier'])
    mask = (x_data >= edge) & (y_data >= edge) & (
        x_data <= imageshape[1] - edge) & (y_data <= imageshape[0] - edge)
    mask_negative = np.invert(mask)
    star_class_data[mask_negative] *= 0.0
    for i in range(0, len(x_data)):
        for j in range(i + 1, len(x_data)):
            if np.sqrt((x_data[i] - x_data[j]) ** 2 + (y_data[i] - y_data[j]) ** 2) < mindist:
                star_class_data[i] = 0
                star_class_data[j] = 0
    starmask = star_class_data >= 0.9
    stars_xcoords = x_data[starmask]
    stars_ycoords = y_data[starmask]
    fluxes = fluxes[starmask]
    return stars_xcoords, stars_ycoords, fluxes, np.max(star_class_data) >= 0.9


def get_field(obj_line):
    core_path = '/mnt/ds3lab/dostark/galaxian/source/sdss/dr12/plates/'
    filter_string = conf.filter_
    run = obj_line['run'].item()
    rerun = obj_line['rerun'].item()
    camcol = obj_line['camcol'].item()
    field = obj_line['field'].item()
    relative_path = '%s/%s' % (run, camcol)

    if not os.path.isfile('%s%s/sdss%s_dr12_%s-%s.fits.bz2' % (core_path, relative_path, filter_string, run, field)):
        os.system(
            'cd %s; rsync --relative -avzh dostark@plompy.ethz.ch:/home/galaxian/source/sdss/dr12/plates/./%s/sdss%s_dr12_%s-%s.fits.bz2 .' % (
            core_path, relative_path, filter_string, run, field))
    os.system('cd %s%s; bzip2 -dk sdss%s_dr12_%s-%s.fits.bz2' % (core_path, relative_path, filter_string, run, field))
    try:
        data = fits.getdata('%s%s/sdss%s_dr12_%s-%s.fits' % (core_path, relative_path, filter_string, run, field))
        os.system('rm %s%s/sdss%s_dr12_%s-%s.fits' % (core_path, relative_path, filter_string, run, field))
    except IOError:
        print('file does not exist')
        return None
    return data


def add_star_PSF(origpath, original, psf_flux, obj_line, whitenoise_var=None, multiple=False, sexdir=None,
                 median_combine=False):

    obj_id = obj_line['dr7ObjID'].item()

    ## First we find stars in the same field

    if not os.path.isdir(sexdir):
        os.makedirs(sexdir)
    try:
        nmgy_per_count = fits.getheader(origpath)['NMGY']
    except KeyError:
        nmgy_per_count = 0.0238446
    # ideally field_data would be the field but we don't have the (huge) data on the sgs machines...
    # field_data = original
    field_data = get_field(obj_line)
    if not field_data:
        return None
    hdu_output = fits.PrimaryHDU(field_data / nmgy_per_count)
    hdulist_output = fits.HDUList([hdu_output])
    hdulist_output.writeto(sexdir + 'field_ADU.fits', overwrite=True)
    exptime = 53.9
    threshold = 5
    saturation_limit = 1000
    gain = 4.73
    pixel_scale = 0.396
    fwhm = 1.4
    zeropoint = calc_zeropoint(exptime, nmgy_per_count)
    sex_edge = 46

    x_coordinates, y_coordinates, fluxes, starboolean = SExtractor_get_stars(sexdir, 'field_ADU.fits', zeropoint,
                                                                             threshold, saturation_limit, gain,
                                                                             pixel_scale, fwhm, field_data.shape,
                                                                             sex_edge, mindist=30)
    if not starboolean:
        files_to_delete = glob.glob(sexdir + '*')
        for f in files_to_delete:
            os.remove(f)
        return None


    ## For each star, we fit a 2D gaussian
    csize = 40
    fluxmask = fluxes >= np.max(fluxes) * 0.5
    sigma= get_semiempirical_param(field_data, x_coordinates[fluxmask], y_coordinates[fluxmask], csize,
                                        fwhm / pixel_scale, obj_id)
    composite_image = add_gaussian_PSF(original, psf_flux, sigma)

    files_to_delete = glob.glob(sexdir + '*')
    for f in files_to_delete:
        os.remove(f)

    return composite_image


def add_gaussian_PSF(original, psf_flux, sigma):
    '''
    :param sigma: sigma of the gaussian distribution
    :return: normalised PSF of size size*size centered on [x0, y0]
    '''
    size = original.shape[0]
    x0, y0 = find_centroid(original)
    x, y = np.mgrid[0:size, 0:size]
    psf = np.exp(- ( ((x - y0)**2)/(sigma[0]**2) + ((y - x0) ** 2)/(sigma[1]**2))/2)

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


def starphot(imdata, position, radius, r_in, r_out, plotting=False, plotfilename=None, plotpath=None):
    '''
    sources: http://spiff.rit.edu/classes/phys373/lectures/signal/signal_illus.html, http://photutils.readthedocs.io/en/stable/photutils/aperture.html
    '''
    statmask = photutils.make_source_mask(imdata, snr=5, npixels=5, dilate_size=10)
    bkg_annulus = photutils.CircularAnnulus(position, r_in, r_out)
    bkg_phot_table = photutils.aperture_photometry(imdata, bkg_annulus, method='subpixel', mask=statmask)
    bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
    src_aperture = photutils.CircularAperture(position, radius)
    src_phot_table = photutils.aperture_photometry(imdata, src_aperture, method='subpixel')
    signal = src_phot_table['aperture_sum'] - bkg_mean_per_pixel * src_aperture.area()
    # noise_squared = signal + bkg_mean_per_pixel*src_aperture.area()
    mean, median, std = sigma_clipped_stats(imdata, sigma=3.0, iters=5, mask=statmask)
    noise_squared = std ** 2
    if plotting == True:
        fig, ax = plt.subplots()
        ax.imshow(imdata, cmap='gray_r', origin='lower', norm=LogNorm())
        src_aperture.plot(ax=ax)
        bkg_annulus.plot(ax=ax)
        plt.title(plotfilename)
        plt.savefig(plotpath + plotfilename)
        plt.close()
    return float(str(signal.data[0])), noise_squared, float(str(bkg_mean_per_pixel.data[0]))


def get_semiempirical_param(data, x_icords, y_icords, cutout_size, fwhm_pix, objid, starplotting=False,
                         plotting_path='no given directory as plot_path'):
    # Rough cutout of stars (with background!) and weights (S/N^2) calculation.
    weights = []
    x_std = []
    y_std = []
    for j in range(0, len(x_icords)):
        cutout = crop_star(data, cutout_size, [x_icords[j], y_icords[j]])
        S, N2, local_bkg = starphot(cutout, [cutout_size, cutout_size], fwhm_pix * 3, r_in=fwhm_pix * 3,
                                    r_out=cutout_size - 1)
        weights.append(S / N2)
        gauss_2Dmodel = photutils.centroids.fit_2dgaussian(cutout)
        x_std.append(gauss_2Dmodel.x_stddev)
        y_std.append(gauss_2Dmodel.y_stddev)
    x_sigma = sum([x_std[i]*weights[i] for i in range(len(x_std))]) / sum(weights)
    y_sigma = sum([y_std[i]*weights[i] for i in range(len(y_std))]) / sum(weights.sum())
    return [x_sigma, y_sigma]
