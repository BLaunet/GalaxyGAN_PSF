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


def generate_sdss_psf(obj_line, psf_filename):
    home_dir = '/mnt/ds3lab/blaunet'
    home_dir_1 = '/mnt/ds3lab/dostark'
    psfTool_path = '%s/readAtlasImages-v5_4_11/read_PSF' % home_dir
    psfFields_dir_1 = '%s/psfFields' % home_dir_1
    psfFields_dir_2 = '/mnt/ds3lab/galaxian/source/sdss/dr12/psf-data'

    filter_string = conf.filter_
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
    filter_dic = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5}
    os.system('%s %s %s %s %s %s' % (psfTool_path, psfField, filter_dic[filter_string], rowc, colc, psf_filename))
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


def add_sdss_PSF(origpath, original, psf_flux, obj_line, whitenoise_var=None, multiple=False, sexdir=None,
                 median_combine=False):
    SDSS_psf_dir = '%s/psf/SDSS' % conf.run_case
    GALFIT_psf_dir = '%s/psf/GALFIT' % conf.run_case
    filter_string = conf.filter_
    if not os.path.exists(SDSS_psf_dir):
        os.makedirs(SDSS_psf_dir)
    if not os.path.exists(GALFIT_psf_dir):
        os.makedirs(GALFIT_psf_dir)

    obj_id = obj_line['dr7ObjID'].item()
    SDSS_psf_filename = '%s/%s-%s.fits' % (SDSS_psf_dir, obj_id, filter_string)
    GALFIT_psf_filename = '%s/%s-%s.fits' % (GALFIT_psf_dir, obj_id, filter_string)
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
        # ideally field_data would be the field but we don't have the (huge) data on the sgs machines...
        # field_data = original
        field_data = get_field(obj_line)
        if field_data is None:
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
        # x_coordinates, y_coordinates, fluxes, starboolean = SExtractor_get_stars(sexdir, 'field_ADU.fits', zeropoint, threshold, saturation_limit, gain, pixel_scale, fwhm, field_data.shape, 46)
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
        if median_combine:
            x_crds = np.array(x_coordinates)
            y_crds = np.array(y_coordinates)
            csize = 40
            fluxes = np.array(fluxes)
            fluxmask = fluxes >= np.max(fluxes) * 0.5
            # psf_unscaled = empirical_PSF_manual(field_data, x_coordinates, y_coordinates, int(round(csize)), int(round(fwhm/pixel_scale)))
            # fluxes = np.array(fluxes)*nmgy_per_count
            # psf_flux is the flux the fake AGN must have (galaxy flux*cratio)
            # fluxdistances = fluxes - psf_flux
            # star_index = np.argmin(np.abs(fluxdistances))
            # fluxmask = fluxes == fluxes[star_index]
            # fluxmask = (fluxes >= psf_flux - 1.2*np.min(np.abs(fluxdistances))) & (fluxes <= psf_flux +1.2*np.min(np.abs(fluxdistances)))
            # psf_unscaled = crop_star(field_data, cut=40, cc=[x_crds[fluxmask][0], y_crds[fluxmask][0]])
            psf_unscaled = empirical_PSF_manual(field_data, x_crds[fluxmask], y_crds[fluxmask], csize,
                                                fwhm / pixel_scale, obj_id)
            psf_centroid = find_centroid(psf_unscaled)
            statmask = photutils.make_source_mask(psf_unscaled, snr=5, npixels=5, dilate_size=10)
            bkg_annulus = photutils.CircularAnnulus(psf_centroid, 3 * fwhm / pixel_scale, 20)
            bkg_phot_table = photutils.aperture_photometry(psf_unscaled, bkg_annulus, method='subpixel', mask=statmask)
            bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
            src_aperture = photutils.CircularAperture(psf_centroid, 3 * fwhm / pixel_scale)
            src_phot_table = photutils.aperture_photometry(psf_unscaled, src_aperture, method='subpixel')
            flux_photutils = src_phot_table['aperture_sum'] - bkg_mean_per_pixel * src_aperture.area()
            scale_factor = psf_flux / flux_photutils
            '''if scale_factor > 6:
                files_to_delete = glob.glob(sexdir+'*')
                for f in files_to_delete:
                    os.remove(f)
                return None
            '''
            # psf = scale_factor*(psf_unscaled - bkg_mean_per_pixel)
            crop_center = find_centroid(psf_unscaled, center=psf_centroid)
            psf = scale_factor * crop_star(psf_unscaled - bkg_mean_per_pixel, 2 * fwhm / pixel_scale, crop_center)

        else:
            xcrds = np.array(x_coordinates)
            ycrds = np.array(y_coordinates)
            fluxes = np.array(fluxes) * nmgy_per_count
            # psf_flux is the flux the fake AGN must have (galaxy flux*cratio)
            fluxdistances = fluxes - psf_flux
            star_index = np.argmin(np.abs(fluxdistances))
            statmask = photutils.make_source_mask(field_data, snr=5, npixels=5, dilate_size=10)
            bkg_annulus = photutils.CircularAnnulus((xcrds[star_index], ycrds[star_index]), 3 * fwhm / pixel_scale, 20)
            bkg_phot_table = photutils.aperture_photometry(field_data, bkg_annulus, method='subpixel', mask=statmask)
            bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
            src_aperture = photutils.CircularAperture((xcrds[star_index], ycrds[star_index]), 3 * fwhm / pixel_scale)
            src_phot_table = photutils.aperture_photometry(field_data, src_aperture, method='subpixel')
            flux_photutils = src_phot_table['aperture_sum'] - bkg_mean_per_pixel * src_aperture.area()
            # scale_factor = psf_flux / fluxes[star_index]
            scale_factor = psf_flux / flux_photutils
            crop_center = find_centroid(field_data, center=[xcrds[star_index], ycrds[star_index]])
            psf = scale_factor * crop_star(field_data - bkg_mean_per_pixel, 2 * fwhm / pixel_scale, crop_center)

        files_to_delete = glob.glob(sexdir + '*')
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


def rebin_oversample(data, factor):
    int_factor = int(round(factor))
    new_x = int(round(data.shape[1] * int_factor))
    new_y = int(round(data.shape[0] * int_factor))
    new_data = np.zeros((new_y, new_x))
    for i in range(0, new_x):
        for j in range(0, new_y):
            new_data[j, i] = data[j / int_factor, i / int_factor]
    return new_data


def rebin_undersample(data, factor):
    int_factor = int(round(factor))
    assert (data.shape[0] % int_factor == 0)
    new_x = int(data.shape[1] / int_factor)
    new_y = int(data.shape[0] / int_factor)
    new_data = np.zeros((new_y, new_x))
    for i in range(0, new_x):
        for j in range(0, new_y):
            new_data[j, i] = np.mean(data[j * factor:(j + 1) * factor, i * factor: (i + 1) * factor])
    return new_data


def weighted_median(data, weights):
    '''
    :param data: 1d numpy array of data
    :param weights: 1d numpy array of weights
    :return: weighted median according to https://en.wikipedia.org/wiki/Weighted_median
    '''
    weights_normalized = weights / weights.sum()
    sorted_data = np.sort(data)
    weights_parallel_sorted = weights_normalized[data.argsort()]
    sum = 0
    for i in range(0, len(weights_parallel_sorted)):
        actual_weight = weights_parallel_sorted[i]
        if sum + actual_weight >= 0.5:
            if sum + actual_weight > 0.5:
                return sorted_data[i]
            elif sum + actual_weight == 0.5:
                return 0.5 * (sorted_data[i + 1] + sorted_data[i])
        sum += actual_weight


def median_stacking(images):
    return np.median(np.array(images), axis=0)


def weighted_median_stacking(images, weights):
    imshape = images[0].shape
    combined_image = np.zeros(imshape)
    for x in range(0, imshape[1]):
        for y in range(0, imshape[0]):
            im_tmp = []
            w_tmp = []
            for k in range(0, len(images)):
                im_tmp.append(images[k][y, x])
                # w_tmp.append(images[k][y,x]/(std**2+images[k][y, x]/gain))  # weights of S/N^2
            combined_image[y, x] = weighted_median(np.array(im_tmp), np.array(weights))
    return combined_image


def empirical_PSF_manual(data, x_icords, y_icords, cutout_size, fwhm_pix, objid, starplotting=False,
                         plotting_path='no given directory as plot_path'):
    # Rough cutout of stars (with background!) and weights (S/N^2) calculation.
    cutouts = []
    weights = []
    # cutout_size_tmp = cutout_size + 5
    cutout_size_tmp = cutout_size + 5
    for j in range(0, len(x_icords)):
        cutouts.append(data[int(y_icords[j]) - cutout_size_tmp:int(y_icords[j]) + cutout_size_tmp + 1,
                       int(x_icords[j]) - cutout_size_tmp:int(x_icords[j]) + cutout_size_tmp + 1])
        S, N2, local_bkg = starphot(cutouts[-1], [cutout_size_tmp, cutout_size_tmp], fwhm_pix * 3, r_in=fwhm_pix * 3,
                                    r_out=cutout_size_tmp - 1)
        # cutouts[-1] = np.array(cutouts[-1])
        weights.append(S / N2)

    # rebinning and recentering
    cutouts_rebinned = []
    cutout_size = cutout_size * 10
    i = 0
    weights_final = []
    for image in cutouts:
        # hdu_output1 = fits.PrimaryHDU(image)
        # hdulist_output1 = fits.HDUList([hdu_output1])
        # hdulist_output1.writeto('/mnt/ds3lab/dostark/testdir/'+str(objid)+'.fits', overwrite=True)
        # star_centroid = find_centroid(image)
        star_centroid = [x_icords[i] - int(x_icords[i]) + cutout_size_tmp,
                         y_icords[i] - int(y_icords[i]) + cutout_size_tmp]
        star_centroid = np.array(star_centroid) * 10
        tmp = rebin_oversample(image, 10)
        # tmp = image
        if star_centroid[0] > cutout_size and tmp.shape[1] - cutout_size > star_centroid[0] and star_centroid[
            1] > cutout_size and tmp.shape[0] - cutout_size > star_centroid[1]:
            cutouts_rebinned.append(
                tmp[int(round(star_centroid[1])) - cutout_size:int(round(star_centroid[1])) + cutout_size + 10,
                int(round(star_centroid[0])) - cutout_size:int(round(star_centroid[0])) + cutout_size + 10])
            weights_final.append(weights[i])
        i += 1
    # combine using the weighted median with weights of S/N^2. Furthermore the image is rebinned back to the original grid.
    imcombined = weighted_median_stacking(cutouts_rebinned, weights_final)
    # imcombined = median_stacking(cutouts_rebinned)
    imcombined_result = rebin_undersample(imcombined, 10)
    # imcombined_result = imcombined
    file_res = open('/mnt/ds3lab/dostark/count_stars.csv', "a")
    file_res.write(str(objid) + ',' + str(len(weights_final)) + ',' + str(len(cutouts_rebinned)) + '\n')
    file_res.close()
    hdu_output = fits.PrimaryHDU(imcombined_result)
    hdulist_output = fits.HDUList([hdu_output])
    hdulist_output.writeto('/mnt/ds3lab/dostark/testdir/' + str(objid) + '_mcombined.fits', overwrite=True)
    hdulist_output.close()
    return imcombined_result

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
