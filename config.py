import math
import numpy as np


class Config:
    # Training Set Parameters
    redshift = 0.05
    stretch_type = 'pow'
    scale_factor = 8
    attention_parameter = 0.05
    max_contrast_ratio = 10
    noise = 0
    filter_ = 'r'
    run_case = "/mnt/ds3lab/blaunet/results/z_%s" % (redshift)
    # run_case = "/mnt/ds3lab/blaunet/results/darg_late_stage"
    # Scaling
    if '0.01' in run_case:
        pixel_max_value = 41100
    elif '0.05' in run_case:
        pixel_max_value = 6140
    elif '0.1' in run_case:
        pixel_max_value = 1450
    elif '0.2' in run_case:
        pixel_max_value = 1657
    elif 'darg_outliers' in run_case:
        pixel_max_value = 22000
    elif 'darg_late_stage' in run_case:
        pixel_max_value = 4000
    pixel_min_value = -0.1

    ## Directory Tree setup
    #run_case = "/mnt/ds3lab/dostark/z_%s/%s-band" % (redshift, filter_)
    # run_case = "/mnt/ds3lab/blaunet/results/darg_late_stage"
    ext = ''
    if max_contrast_ratio != 10:
        ext += '_ratio_%s' % max_contrast_ratio
    if noise != 0:
        ext += '_noise_%s' % noise

    stretch_setup = '%s/%s_%s%s' % (run_case, stretch_type, scale_factor, ext)
    sub_config = '%s/WGAN_%s' % (stretch_setup, attention_parameter)
    output_path = '%s/GAN_output' % sub_config
    result_path = output_path

    data_path = "%s/npy_input" % stretch_setup
    save_path = "%s/model" % sub_config

    ## GAN Parameters
    use_gpu = 0
    # if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ''  # /mnt/ds3lab/dostark/z_0.1/r-band/pow_8_mcombined_stars/WGAN_0.05/model/model.ckpt  or  "/mnt/ds3lab/blaunet/results/%s/asinh_20/model/model.ckpt"%(model_to_use)
    start_epoch = 0
    save_per_epoch = 5
    max_epoch = 50

    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    L1_lambda = 100
    sum_lambda = 0  ####

    @classmethod
    def stretch(cls, data):
        if cls.stretch_type == 'log':
            return np.log10(cls.scale_factor * (
                (data - cls.pixel_min_value) / (cls.pixel_max_value - cls.pixel_min_value)) + 1) / math.log10(
                cls.scale_factor)
        elif cls.stretch_type == 'asinh':
            return np.arcsinh(cls.scale_factor * data) / math.asinh(cls.scale_factor * cls.pixel_max_value)

        elif cls.stretch_type == 'pow':
            return np.power((data - cls.pixel_min_value) / (cls.pixel_max_value - cls.pixel_min_value),
                            1 / float(cls.scale_factor))
        elif cls.stretch_type == 'linear':
            return data / cls.pixel_max_value
        elif cls.stretch_type == 'normalized_linear':
            return (data - cls.pixel_min_value) / (cls.pixel_max_value - cls.pixel_min_value)

        elif cls.stretch_type == 'sigmoid':
            return (1 / (1 + np.exp(-cls.scale_factor * np.sqrt(
                (data - cls.pixel_min_value) / (cls.pixel_max_value - cls.pixel_min_value)))) - 1 / 2) * 2
        else:
            raise ValueError('Unknown stretch_type : %s' % cls.stretch_type)

    @classmethod
    def unstretch(cls, data):
        if cls.stretch_type == 'log':
            return cls.pixel_min_value + (cls.pixel_max_value - cls.pixel_min_value) * (
                np.power(data * math.log10(cls.scale_factor), 10) - 1) / cls.scale_factor
        elif cls.stretch_type == 'asinh':
            return np.sinh(data * math.asinh(cls.scale_factor * cls.pixel_max_value)) / cls.scale_factor
        elif cls.stretch_type == 'pow':
            return np.power(data, cls.scale_factor) * (cls.pixel_max_value - cls.pixel_min_value) + cls.pixel_min_value
        elif cls.stretch_type == 'linear':
            return data * cls.pixel_max_value
        elif cls.stretch_type == 'normalized_linear':
            return data * (cls.pixel_max_value - cls.pixel_min_value) + cls.pixel_min_value
        elif cls.stretch_type == 'sigmoid':
            return np.square(np.log(-1 + 2 / (data + 1)) / cls.scale_factor) * (
                cls.pixel_max_value - cls.pixel_min_value) + cls.pixel_min_value

        else:
            raise ValueError('Unknown stretch_type : %s' % cls.stretch_type)
