import math
import numpy as np
class Config:

    #Redshift
    redshift = 0.05
    stretch_type = 'pow' #'linear' 'log' normalized_linear
    attention_parameter = 0.1
    #model_to_use = 'z_0.1'
    use_gpu = 0

    run_case = "/mnt/ds3lab/blaunet/results/z_%s"%(redshift)
    #run_case = "/mnt/ds3lab/blaunet/results/darg_late_stage"
    #Scaling
    if '0.01' in run_case:
        pixel_max_value = 41100
    elif '0.05' in run_case:
        pixel_max_value = 6140
    elif '0.1' in run_case:
        pixel_max_value = 1450
    elif 'darg_outliers' in run_case:
        pixel_max_value = 22000
    elif 'darg_late_stage' in run_case:
        pixel_max_value = 4000
    pixel_min_value = -0.1

    stretch_setup = '%s/%s_auto'%(run_case, stretch_type)
    sub_config = '%s/WGAN_%s'%(stretch_setup, attention_parameter)
    output_path = '%s/GAN_output'%(sub_config)
    result_path = output_path
    #result_path = '%s/%s_model'%(output_path, model_to_use)
    #used for training
    data_path = "%s/npy_input"%(stretch_setup)
    save_path =  "%s/model"%(sub_config)
    #if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ''#"/mnt/ds3lab/blaunet/results/%s/asinh_20/model/model.ckpt"%(model_to_use)
    start_epoch = 0

    #changed to FITs, mainly refer to the size
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    @classmethod
    def stretch(cls, data, factor):
        MAX = cls.pixel_max_value
        MIN = cls.pixel_min_value
        #data[data<MIN]=MIN
        #data[data>MAX]=MAX
        if cls.stretch_type == 'log':
            return np.log10(factor*((data - MIN)/(MAX - MIN))+1)/math.log10(factor)
        elif cls.stretch_type == 'asinh':
            return np.arcsinh(factor*data)/math.asinh(factor*MAX)

        elif cls.stretch_type == 'pow':
            return np.power((data - MIN)/(MAX - MIN),1/float(factor))
        elif cls.stretch_type == 'linear':
            return data/MAX
        elif cls.stretch_type == 'normalized_linear':
            return (data-MIN)/(MAX-MIN)
        else:
            raise ValueError('Unknown stretch_type : %s'%cls.stretch_type)

    @classmethod
    def unstretch(cls, data, factor):
        MAX = cls.pixel_max_value
        MIN = cls.pixel_min_value
        if cls.stretch_type == 'log':
            return MIN + (MAX - MIN)* (np.power(data*math.log10(factor), 10)-1)/factor
        elif cls.stretch_type == 'asinh':
            return np.sinh(data*math.asinh(factor*MAX))/factor
        elif cls.stretch_type == 'pow':
            return np.power(data,factor)*(MAX - MIN) + MIN
        elif cls.stretch_type == 'linear':
            return data*MAX
        elif cls.stretch_type == 'normalized_linear':
            return data*(MAX-MIN)+ MIN
        else:
            raise ValueError('Unknown stretch_type : %s'%cls.stretch_type)

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 50
    L1_lambda = 100
    sum_lambda = 0####
    save_per_epoch=1
