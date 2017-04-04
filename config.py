
# %load /home/blaunet/GalaxyGAN_python/config.py

import math
import numpy as np
class Config:
    #used for training	
    data_path = "./figures"
    save_path = "./model"

    #if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ""#./model/model.ckpt"

    start_epoch = 0
    output_path = "/home/blaunet/results/test"

    #used GPU
    use_gpu = 1

    #changed to FITs, mainly refer to the size
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    #Scaling
    pixel_max_value = 700#3500
    pixel_min_value = -0.1
    scale_factor = 20
    stretch_type = 'asinh'#'log' #'asinh'
    
    @classmethod
    def stretch(cls, data):
        if cls.stretch_type == 'log':
            return np.log10(cls.scale_factor*((data - cls.pixel_min_value)/(cls.pixel_max_value - cls.pixel_min_value))+1)/math.log10(cls.scale_factor)
        elif cls.stretch_type == 'asinh':
            return np.arcsinh(cls.scale_factor*data)/math.asinh(cls.scale_factor*cls.pixel_max_value)
        elif cls.stretch_type == '0':
            return data/cls.pixel_max_value
        else:
            raise ValueError('Unknown stretch_type : %s'%cls.stretch_type)
            
    @classmethod
    def unstretch(cls, data):
        if cls.stretch_type == 'log':
            return cls.pixel_min_value + (cls.pixel_max_value - cls.pixel_min_value)* (np.power(data*math.log10(cls.scale_factor), 10)-1)/cls.scale_factor
        elif cls.stretch_type == 'asinh':
            return np.sinh(data*math.asinh(cls.scale_factor*cls.pixel_max_value))/cls.scale_factor
        elif cls.stretch_type == '0':
            return data*cls.pixel_max_value
        else:
            raise ValueError('Unknown stretch_type : %s'%cls.stretch_type)

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    sum_lambda = 0####
    save_per_epoch=5
