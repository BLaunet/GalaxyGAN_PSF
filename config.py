import math
import numpy as np
class Config:
    #used for training	
    data_path = "./figures"
    save_path = "./model"

    #if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ""#./model/model.ckpt"

    start_epoch = 0
    output_path = "./result"

    #used GPU
    use_gpu = 1

    #changed to FITs, mainly refer to the size
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    #Scaling
    pixel_max_value = 700
    pixel_min_value = -0.1
    scale_factor = 1000
    stretch_type = 'log' #'asinh'
    def unstretch(self, data):
        if self.stretch_type == 'log':
            return self.pixel_min_value + (self.pixel_max_value - self.pixel_min_value)* (np.pow(data*math.log10(self.scale_factor), 10)-1)/self.scale_factor
        elif self.stretch_type == 'asinh':
            return np.sinh(data*math.asinh(conf.scale_factor*conf.pixel_max_value))/conf.scale_factor
        else:
            raise ValueError('Unknown stretch_type : %s'%self.stretch_type)

    def stretch(self,data):
        if self.stretch_type = 'log':
            return np.log10(conf.scale_factor*((data - conf.pixel_min_value)/(conf.pixel_max_value - conf.pixel_min_value))+1)/math.log10(conf.scale_factor)
        elif self.stretch_type == 'asinh':
            return np.asinh(conf.scale_factor*data)/math.asinh(conf.scale_factor*conf.pixel_max_value)

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    sum_lambda = 0####
    save_per_epoch=5
