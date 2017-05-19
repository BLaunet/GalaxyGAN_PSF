from subprocess import Popen
import fileinput
import os
import glob

def overwrite_config(file_path, z, factor, stretch, WGAN, gpu):
    for line in fileinput.input(file_path, inplace=True):
        if 'scale_factor = ' in line:
            newline = '    scale_factor = %s\n'%factor
        elif 'redshift = ' in line:
            newline = '    redshift = %s\n'%z
        elif 'stretch_type = ' in line:
            newline = '    stretch_type = \'%s\'\n'%stretch
        elif 'attention_parameter = ' in line:
            newline = '    attention_parameter = %s\n'%WGAN
        elif 'use_gpu =' in line:
            newline = '    use_gpu = %s\n'%gpu

        else:
            newline = line
        print newline,

main_path = '/mnt/ds3lab/blaunet'
file_path = '%s/GalaxyGAN_python/config.py'%main_path
scale_factor_list = [50]
z_list = [0.1]
stretch_list = ['asinh']
WGAN_list = [1,10,100]
gpu = 7

for z in z_list:
    for scale_f in scale_factor_list:
        for stretch in stretch_list:
            for WGAN in WGAN_list:
		## Roou
                npy_input_dir = '%s/results/z_%s/%s_%s/npy_input'%(main_path, z, stretch, scale_f)
                if not (glob.glob('%s/test/*'%npy_input_dir)):
                    overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu)
                    os.system('. ../my_env/bin/activate; python -u roou.py | tee tmp.log')

                overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu)
                if not (glob.glob('%s/train/*'%npy_input_dir)):
                    overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu)
                    os.system('. ../my_env/bin/activate; python -u roou.py --mode 0| tee tmp.log')

                overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu)
                os.system('. ../my_env/bin/activate; python -u train.py | tee %s/results/z_%s/%s_%s/WGAN_%s.log'%(main_path, z, stretch, scale_f, WGAN))
