import fileinput
import glob
import os


def overwrite_config(file_path, z, factor, stretch, WGAN, noise, contrast_ratio, gpu):
    for line in fileinput.input(file_path, inplace=True):
        if 'scale_factor = ' in line:
            newline = '    scale_factor = %s\n' % factor
        elif 'redshift = ' in line:
            newline = '    redshift = %s\n' % z
        elif 'stretch_type = ' in line:
            newline = '    stretch_type = \'%s\'\n' % stretch
        elif 'attention_parameter = ' in line:
            newline = '    attention_parameter = %s\n' % WGAN
        elif 'use_gpu =' in line:
            newline = '    use_gpu = %s\n' % gpu
        elif 'max_contrast_ratio =' in line:
            newline = '    max_contrast_ratio = %s\n' % contrast_ratio
        elif 'noise =' in line:
            newline = '    noise = %s\n' % noise
        else:
            newline = line
        print newline,


main_path = '/mnt/ds3lab/blaunet'
file_path = '%s/GalaxyGAN_python/config.py' % main_path
scale_factor_list = [5, 8]
z_list = [0.2]
stretch_list = ['pow']
WGAN_list = [0.05]
ratio_list = [20]
noise_list = [10]

gpu = 0

for z in z_list:
    for scale_f in scale_factor_list:
        for stretch in stretch_list:
            for WGAN in WGAN_list:
                for ratio in ratio_list:
                    for noise in noise_list:
                        ext = ''
                        if ratio != 10:
                            ext += '_ratio_%s' % ratio
                        if noise != 0:
                            ext += '_noise_%s' % noise
                        ## Roou
                        npy_input_dir = '%s/results/z_%s/%s_%s%s/npy_input' % (main_path, z, stretch, scale_f, ext)
                        if not (glob.glob('%s/test/*' % npy_input_dir)):
                            overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu, contrast_ratio=ratio, noise=noise)
                            os.system('. ../my_env/bin/activate; python -u roou.py | tee tmp.log')

                        
                        if not (glob.glob('%s/train/*' % npy_input_dir)):
                            overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu, contrast_ratio=ratio, noise=noise)
                            os.system('. ../my_env/bin/activate; python -u roou.py --mode 0| tee tmp.log')

                        overwrite_config(file_path, z=z, factor=scale_f, stretch=stretch, WGAN=WGAN, gpu=gpu, contrast_ratio=ratio, noise=noise)
                        os.system(
                            '. ../my_env/bin/activate; python -u train.py | tee %s/results/z_%s/%s_%s%s/WGAN_%s.log' % (
                                main_path, z, stretch, scale_f, ext, WGAN))
