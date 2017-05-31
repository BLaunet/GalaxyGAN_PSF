#!/usr/bin/python
import argparse
from config import Config as conf
from data import *
import scipy.misc
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
from utils import imsave
from astropy.io import fits

parser = argparse.ArgumentParser()

def prepocess_test(img, cond):

    #img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    #cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    #img = img/127.5 - 1.
    #cond = cond/127.5 - 1.
    return img,cond

def test():

    parser.add_argument("--input", default=conf.data_path)
    parser.add_argument("--model", default=conf.model_path)
    parser.add_argument("--out", default=conf.result_path)
    args = parser.parse_args()

    model_path = args.model
    data_path = args.input
    out_dir = args.out

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_data(data_path)
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    counter = 0

    with tf.Session() as sess:
        saver.restore(sess, conf.model_path)
        test_data = data["test"]()
        for img, cond, name in test_data:
            name = name.replace('-r.npy','')
            print(name)
            pimg, pcond = prepocess_test(img, cond)
            gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
            gen_img = gen_img.reshape(gen_img.shape[1:])

            fits_recover = conf.unstretch(gen_img[:,:,0])
            hdu = fits.PrimaryHDU(fits_recover)
            save_dir = '%s/fits_output'%(out_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = '%s/%s-r.fits'%(save_dir,name)
            if os.path.exists(filename):
                os.remove(filename)
            hdu.writeto(filename)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf.use_gpu)
    test()
