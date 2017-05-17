
from config import Config as conf
from data import *
import scipy.misc
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
import os
from astropy.io import fits
import math
import glob
import shutil

def prepocess_train(img, cond):
    #img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    #cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    #h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    #w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.adjust_size)))
    #img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    #cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]

    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)

    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    return img,cond

def prepocess_test(img, cond):
    #img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    #cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    #img = img/127.5 - 1.
    #cond = cond/127.5 - 1.
    return img,cond

def train():


    data = load_data()
    model = CGAN()
    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    ##TensorBoard setup
    tf.summary.scalar('d_loss', model.d_loss)
    tf.summary.scalar('g_loss', model.g_loss)
    tf.summary.scalar('flux', model.delta)
    tf.summary.tensor_summary('scale_factor', model.scale_factor)
    merged = tf.summary.merge_all()

    summary_test_folder = '%s/summary/test'%conf.sub_config
    summary_train_folder = '%s/summary/train'%conf.sub_config
    shutil.rmtree(summary_test_folder)
    shutil.rmtree(summary_train_folder)
    os.makedirs(summary_test_folder)
    os.makedirs(summary_train_folder)
    generated_images = {name:tf.summary.image(name, model.gen_img) for _,_,name in data["test"]()}
    test_writer = tf.summary.FileWriter(summary_test_folder)
    ##
    
    counter = 0
    start_time = time.time()
    out_dir = conf.result_path
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    start_epoch = 0
    with tf.Session() as sess:
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
            try:
                log = open(conf.save_path + "/log")
                start_epoch = int(log.readline())
                log.close()
            except:
                pass
        train_writer = tf.summary.FileWriter(summary_train_folder, sess.graph)
        for epoch in xrange(start_epoch, conf.max_epoch):
            train_data = data["train"]()
            for img, cond, name in train_data:

                img, cond = prepocess_train(img, cond)
                print(sess.run(model.scale_factor))
                #img = sess.run(model.img_str, feed_dict={model.image:img, model.cond:cond})
                #cond = sess.run(model.cond_str, feed_dict={model.image:img, model.cond:cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M= sess.run([g_opt, model.g_loss], feed_dict={model.image:img, model.cond:cond})
                #generated = sess.run(model.gen_img, feed_dict={model.image:img, model.cond:cond})
                
                summary = sess.run(merged, feed_dict={model.image:img, model.cond:cond})
                counter += 1
                train_writer.add_summary(summary,counter)
                print('ObjID = %s'%name)
                print "Iterate [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f"% (counter, time.time() - start_time, m, M)
                #print "Iterate [%d]: time: %4.4f" % (counter, time.time() - start_time)
            if (epoch + 1) % conf.save_per_epoch == 0:
                #save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))
                save_path = saver.save(sess, conf.save_path + "/model.ckpt")
                print "Model at epoch %s saved in file: %s" % (epoch+1, save_path)

                log = open(conf.save_path + "/log", "w")
                log.write(str(epoch + 1))
                log.close()

                test_data = data["test"]()
                for img, cond, name in test_data:
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    recov = sess.run(generated_images[name], feed_dict={model.image:pimg, model.cond:pcond})
                    test_writer.add_summary(recov, (epoch))
                    gen_img = gen_img.reshape(gen_img.shape[1:])

                    # fits_recover = conf.unstretch(gen_img[:,:,0])
                    # hdu = fits.PrimaryHDU(fits_recover)
                    # save_dir = '%s/epoch_%s/fits_output'%(out_dir, epoch+1)
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir)
                    # filename = '%s/%s-r.fits'%(save_dir,name)
                    # if os.path.exists(filename):
                    #     os.remove(filename)
                    # hdu.writeto(filename)
                    recover = gen_img[:,:,0]
                    save_dir = '%s/epoch_%s/npy_output'%(out_dir, epoch+1)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    name = name.replace('-r.npy','')
                    filename = '%s/%s-r.fits'%(save_dir,name)
                    if os.path.exists(filename):
                        os.remove(filename)
                    np.save(filename, recover)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf.use_gpu)
    train()
