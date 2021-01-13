from multiprocessing import Process, Manager
import sys
import os
import tensorflow as tf
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import time
from IPython import get_ipython

def inferMnist(img_fns, dontRead=False):
    setGrowth()
    model_dir = os.path.join(upDir(os.getcwd()), 'sharedmodels', 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8', 'saved_model')
    inception = tf.saved_model.load(model_dir)
    imgs = load_mnist()
    for i, (img, img_fn) in imgs.enumerate():
        img_fn_str = img_fn[0].numpy().decode('ascii')
        if img_fn_str in img_fns and dontRead is False:
            img_fns['skips'] += 1
            continue
        img_fns[img_fn_str] = 1
        inception.signatures['serving_default'](img)

def main(dontRead=False):
    manager = Manager()
    img_fns = manager.dict()
    img_fns['skips'] = 0
    p1 = Process(target=inferMnist, args=(img_fns,dontRead))
    p2 = Process(target=inferMnist, args=(img_fns,dontRead))
    start_time = time.time()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(img_fns['skips'])
    print(f'time elapsed: {time.time() - start_time}')

def load_img(img_fp):
    img = tf.io.read_file(img_fp)
    img = tf.image.decode_image(img, dtype=tf.uint8)
    img = tf.image.grayscale_to_rgb(img)
    img_fn = tf.strings.split(img_fp, '\\')[-1]
    return img, img_fn

def load_mnist():
    extract_mnist()
    train_dir = os.path.join(upDir(os.getcwd()), 'sharedfiles','mnist_tiny','training', '*.png')
    imgs = tf.data.Dataset.list_files(train_dir)
    imgs = imgs.map(load_img)
    imgs = imgs.batch(1)
    return imgs
#%%

def setGrowth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
setGrowth()

#%%

def upDir(dir):
    return os.path.dirname(dir)

#%%

def extract_mnist():
    mnist_path = os.path.join(upDir(os.getcwd()), 'sharedfiles', 'mnist_tiny')
    if not os.path.isdir(mnist_path):
        mnist_path = os.path.join(upDir(os.getcwd()), 'sharedfiles', 'mnist_tiny.zip')
        mnist_extract_path = os.path.join(os.path.dirname(mnist_path))
        with ZipFile(mnist_path, 'r') as f:
            f.extractall(mnist_extract_path)
        f.close()



# if __name__ == '__main__':

