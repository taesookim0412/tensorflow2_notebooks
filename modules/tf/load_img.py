import tensorflow as tf
import os
from zipfile import ZipFile
import glob

def load_img(img_fp):
    img = tf.io.read_file(img_fp)
    img = tf.image.decode_image(img, dtype=tf.uint8)
    img = tf.image.grayscale_to_rgb(img)
    img_fn = tf.strings.split(img_fp, '\\')[-1]
    return img, img_fn

def load_mnist(batch_size=1, tiny=True, filterFns=False):
    mnist_subdir = 'mnist_png'
    if tiny:
        mnist_subdir = 'mnist_tiny'
    extract_mnist(mnist_subdir)
    train_dir = os.path.join(os.path.dirname(os.getcwd()), 'sharedfiles',mnist_subdir,'training', '*', '*.png')
    if filterFns:
        skips = set([f"{str(s*2)}.png" for s in range(30000)])
        globbedFiles = [x for x in glob.glob(train_dir)
                        if x.split('\\')[-1] not in skips]
    else:
        globbedFiles = glob.glob(train_dir)
    imgs = tf.data.Dataset.from_tensor_slices(globbedFiles)
    imgs = imgs.map(load_img)
    imgs = imgs.batch(batch_size)
    return imgs

def extract_mnist(mnist_subdir):
    mnist_path = os.path.join(os.path.dirname(os.getcwd()), 'sharedfiles', mnist_subdir)
    if not os.path.isdir(mnist_path):
        mnist_path = os.path.join(os.path.dirname(os.getcwd()), 'sharedfiles', f"{mnist_subdir}.zip")
        mnist_extract_path = os.path.dirname(mnist_path)
        with ZipFile(mnist_path, 'r') as f:
            f.extractall(mnist_extract_path)
        f.close()