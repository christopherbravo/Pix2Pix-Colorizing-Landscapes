import tensorflow as tf
import numpy as np
import pathlib

def load_data(image_path):
    ## Load dataset from Keras library.
    dataset_name = "facades"
    _URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

    path_to_zip = tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz",
        origin=_URL,
        extract=True)

    path_to_zip  = pathlib.Path(path_to_zip)
    PATH = path_to_zip.parent/dataset_name\

    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
    split = int(tf.shape(image)[1]/2)

    input_image = tf.cast(image[:, :split, :], tf.float32)
    target_image = tf.cast(image[:, split:, :], tf.float32)

    return input_image, target_image

def get_path(image_path):
    dataset_name = "facades"
    _URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

    path_to_zip = tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz",
        origin=_URL,
        extract=True)

    path_to_zip  = pathlib.Path(path_to_zip)
    PATH = path_to_zip.parent/dataset_name\

    return PATH

def resize(images, height, width):
    '''images: tuple containing input image and target image.
       height: resize heights
       width: resize width'''

    resized_input = tf.image.resize(images[0], [height, width])
    resized_target = tf.image.resize(images[1], [height, width])

    return resized_input, resized_target

def random_crop(images, height, width):
    '''images: tuple containing input image and target image.'''

    cropped = tf.image.random_crop(tf.stack([images[0],images[1]], axis=0), size=[2, height, width, 3])

    return cropped[0], cropped[1]

def normalize_images(images):
    input_image = images[0]/127.5 - 1
    target_image = images[1]/127.5 - 1

    return input_image, target_image

@tf.function()
def random_jitter(images):
    resized_input, resized_target = resize(images, 286, 286)
    input_image, target_image = random_crop((resized_input, resized_target), 256, 256)
    random = np.random.uniform()
    if random>0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image

def load_process_train_data(image_path):
    input, target = load_data(image_path)
    input, target = random_jitter((input, target))
    input, target = normalize_images((input, target))

    return input, target

def load_process_test_data(image_path):
    input, target = load_data(image_path)
    input, target = resize((input, target), 256, 256)
    input, target = normalize_images((input, target))

    return input, target

def get_data(image_path):
    PATH = get_path(image_path)

    train = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
    train = train.map(load_process_train_data,
                                  num_parallel_calls=tf.data.AUTOTUNE)
    train = train.shuffle(400)
    train = train.batch(1)

    try:
        test = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
    except tf.errors.InvalidArgumentError:
        test = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
    test = test.map(load_process_test_data)
    test = test.batch(1)

    return train, test
