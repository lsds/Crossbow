import sys
import os
import math
import numpy as np

import argparse

import tensorflow as tf

from tensorflow.python.platform import gfile

from tensorflow.contrib.image.python.ops import distort_image_ops

from tensorflow.python.layers import utils


def _count(filename):
    count = 0
    for record in tf.python_io.tf_record_iterator(filename):
        count += 1
    return count


def _parse(record):
    feature_map = {
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    feature_map.update(
        {
            k: sparse_float32
            for k in [
            'image/object/bbox/xmin',
            'image/object/bbox/ymin',
            'image/object/bbox/xmax',
            'image/object/bbox/ymax'
        ]
        }
    )

    features = tf.parse_single_example(record, feature_map)

    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    image = features['image/encoded']

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    # bbox = tf.expand_dims(bbox, 0)
    # bbox = tf.transpose(bbox, [0, 2, 1])

    return image, height, width, label, xmin, ymin, xmax, ymax, features['image/class/text']


_RESIZE_METHOD_MAP = {

    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}


def _get_image_resize_method(method, position=0):
    if method != 'round_robin':
        return _RESIZE_METHOD_MAP[method]

    # return a resize method based on batch position in a round-robin fashion.
    methods = list(_RESIZE_METHOD_MAP.values())

    def lookup(index):
        return methods[index]

    def resize_method_0():
        return utils.smart_cond(position % len(methods) == 0, lambda: lookup(0), resize_method_1)

    def resize_method_1():
        return utils.smart_cond(position % len(methods) == 1, lambda: lookup(1), resize_method_2)

    def resize_method_2():
        return utils.smart_cond(position % len(methods) == 2, lambda: lookup(2), lambda: lookup(3))

    return resize_method_0()

def _normalized_image(image):
    # Rescale from [0, 255] to [0, 2]
    image = tf.multiply(image, 1. / 127.5)
    # Rescale to [-1, 1]
    image = tf.subtract(image, 1.0)
    return image

def _eval_image(image, height, width, position, method):
    with tf.name_scope('eval_image'):
        shape = tf.shape(image)

        image_height = shape[0]
        image_width = shape[1]

        image_height_float = tf.cast(image_height, tf.float32)
        image_width_float = tf.cast(image_width, tf.float32)

        # This value is chosen so that in ResNet, images are cropped to a size of
        # 256 x 256, which matches what other implementations do. The final image
        # size for ResNet is 224 x 224, and floor(224 x 1.145) = 256.
        scale_factor = 1.145

        # Compute resize_height and resize_width to be the minimum values such that:
        #
        #   1. The aspect ratio is maintained (i.e. resize_height / resize_width is
        #      image_height / image_width), and
        #
        #   2. resize_height >= height * `scale_factor`, and
        #
        #   3. resize_width >= width * `scale_factor`
        #
        max_ratio = tf.maximum(height / image_height_float, width / image_width_float)
        resize_height = tf.cast(image_height_float * max_ratio * scale_factor, tf.int32)
        resize_width = tf.cast(image_width_float * max_ratio * scale_factor, tf.int32)

        # Resize the image to shape (`resize_height`, `resize_width`)

        image_resize_method = _get_image_resize_method(method, position)

        distorted_image = tf.image.resize_images(image,
                                                 [resize_height, resize_width],
                                                 image_resize_method,
                                                 align_corners=False)

        # Do a central crop of the image to size (height, width).

        total_crop_height = (resize_height - height)
        crop_top = total_crop_height // 2

        total_crop_width = (resize_width - width)
        crop_left = total_crop_width // 2

        distorted_image = tf.slice(distorted_image, [crop_top, crop_left, 0], [height, width, 3])

        distorted_image.set_shape([height, width, 3])
       	
        image = distorted_image
        
        # Normalise image
        image = _normalized_image(image)

    return image


def _train_image(image_buffer, height, width, bbox, position, method):
    """
    Distort one image for training a network.
    """

    with tf.name_scope('distort_image'):

        # A large fraction of image data sets contain a human-annotated bounding box

        min_object_covered = 0.1
        aspect_ratio_range = [0.75, 1.33]
        area_range = [0.05, 1.0]
        max_attempts = 100

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)

        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = tf.image.decode_jpeg(image_buffer, channels=3, dct_method='INTEGER_FAST')
        image = tf.slice(image, bbox_begin, bbox_size)

        distorted_image = tf.image.random_flip_left_right(image)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.

        image_resize_method = _get_image_resize_method(method, position)
        distorted_image = tf.image.resize_images(distorted_image,
                                                 [height, width],
                                                 image_resize_method,
                                                 align_corners=False)

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        
        # Normalise image
        distorted_image = _normalized_image(distorted_image)

        return bbox_begin, bbox_size, distorted_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', choices=['train', 'validation'], default='train')
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()
    with tf.Session() as session:
        # subset = "train"
        subset = args.subset
        input_dir = args.input_dir
        output_dir = args.output_dir
        maxrecordsperfile = 2048
        N = 0  # Expect 1251 records in 1 file (for training)
        # Number of records per file...
        mx = 0
        # directory = "/data/tf/imagenet/records"
        # directory = "."
        pattern = os.path.join(input_dir, '%s-*-of-*' % subset)
        files = gfile.Glob(pattern)
        if not files:
            raise ValueError()
        files = sorted(files)
        print("Counting records...")
        index = 0
        for filename in files:
            recordsinfile = _count(filename)
            if (mx < recordsinfile):
                mx = recordsinfile
            N += recordsinfile
            index += 1
            if (index % 10 == 0):
                print("%4d files counted" % (index))
        print("%d records in %d file%s (max. %d records/file)" % (N, len(files), "s" if len(files) > 1 else "", mx))

        queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)
        reader = tf.TFRecordReader()
        _, record = reader.read(queue)

        imagebuffer, height, width, label, xmin, ymin, xmax, ymax, _ = _parse(record)

        # Decode image
        image = tf.image.decode_jpeg(imagebuffer, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')

        eval_image = _eval_image(image, 224, 224, 0, 'bilinear')

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])
        bbox_begin, bbox_size, train_image = _train_image(imagebuffer, 224, 224, bbox, 0, 'bilinear')

        # Let's run the evaluation image transformations first

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(10):
            buf, img, h, w, lbl, xmn, ymn, xmx, ymx, eval_img, bbx, bbx_bgn, bbx_sz, train_img = session.run(
                [imagebuffer, image, height, width, label, xmin, ymin, xmax, ymax, eval_image, bbox, bbox_begin, bbox_size, train_image])

            # Change label in the range [0,999]
            lbl[0] -= 1

            csum = np.sum(img)
            print("=== [Record dump] ===")
            print("Image shape %s (height %d width %d), checksum %d and label %d" % (img.shape, h, w, csum, lbl))
            print("xmin", xmn)
            print("ymin", ymn)
            print("xmax", xmx)
            print("ymax", ymx)
            print("bbox", bbx)
            print("bbox_begin", bbx_bgn)
            print("bbox_size", bbx_sz)
            print("Evalaution image of shape %s and checksum %d" % (eval_img.shape, np.sum(eval_img)))
            print("Training image of shape %s and checksum %d" % (train_img.shape, np.sum(train_img)))

        coord.request_stop()
        coord.join(threads)
        session.close()
    print("Bye.")
