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
        'image/height':      tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/width':       tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/encoded':     tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/class/text':  tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
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
    width  = tf.cast(features['image/width'],  dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    # bbox = tf.expand_dims(bbox, 0)
    # bbox = tf.transpose(bbox, [0, 2, 1])

    return image, height, width, label, xmin, ymin, xmax, ymax, features['image/class/text']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', choices=['train', 'validation'], default='train')
    parser.add_argument('--input-dir', type=string, default='.')
    parser.add_argument('--output-dir', type=string, default='.')
    args = parser.parse_args()
    with tf.Session() as session:
        # subset = "train"
        subset = args.subset
        input_dir = args.input_dir
        output_dir = args.output_dir
        maxrecordsperfile = 2048
        N = 0 # Expect 1251 records in 1 file (for training)
		# Number of records per file...
        mx = 0
        # directory = "/data/tf/imagenet/records"
        # directory = "."
        pattern = os.path.join(input_dir, '%s-*-of-*' % subset)
        files = gfile.Glob(pattern)
        if not files:
            raise ValueError()
        files = sorted(files)
        print "Counting records..."
        index = 0
        for filename in files:
            recordsinfile = _count (filename)
            if (mx < recordsinfile):
                mx = recordsinfile
            N += recordsinfile
            index += 1
            if (index % 10 == 0):
                print "%4d files counted" % (index)
        print "%d records in %d file%s (max. %d records/file)" % (N, len(files), "s" if len(files) > 1 else "", mx)
        
        queue = tf.train.string_input_producer(files, num_epochs=1)
        reader = tf.TFRecordReader()
        _, record = reader.read(queue)
        
        imagebuffer, height, width, label, xmin, ymin, xmax, ymax, _ = _parse(record)
        image = tf.image.decode_jpeg(imagebuffer, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        # print image, height, width
        
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        lbl_checksums = []
        img_checksums = []
	    
        filecounter = 1
        filename = "%s/crossbow-%s.records.%d" % (output_dir, subset, filecounter)
        f = open(filename, "wb")
        # Write number of records as a file header
        recordsinfile = 0
        if (N < maxrecordsperfile):
            recordsinfile = N
        else:
            recordsinfile = maxrecordsperfile
        header = np.array([recordsinfile], np.int32)
        f.write(header)
        
        print "Writing records..."
        
        totalrecordswritten = 0
        recordswritten = 0
        # Iterate over dataset records
        for index in range(N):
            
            buf, img, h, w, lbl, xmn, ymn, xmx, ymx = session.run(
               [imagebuffer, image, height, width, label, xmin, ymin, xmax, ymax])
            
            # Change label in the range [0,999]
            lbl[0] -= 1
            
            csum = np.sum(img)
            
            lbl_checksums.append(lbl[0])
            img_checksums.append(csum)
            
			# How many bounding boxes?
            numberofboxes  = len(xmn[0]) # All arrays have the same length
            numberofboxesd = np.array([numberofboxes], np.int32)
            
            #
            # Binary file contains records of the following format:
            # [length (n)]
            # [label]
            # [number of bounding boxes (k)]
            # [xmin (k times)]
            # [ymin (k times)]
            # [xmax (k times)]
            # [ymax (k times)]
            # [height]
            # [width]
            # [pixels]
            # .
            # So, n = 4 + 4 + 4 + (k x 16) + 4 + 4 + len(buffer).
            
            length  = 4 + 4 + 4 + (numberofboxes * 16) + 4 + 4 +len(buf)
            lengthd = np.array([length], np.int32)
            
            if (index < 10):
                print "Writing record #%07d of length %6d; shape %s" % (index, length, img.shape)
            
            f.write(lengthd)
            f.write(lbl)
            f.write(numberofboxesd)
            if numberofboxes > 0:
                f.write(xmn)
                f.write(ymn)
                f.write(xmx)
                f.write(ymx)
            f.write(h)
            f.write(w)
            f.write(buf)
            
            recordswritten += 1
            if recordswritten == maxrecordsperfile:
                # Rotate files
                f.close()
                # Reset counter
                totalrecordswritten += recordswritten
                recordswritten = 0
                # Should we open another file?
                remaining = N - totalrecordswritten
                if remaining > 0:
                    filecounter += 1
                    filename = "%s/crossbow-%s.records.%d" % (output_dir, subset, filecounter)
                    f = open(filename, "wb")
                    # Write file header
                    recordsinfile = 0
                    if (remaining < maxrecordsperfile):
                        recordsinfile = remaining
                    else:
                        recordsinfile = maxrecordsperfile
                    header = np.array([recordsinfile], np.int32)
                    f.write(header)
        # Finish up
        if (recordswritten > 0):
            # Close last file & increment counter
            f.close()
            totalrecordswritten += recordswritten
            
        print "%d/%d records written in %d files" % (totalrecordswritten, N, filecounter)
        
        print "[DBG] %d checksums" % len(img_checksums)
        print "[DBG] Writing to file..."
        cf = open("%s-data-checksums.txt" % (subset), "w")
        z = zip(lbl_checksums, img_checksums)
        _ndx = 0
        for x,y in z:
            cf.write("%7d: %+7d %+20.5f\n" % (_ndx, x, y))
            _ndx += 1
        cf.close()
        
        coord.request_stop()
        coord.join(threads)
        session.close()
    print "Bye."
