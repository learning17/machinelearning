#-*- coding:utf-8 -*-
from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import six
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/data/ImageNet/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/data/ImageNet/validation',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/data/ImageNet/tfrecord',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file',
                           '/data/ImageNet/imagenet_lsvrc_2015_synsets.txt',
                           'Labels file')
tf.app.flags.DEFINE_string('imagenet_metadata_file',
                           '/data/ImageNet/imagenet_metadata.txt',
                           'ImageNet metadata file')
tf.app.flags.DEFINE_string('bounding_box_file',
                           '/data/ImageNet/imagenet_2012_bounding_boxes.csv',
                           'Bounding box file')
FLAGS = tf.app.flags.FLAGS

'''
保存tfrecords流程：提取features -> 保存为Example结构对象 -> TFRecordWriter写入文件
'''

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto"""
    if isinstance(value, six.string_types):
        value = six.binary_type(value,encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _convert_to_example(filename, image_buffer, label, synset, human, bbox, height, width):
    """
    height: integer, image height
    width: integer, image width
    label: integer, 1-1000
    synset: string, unique WordNet ID specifying the label e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers,
            specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
            the same label as the image label.
     """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    xmin = list()
    ymin = list()
    xmax = list()
    ymax = list()
    for b in bbox:
        assert len(b) == 4
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature([label] * len(xmin)),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example

class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()

        self._png_data = tf.placeholder(dtype=tf.string)
        png_image = tf.image.decode_png(self._png_data,channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(png_image,format='rgb',quality=100)

        self._cmyk_data = tf.placeholder(dtype=tf.string)
        cmyk_image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(cmyk_image, format='rgb', quality=100)

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self,image_data):
        return self._sess.run(self._png_to_jpeg,feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self,image_data):
        image = self._sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _is_png(filename):
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
    return filename.endswith('.png') or 'n02105855_2933.JPEG' in filename

def _is_cmyk(filename):
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                 'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                 'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                 'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                 'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                 'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                 'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                 'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                 'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                 'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                 'n07583066_647.JPEG', 'n13037406_4650.JPEG']
    return filename.split('/')[-1] in blacklist

def _process_image(filename,coder):
    with tf.gfile.FastGFile(filename,'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        print('Converting CMYK to RGB for %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)
    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    height = image.shape[0]
    width = image.shape[1]

    return image_data,height,width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, bboxes, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    # 全部数据分 num_threads 个线程处理，生成 num_shards 份 TFRecord
    # 数据，每个线程生成 int(num_shards/num_threads) 个TFRecord
    num_shards_per_batch = int(num_shards / num_threads)
    # 将该线程处理的数据总量分配给每个TFRecord
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0] # 该线程处理的数据量
    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards) # train-00002-of-00010
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]
            bbox = bboxes[i]
            image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(filename, image_buffer, label,
                                          synset, human, bbox,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if counter % 1000 == 0:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))

def _process_image_files(name, num_shards, filenames, synsets, labels, humans, bboxes):
    assert len(filenames) == len(synsets)
    assert len(filenames) == len(labels)
    assert len(filenames) == len(humans)
    assert len(filenames) == len(bboxes)
    # Break all images into batches
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = list()
    for i in range(len(spacing) -1 ):
        ranges.append([spacing[i], spacing[i + 1]])
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))

    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = list()
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                synsets, labels, humans, bboxes, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    # Wait for all the threads to terminate
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))


def _build_synset_lookup(imagenet_metadata_file):
    """
        input: n02119247    black fox
        output: {'n02119247': 'black fox'}
    """
    synset_to_human = dict()
    with tf.gfile.FastGFile(imagenet_metadata_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            assert len(parts) == 2, ('Failed to parse: %s' % line)
            synset_to_human[parts[0]] = parts[1]
    return synset_to_human

def _build_bounding_box_lookup(bounding_box_file):
    """
        input: n00007846_64193.JPEG,0.1,0.2,0.3,0.4
        output: {'n00007846_64193.JPEG': [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}
    """
    images_to_bboxes = dict()
    with tf.gfile.FastGFile(bounding_box_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            assert len(parts) == 5, ('Failed to parse: %s' % line)
            filename = parts[0]
            box = [float(i) for i in parts[1:]] #xmin,ymin,xmax,ymax
            if filename not in images_to_bboxes:
                images_to_bboxes[filename] = list()
            images_to_bboxes[filename].append(box)
    return images_to_bboxes

def _find_image_files(data_dir, labels_file):
    """
        input: labels_file(imagenet_lsvrc_2015_synsets.txt, n01440764)
                data_dir('/data/ImageNet')
        output: filenames('/data/ImageNet/n01440764/ILSVRC2012_val_00000293.JPEG')
                synsets('n01440764')
                labels(7)
    """
    challenge_synsets = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    labels = list()
    filenames = list()
    synsets = list()
    label_index = 1
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir,synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)
        if label_index % 100 == 0:
            print('Finished finding files in %d of %d classes'
                  % (label_index, len(challenge_synsets)))
        label_index += 1
        # 打乱labels,filenames,synsets
        shuffled_index = list(range(len(filenames)))
        random.seed(12345)
        random.shuffle(shuffled_index)
        filenames = [filenames[i] for i in shuffled_index]
        synsets = [synsets[i] for i in shuffled_index]
        labels = [labels[i] for i in shuffled_index]
    print('Found %d JPEG files across %d labels inside %s' %
          (len(filenames), len(challenge_synsets), data_dir))
    return filenames,synsets,labels

def _find_human_readable_labels(synsets, synset_to_human):
    """
        input: synsets: ['n01440764']
                synset_to_human: {'n02119247': 'black fox'}
        output: humans ['black fox']
    """
    humans = list()
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    return humans

def _find_image_bounding_boxes(filenames, image_to_bboxes):
    """
        input: image_to_bboxes: {'n00007846_64193.JPEG': [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}
                filenames: '/data/ImageNet/n01440764/ILSVRC2012_val_00000293.JPEG'
        output: bboxes [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]
    """
    bboxes = list()
    for f in filenames:
        basename = os.path.basename(f)
        if basename in image_to_bboxes:
            bboxes.append(image_to_bboxes[basename])
        else:
            bboxes.append([])
    return bboxes


def _process_dataset(name, data_dir, num_shards, synset_to_human,image_to_bboxes):

    filenames, synsets, labels = _find_image_files(data_dir, FLAGS.labels_file)
    humans = _find_human_readable_labels(synsets, synset_to_human)
    bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)
    _process_image_files(name,num_shards,filenames, synsets, labels, humans, bboxes)


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)
    synset_to_human = _build_synset_lookup(FLAGS.imagenet_metadata_file)
    image_to_bboxes = _build_bounding_box_lookup(FLAGS.bounding_box_file)

    _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards,
                synset_to_human,image_to_bboxes)
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                synset_to_human,image_to_bboxes)

if __name__ == '__main__':
    tf.app.run()
