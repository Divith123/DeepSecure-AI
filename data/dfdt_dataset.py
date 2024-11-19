'''Module for loading the fakeavceleb dataset from tfrecord format'''
import numpy as np
import tensorflow as tf
from data.augmentation_utils import create_frame_transforms, create_spec_transforms

FEATURE_DESCRIPTION = {
    'video_path': tf.io.FixedLenFeature([], tf.string), 
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'clip/label/index': tf.io.FixedLenFeature([], tf.int64),
    'clip/label/text': tf.io.FixedLenFeature([], tf.string), 
    'WAVEFORM/feature/floats': tf.io.FixedLenFeature([], tf.string)
}

@tf.function
def _parse_function(example_proto):

    #Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    
    video_path = example['video_path']
    video = tf.io.decode_raw(example['image/encoded'], tf.int8)    
    spectrogram = tf.io.decode_raw(example['WAVEFORM/feature/floats'], tf.float32)
    
    label = example["clip/label/text"]
    label_map = example["clip/label/index"]
    
    return video, spectrogram, label_map

@tf.function
def decode_inputs(video, spectrogram, label_map):
    '''Decode tensors to arrays with desired shape'''
    frame = tf.reshape(video, [10, 3, 256, 256])
    frame = frame[0] / 255 #Pick the first frame and normalize it.
    # frame = tf.cast(frame, tf.float32)

    label_map = tf.expand_dims(label_map, axis = 0)
    
    sample = {'video_reshaped': frame, 'spectrogram': spectrogram, 'label_map': label_map}
    return sample


def decode_train_inputs(video, spectrogram, label_map):
    #Data augmentation for spectograms
    spectrogram_shape = spectrogram.shape
    spec_augmented = tf.py_function(aug_spec_fn, [spectrogram], tf.float32)
    spec_augmented.set_shape(spectrogram_shape)

    frame = tf.reshape(video, [10, 256, 256, 3])
    frame = frame[0] #Pick the first frame.
    frame = frame / 255 #Normalize tensor.

    frame_augmented = tf.py_function(aug_img_fn, [frame], tf.uint8)
    # frame_augmented.set_shape(frame_shape)

    frame_augmented.set_shape([3, 256, 256])
    label_map = tf.expand_dims(label_map, axis = 0)

    augmented_sample = {'video_reshaped': frame_augmented, 'spectrogram': spec_augmented, 'label_map': label_map}
    return augmented_sample


def aug_img_fn(frame):
  frame = frame.numpy().astype(np.uint8)
  frame_data = {'image': frame}
  aug_frame_data = create_frame_transforms(**frame_data)
  aug_img = aug_frame_data['image']
  aug_img = aug_img.transpose(2, 0, 1)
  return aug_img

def aug_spec_fn(spec):
  spec = spec.numpy()
  spec_data = {'spec': spec}
  aug_spec_data = create_spec_transforms(**spec_data)
  aug_spec = aug_spec_data['spec']
  return aug_spec


class FakeAVCelebDatasetTrain:

    def __init__(self, args):
        self.args = args
        self.samples = self.load_features_from_tfrec()

    def load_features_from_tfrec(self):
        '''Loads raw features from a tfrecord file and returns them as raw inputs'''
        ds = tf.io.matching_files(self.args.data_dir)
        files = tf.random.shuffle(ds)

        shards = tf.data.Dataset.from_tensor_slices(files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.map(_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.map(decode_train_inputs, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size = self.args.batch_size)
        return dataset
    
    
    def __len__(self):
        self.samples = self.load_features_from_tfrec(self.args.data_dir)
        cnt = self.samples.reduce(np.int64(0), lambda x, _: x + 1)
        cnt = cnt.numpy()
        return cnt

class FakeAVCelebDatasetVal:

    def __init__(self, args):
        self.args = args
        self.samples = self.load_features_from_tfrec()

    def load_features_from_tfrec(self):
        '''Loads raw features from a tfrecord file and returns them as raw inputs'''
        ds = tf.io.matching_files(self.args.data_dir)
        files = tf.random.shuffle(ds)

        shards = tf.data.Dataset.from_tensor_slices(files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.map(_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.map(decode_inputs, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size = self.args.batch_size)
        return dataset
    
    
    def __len__(self):
        self.samples = self.load_features_from_tfrec(self.args.data_dir)
        cnt = self.samples.reduce(np.int64(0), lambda x, _: x + 1)
        cnt = cnt.numpy()
        return cnt