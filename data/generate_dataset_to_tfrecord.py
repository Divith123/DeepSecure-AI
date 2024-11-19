#Code outsourced from https://github.com/deepmind/dmvr/tree/master and later modified.

"""Python script to generate TFRecords of SequenceExample from raw videos."""

import contextlib
import math
import os
import cv2
from typing import Dict, Optional, Sequence
import moviepy.editor
from absl import app
from absl import flags
import ffmpeg
import numpy as np
import pandas as pd
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

flags.DEFINE_string("csv_path", "fakeavceleb_1k.csv", "Input csv")
flags.DEFINE_string("output_path", "fakeavceleb_tfrec", "Tfrecords output path.")
flags.DEFINE_string("video_root_path", "./",
                    "Root directory containing the raw videos.")
flags.DEFINE_integer(
    "num_shards", 4, "Number of shards to output, -1 means"
    "it will automatically adapt to the sqrt(num_examples).")
flags.DEFINE_bool("decode_audio", False, "Whether or not to decode the audio")
flags.DEFINE_bool("shuffle_csv", False, "Whether or not to shuffle the csv.")
FLAGS = flags.FLAGS


_JPEG_HEADER = b"\xff\xd8"


@contextlib.contextmanager
def _close_on_exit(writers):
  """Call close on all writers on exit."""
  try:
    yield writers
  finally:
    for writer in writers:
      writer.close()


def add_float_list(key: str, values: Sequence[float],
                   sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add(
  ).float_list.value[:] = values


def add_bytes_list(key: str, values: Sequence[bytes],
                   sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add().bytes_list.value[:] = values


def add_int_list(key: str, values: Sequence[int],
                 sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add().int64_list.value[:] = values


def set_context_int_list(key: str, value: Sequence[int],
                         sequence: tf.train.SequenceExample):
  sequence.context.feature[key].int64_list.value[:] = value


def set_context_bytes(key: str, value: bytes,
                      sequence: tf.train.SequenceExample):
  sequence.context.feature[key].bytes_list.value[:] = (value,)

def set_context_bytes_list(key: str, value: Sequence[bytes],
                           sequence: tf.train.SequenceExample):
   sequence.context.feature[key].bytes_list.value[:] = value


def set_context_float(key: str, value: float,
                      sequence: tf.train.SequenceExample):
  sequence.context.feature[key].float_list.value[:] = (value,)


def set_context_int(key: str, value: int, sequence: tf.train.SequenceExample):
  sequence.context.feature[key].int64_list.value[:] = (value,)


def extract_frames(video_path, fps = 10, min_resize = 256):
    '''Load n number of frames from a video'''
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, fps).astype(int)

    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            success, frame = v_cap.retrieve()
            if not success:
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (min_resize, min_resize))
            frames.append(frame)
    
    v_cap.release()
    frame_np = np.stack(frames)
    return frame_np.tobytes()

def extract_audio(video_path: str,
                  sampling_rate: int = 16_000):
  """Extract raw mono audio float list from video_path with ffmpeg."""
  video = moviepy.editor.VideoFileClip(video_path)
  audio = video.audio.to_soundarray()
  #Load first channel.
  audio = audio[:, 0]
  
  return np.array(audio)

#Each of the features can be coerced into a tf.train.Example-compatible type using one of the _bytes_feature, _float_feature and the _int64_feature.
#You can then create a tf.train.Example message from these encoded features.

def serialize_example(video_path: str, label_name: str, label_map: Optional[Dict[str, int]] = None):
    # Initiate the sequence example.
    seq_example = tf.train.SequenceExample()

    imgs_encoded = extract_frames(video_path, fps = 10)

    audio = extract_audio(video_path)

    set_context_bytes(f'image/encoded', imgs_encoded, seq_example)
    set_context_bytes("video_path", video_path.encode(), seq_example)
    set_context_bytes("WAVEFORM/feature/floats", audio.tobytes(), seq_example)
    set_context_int("clip/label/index", label_map[label_name], seq_example)
    set_context_bytes("clip/label/text", label_name.encode(), seq_example)
    return seq_example


def main(argv):
    del argv
    # reads the input csv.
    input_csv = pd.read_csv(FLAGS.csv_path)
    if FLAGS.num_shards == -1:
        num_shards = int(math.sqrt(len(input_csv)))
    else:
        num_shards = FLAGS.num_shards
    # Set up the TFRecordWriters.
    basename = os.path.splitext(os.path.basename(FLAGS.csv_path))[0]
    shard_names = [
      os.path.join(FLAGS.output_path, f"{basename}-{i:05d}-of-{num_shards:05d}")
      for i in range(num_shards)
    ]
    writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]

    if "label" in input_csv:
        unique_labels = list(set(input_csv["label"].values))
        l_map = {unique_labels[i]: i for i in range(len(unique_labels))}
    else:
        l_map = None

    if FLAGS.shuffle_csv:
        input_csv = input_csv.sample(frac=1)
    with _close_on_exit(writers) as writers:
        row_count = 0
        for row in input_csv.itertuples():
           index = row[0]
           v = row[1]
           if os.name == 'posix':
            v = v.str.replace('\\', '/')
           l = row[2]
           row_count += 1
           print("Processing example %d of %d   (%d%%) \r" %(row_count, len(input_csv), row_count * 100 / len(input_csv)), end="")
           seq_ex = serialize_example(video_path = v, label_name = l,label_map = l_map)
           writers[index % len(writers)].write(seq_ex.SerializeToString())

if __name__ == "__main__":
  app.run(main)
