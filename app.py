import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons

from facenet_pytorch import MTCNN
from PIL import Image
import moviepy.editor as mp
import os
import zipfile

local_zip = "FINAL-EFFICIENTNETV2-B0.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('FINAL-EFFICIENTNETV2-B0')
zip_ref.close()

# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device='cpu')

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class."""

        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces."""

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:

                    boxes, probs = self.detector.detect(frames)

                    for i in range(len(frames)):

                        if boxes[i] is None:
                            faces.append(face2)     #append previous face frame if no face is detected
                            continue

                        box = boxes[i][0].astype(int)
                        frame = frames[i]
                        face = frame[box[1]:box[3], box[0]:box[2]]

                        if not face.any():
                            faces.append(face2)     #append previous face frame if no face is detected
                            continue

                        face2 = cv2.resize(face, (224, 224))

                        faces.append(face2)

                    frames = []

        v_cap.release()

        return faces


detection_pipeline = DetectionPipeline(detector=mtcnn, n_frames=20, batch_size=60)

model = tf.keras.models.load_model("FINAL-EFFICIENTNETV2-B0")


def deepfakespredict(input_video):

    faces = detection_pipeline(input_video)

    total = 0
    real = 0
    fake = 0

    for face in faces:

        face2 = face / 255
        pred = model.predict(np.expand_dims(face2, axis=0))[0]
        total += 1

        pred2 = pred[1]

        if pred2 > 0.5:
            fake += 1
        else:
            real += 1

    fake_ratio = fake / total

    text = ""
    text2 = "DeepSecureAI Confidence: " + str(fake_ratio * 100) + "%"

    if fake_ratio >= 0.5:
        text = "The video is FAKE."
    else:
        text = "The video is REAL."

    face_frames = []

    for face in faces:
        face_frame = Image.fromarray(face.astype('uint8'), 'RGB')
        face_frames.append(face_frame)

    face_frames[0].save('results.gif', save_all=True, append_images=face_frames[1:], duration=250, loop=100)
    clip = mp.VideoFileClip("results.gif")
    clip.write_videofile("video.mp4")

    return text, text2, "video.mp4"


title = "DeepSecure-AI ( Video )"
description = ""

examples = [
    ['real-1'],
    ['real-1'],
    ['real-1'],
    ['real-1'],
    ['real-1'],
    ['fake-1'],
]

gr.Interface(
    fn=deepfakespredict,
    inputs=gr.Video(),
    outputs=[gr.Text(), gr.Text(), gr.Video(label="Detected face sequence")],
    title=title,
    description=description,
    examples=examples
).launch()
