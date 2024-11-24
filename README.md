# DeepSecure-AI

DeepSecure-AI is a powerful open-source tool designed to detect fake images, videos, and audios. Utilizing state-of-the-art deep learning techniques like EfficientNetV2 and MTCNN, DeepSecure-AI offers frame-by-frame video analysis, enabling high-accuracy deepfake detection. It's developed with a focus on ease of use, making it accessible for researchers, developers, and security analysts..

---

## Features

- Multimedia Detection: Detect deepfakes in images, videos, and audio files using a unified platform.
- High Accuracy: Leverages EfficientNetV2 for enhanced prediction performance and accurate results.
- Real-Time Video Analysis: Frame-by-frame analysis of videos with automatic face detection.
- User-Friendly Interface: Easy-to-use interface built with Gradio for uploading and processing media files.
- Open Source: Completely open source under the MIT license, making it available for developers to extend and improve.

---

## Demo-Data

You can test the deepfake detection capabilities of DeepSecure-AI by uploading your video files. The tool will analyze each frame of the video, detect faces, and determine the likelihood of the video being real or fake.

Examples:  
1. [Video1-fake-1-ff.mp4](#)
2. [Video6-real-1-ff.mp4](#)

---

## How It Works

DeepSecure-AI uses the following architecture:

1. Face Detection:  
   The [MTCNN](https://arxiv.org/abs/1604.02878) model detects faces in each frame of the video. If no face is detected, it will use the previous frame's face to ensure accuracy.

2. Fake vs. Real Classification:  
   Once the face is detected, it's resized and fed into the [EfficientNetV2](https://arxiv.org/abs/2104.00298) deep learning model, which determines the likelihood of the frame being real or fake.

3. Fake Confidence:  
   A final prediction is generated as a percentage score, indicating the confidence that the media is fake.

4. Results:  
   DeepSecure-AI provides an output video, highlighting the detected faces and a summary of whether the input is classified as real or fake.

---

## Project Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.10
- Gradio (pip install gradio)
- TensorFlow (pip install tensorflow)
- OpenCV (pip install opencv-python)
- PyTorch (pip install torch torchvision torchaudio)
- facenet-pytorch (pip install facenet-pytorch)
- MoviePy (pip install moviepy)

### Installation

1. Clone the repository:
        git clone https://github.com/Divith123/DeepSecure-AI.git
    cd DeepSecure-AI
    

2. Install required dependencies:
        pip install -r requirements.txt
    

3. Download the pre-trained model weights for EfficientNetV2 and place them in the project folder.

### Running the Application

1. Launch the Gradio interface:
        python app.py
    

2. The web interface will be available locally. You can upload a video, and DeepSecure-AI will analyze and display results.

---

## Example Usage

Upload a video or image to DeepSecure-AI to detect fake media. Here are some sample predictions:

- Video Analysis: The tool will detect faces from each frame and classify whether the video is fake or real.
- Result Output: A GIF or MP4 file with the sequence of detected faces and classification result will be provided.

---

## Technologies Used

- TensorFlow: For building and training deep learning models.
- EfficientNetV2: The core model for image and video classification.
- MTCNN: For face detection in images and videos.
- OpenCV: For video processing and frame manipulation.
- MoviePy: For video editing and result generation.
- Gradio: To create a user-friendly interface for interacting with the deepfake detector.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributions

Contributions are welcome! If you'd like to improve the tool, feel free to submit a pull request or raise an issue.

For more information, check the [Contribution Guidelines](CONTRIBUTING.md).

---

## References
- Li et al. (2020): [Celeb-DF(V2)](https://arxiv.org/abs/2008.06456)
- Rossler et al. (2019): [FaceForensics++](https://arxiv.org/abs/1901.08971)
- Timesler (2020): [Facial Recognition Model in PyTorch](https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch)

---

### Disclaimer

DeepSecure-AI is a research project and is designed for educational purposes.Please use responsibly and always give proper credit when utilizing the model in your work.
