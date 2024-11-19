import gradio as gr
import inference_2 as inference


title="Multimodal deepfake detector"
description="Deepfake detection for videos, images and audio modalities."
            
           
video_interface = gr.Interface(inference.deepfakes_video_predict,
                    gr.Video(),
                    "text",
                    examples = ["videos/aaa.mp4", "videos/bbb.mp4"],
                    cache_examples = False
                    )


image_interface = gr.Interface(inference.deepfakes_image_predict,
                    gr.Image(),
                    "text",
                    examples = ["images/lady.jpg", "images/fake_image.jpg"],
                    cache_examples=False
                    )

audio_interface = gr.Interface(inference.deepfakes_spec_predict,
                               gr.Audio(),
                               "text",
                               examples = ["audios/DF_E_2000027.flac", "audios/DF_E_2000031.flac"],
                               cache_examples = False)


app = gr.TabbedInterface(interface_list= [video_interface, audio_interface,image_interface], 
                         tab_names = ['Video inference', 'Audio inference', 'Image inference'])

if __name__ == '__main__':
    app.launch(share = False)