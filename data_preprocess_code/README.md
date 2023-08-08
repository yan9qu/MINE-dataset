The data preprocess code for MINE:Multimodal IntentioN and Emotion Understanding in the Wild.
---
Including: 
Data splitation;
Feature extraction(four modalities);
Json generation.

---
Feature extraction files:

Image : q2l_infer.py

Audio : video_to_audio.py, infer_audio.py

Text  : infer_text.py

Video : 003_video_frame.py
(video to frames), infer_video.py (require [repo](https://github.com/haofanwang/video-swin-transformer-pytorch))
