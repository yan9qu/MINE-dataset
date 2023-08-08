from moviepy.editor import VideoFileClip
video_list = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/video_list.txt'
audio_list = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/audio_list.txt'


# read lines from video_list.txt
with open(video_list, 'r') as f:
    lines = f.readlines()


# for every video, generate .wav file if is .mp4
# writing a .wav file_list to audio_list.txt
audio_path = []

for line in lines:
    if not line.strip().split('/')[-1].split('.')[0].startswith('vid'):
        continue
    video_path = line.strip()
    print(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    print(video_name)
    video_segments = VideoFileClip(video_path)
    audio = video_segments.audio
    audio.write_audiofile(video_path[0:-4] + ".wav")
    audio_path.append(video_path[0:-4] + ".wav")

# write audio_path to a txt file
with open(audio_list, 'w') as f:
    for item in audio_path:
        f.write("%s\n" % item)
