# extraction of video frames with a given ratio

import os
from moviepy.editor import VideoFileClip
# from path get the video list, folder_path is the path of the folder containing the videos
# reading folder_path from the command line
folder_path = '/data1/yq/004_intention/000_m3_dataset/data_points'

video_path_list = []


def get_video_list_from_folder(folder_path):
    # list all folders in the folder_path
    folder_list = os.listdir(folder_path)

    video_path_list = []

    for i in folder_list:
        # check if there is a mp4 file in the folder
        for j in os.listdir(folder_path + '/' + i):
            if j.endswith('.mp4'):
                video_path_list.append(folder_path+'/' + i + '/' + j)

    return video_path_list


def get_audio_list_from_folder(folder_path):
    # list all folders in the folder_path
    folder_list = os.listdir(folder_path)

    audio_path_list = []

    for i in folder_list:
        # check if there is a mp4 file in the folder
        for j in os.listdir(folder_path + '/' + i):
            if j.endswith('.mp4') and j.startswith('vid'):
                audio_path_list.append(folder_path+'/' + i + '/' + j)

    return audio_path_list


def extract_frames(video_path, dest_path, ratio):
    if not os.path.exists(dest_path):
        os.system('mkdir -p ' + dest_path)

    # get the video name
    video_name = video_path.split('/')[-2]
    # create a folder for the video
    os.system('mkdir -p ' + dest_path+'/' + video_name)
    # extract frames
    os.system('ffmpeg -i ' + video_path + ' -r ' + str(ratio) + ' ' +
              dest_path + '/' + video_name + '/' + video_name + '_%05d.jpg')


video_path_list = get_video_list_from_folder(folder_path)
# save the video list
with open('video_list.txt', 'w') as f:
    for i in video_path_list:
        f.write(i + '\n')


audio_path_list = get_audio_list_from_folder(folder_path)
# save the audio list
with open('audio_list.txt', 'w') as f:
    for i in audio_path_list:
        f.write(i + '\n')


def extract_frames_from_video(video_path_list, dest_path, ratio=1):
    # extract with one frame of every 1 second
    for i in video_path_list:
        extract_frames(i, dest_path, 1)
        # every 100 video, print the number of videos extracted
        if video_path_list.index(i) % 100 == 0:
            print('extracted ' + str(video_path_list.index(i)) + ' videos')


def extract_audio(audio_path_list):
    generated_audio_path = []
    no_audio_id = []
    # read lines from video_list.txt
    for line in audio_path_list:
        audio_path = line.split('.')[0]+'.wav'
        generated_audio_path.append(audio_path)
        video_segments = VideoFileClip(line)
        audio = video_segments.audio
        if audio is None:
            print('no audio in ' + line)
            no_audio_id.append(line)
            continue
            # save id of video with no audio
        audio.write_audiofile(audio_path)
        generated_audio_path.append(audio_path)
        # every 1000 video, print the number of videos extracted
        if video_path_list.index(line) % 1000 == 0:
            print('extracted ' + str(video_path_list.index(line)) + ' audios')

    # save the id of video with no audio
    with open('no_audio_id.txt', 'w') as f:
        for i in no_audio_id:
            f.write(i + '\n')

    # save the generated audio path
    with open('generated_audio_path.txt', 'w') as f:
        for i in generated_audio_path:
            f.write(i + '\n')


# extract frames
# extract_frames_from_video(
#     video_path_list, '/data1/yq/004_intention/000_m3_dataset/frames/', 1)
# print('extracted all frames')
extract_audio(audio_path_list)
print('extracted all audios')
