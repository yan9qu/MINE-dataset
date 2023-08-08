from moviepy.editor import *
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import argparse
import pickle
import argparse
import librosa
import torch

__all__ = ['AudioFeature']


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_path', type=str, default='raw_video',
                        help="The directory of the raw video path.")
    parser.add_argument('--audio_data_path', type=str, default='/data1/yq/004_intention/final_audio_list.txt',
                        help="The directory of the audio data path.")
    parser.add_argument('--raw_audio_path', type=str, default='raw_audio',
                        help="The directory of the raw audio path.")
    parser.add_argument("--audio_feats_path", type=str,
                        default='/data1/yq/004_intention/missing_aware_prompts/datasets/m3/MIntRec/tools/audio_feats_only_exist.pkl', help="The directory of audio features.")

    args = parser.parse_args()

    return args


class AudioFeature:

    def __init__(self, args):

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")

        # self.__get_raw_audio(args)

        audio_feats = self.__gen_feats_from_audio(args, use_wav2vec2=True)
        # self.__save_audio_feats(args, audio_feats)

    def __gen_feats_from_audio(self, args, use_wav2vec2=False):

        have_saved_id_list = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/MIntRec/tools/have_saved_id_list.txt'

        # load the have_saved_id_list
        saved_id = []
        with open(have_saved_id_list, 'r') as f:
            for line in f.readlines():
                saved_id.append(line.strip())
        audio_path_txt = args.audio_data_path
        audio_paths = []
        # read audio_file_list from audio_path_txt
        with open(audio_path_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                audio_paths.append(line.strip())
        audio_feats = {}
        count = 0

        for audio_path in audio_paths:

            audio_id = audio_path.split('/')[-2]
            audio_id_str = str(audio_id)
            if audio_id_str in saved_id:
                print('audio_id {} has been saved'.format(audio_id_str))
                continue
            print('audio_id {}'.format(audio_id_str))
            wav2vec2_feats = self.__process_audio(audio_path)
            # max length of wav2vec2_feats is 60
            if wav2vec2_feats.shape[0] > 60:
                #  subsample wav2vec2_feats to 60, with padding
                sample_rate = wav2vec2_feats.shape[0] // 60
                new_data = wav2vec2_feats[::sample_rate].clone()
                if new_data.shape[0] < 60:
                    padding = torch.zeros(60-new_data.shape[0], 768)
                    final_data = torch.cat((new_data, padding), 0)
                else:
                    final_data = new_data[:60].clone()
                audio_feats[audio_id_str] = final_data
                # print shape
                print('wav2vec2_feats.shape: {}'.format(final_data.shape))
            else:
                # padding wav2vec2_feats to 60
                padding = torch.zeros(60-wav2vec2_feats.shape[0], 768)
                wav2vec2_feats = torch.cat((wav2vec2_feats, padding), 0)
                # print shape
                print('wav2vec2_feats.shape: {}'.format(wav2vec2_feats.shape))
                audio_feats[audio_id_str] = wav2vec2_feats
            count += 1
            if count % 100 == 0:
                # save audio_feats
                partial_save_path = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/MIntRec/tools' + \
                    str(audio_id)+'_audio_feats.pkl'
                with open(partial_save_path, 'wb') as f:
                    pickle.dump(audio_feats, f)
                # save have_saved_id_list
                audio_ids = list(audio_feats.keys())
                with open(have_saved_id_list, 'a') as f:
                    for audio_id in audio_ids:
                        f.write(audio_id+'\n')
                # print
                print('have saved {} audio feats'.format(count))
                audio_feats = {}
        # save audio_feats
        partial_save_path = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/MIntRec/tools' + \
            str(audio_id)+'_audio_feats.pkl'
        with open(partial_save_path, 'wb') as f:
            pickle.dump(audio_feats, f)
        # save have_saved_id_list
        audio_ids = list(audio_feats.keys())
        with open(have_saved_id_list, 'a') as f:
            for audio_id in audio_ids:
                f.write(audio_id+'\n')
        # print
        print('have saved {} audio feats'.format(count))

        return audio_feats

    def __process_audio(self, read_file_path):

        y, sr = librosa.load(read_file_path, sr=16000)
        audio_feats = self.processor(
            y, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            audio_feats = self.model(audio_feats).last_hidden_state.squeeze(0)

        return audio_feats

    def __save_audio_feats(self, args, audio_feats):

        audio_feats_path = args.audio_feats_path

        with open(audio_feats_path, 'wb') as f:
            pickle.dump(audio_feats, f)


if __name__ == '__main__':

    args = parse_arguments()
    audio_data = AudioFeature(args)
