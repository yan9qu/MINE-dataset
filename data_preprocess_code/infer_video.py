import pickle
import os
import cv2
from mmcv import Config, DictAction
import numpy as np
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'


def get_feat(frames, model):    # alternative way
    feat = model.extract_feat(frames)

    # mean pooling
    feat = feat.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]

    # project
    batch_size, hidden_dim = feat.shape
    # print(feat.shape)
    feat_dim = 1024
    proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))  # .cuda()

    # final output
    output = feat @ proj  # [batch_size, feat_dim]
    return output


cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None,
                    test_cfg=cfg.get('test_cfg'))  # .cuda()
load_checkpoint(model, checkpoint, map_location='cpu')

# # [batch_size, channel, temporal_dim, height, width]
# dummy_x = torch.rand(1, 3, 32, 224, 224)

# frame_folder = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/first_validation_frames/economy_1608605411728568321'

frame_folder_father = '/data1/yq/004_intention/000_m3_dataset/frames'

# split the frame_folder_father into 10 parts
# partial_frame_folder_father = '/data1/yq/004_intention/000_m3_dataset/frames_parts_for_extraction'

# copy the frame_folder_father to partial_frame_folder_father
# count = 0
# for frame_folder in os.listdir(frame_folder_father):
#     # every 300 folders, create a new folder
#     if count % 300 == 0:
#         batch = count // 300
#         new_folder = os.path.join(partial_frame_folder_father, str(batch))
#         os.mkdir(new_folder)
#     # copy the frame_folder to the new folder
#     os.system('cp -r ' + os.path.join(frame_folder_father, frame_folder) +
#               ' ' + os.path.join(partial_frame_folder_father, str(batch)))
#     count += 1

have_saved_id_list = '/data1/yq/004_intention/missing_aware_prompts/datasets/m3/Video-Swin-Transformer/have_saved_id_list.txt'

# load the have_saved_id_list
saved_id = []
with open(have_saved_id_list, 'r') as f:
    for line in f.readlines():
        saved_id.append(line.strip())


def extract_feat_from_folder(frame_folder_father, batch):
    frame_folders = os.listdir(frame_folder_father)
    video_feat = {}
    video_feat_partial = {}
    count = 0

    for frame_folder in frame_folders:
        video_id = frame_folder
        if video_id in saved_id:
            continue
        # load frames and convert to tensor
        frames = []
        # get all frames from frame_folder
        frame_path = os.path.join(frame_folder_father, frame_folder)
        for i in os.listdir(frame_path):
            frames.append(cv2.imread(os.path.join(frame_path, i)))
        # resize frames to 224x224
        frames = [cv2.resize(i, (224, 224)) for i in frames]
        # convert to np array
        frames = np.array(frames)
        # if frames > 64, sample 64 frames
        if frames.shape[0] > 64:
            frames = frames[::frames.shape[0]//64]
        # convert to tensor and permute to [batch_size, channel, temporal_dim, height, width]
        dummy_x = torch.tensor(frames).permute(
            3, 0, 1, 2).unsqueeze(0).float()  # .cuda()
        feat = get_feat(dummy_x, model)
        video_feat[video_id] = feat.cpu().detach().numpy()
        video_feat_partial[video_id] = feat.cpu().detach().numpy()
        # clear memory
        del frames
        del dummy_x
        del feat
        torch.cuda.empty_cache()
        count += 1
        # print id

        # every 100 videos save once
        if count % 100 == 0:
            print(count)
            partial_batch = count // 100
            # saving a partial file
            save_path = '/data1/yq/004_intention/000_m3_dataset/video_feat' + \
                str(batch)+'_'+str(video_id)+'.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(video_feat_partial, f)
            video_feat_partial = {}
            # save the video_id to have_saved_id_list
            video_id_list = list(video_feat.keys())
            with open(have_saved_id_list, 'a') as f:
                for i in video_id_list:
                    f.write(i+'\n')
            print('partial file saved')

    save_path = '/data1/yq/004_intention/000_m3_dataset/video_feat' + \
        str(batch)+'.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(video_feat, f)


father_path = '/data1/yq/004_intention/000_m3_dataset/frames_parts_for_extraction'

# list all folders in father_path
folders = os.listdir(father_path)
batch = [3, 6, 7, 8, 9, 10, 11, 12]
# extract features from each folder
for i in batch:
    extract_feat_from_folder(os.path.join(father_path, str(i)), i)
    print('batch '+str(i)+' finished')
