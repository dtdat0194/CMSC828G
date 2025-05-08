import os 
import torch
import torch.nn as nn
import torch.nn.functional as F

from Backbones import VisionClassifier, AudioClassifier
import utils



video_dataset_path = "/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/clips"
audio_dataset_path = "/cmlscratch/xyu054/DistMM/Contrastive_Learning_ResNet/dataset/audio"
video_name_list = os.listdir(video_dataset_path)
audio_name_list = os.listdir(audio_dataset_path)
count = 0
name_path_dict = {}

#video_pure_names = []
for video_name in video_name_list:
    print(video_name)
    current_name = video_name[:-10]
    video_path = video_dataset_path + "/" + video_name
    name_path_dict[current_name] = {"video_path":video_path}
    
    #video_pure_names.append(current_name)
#print(name_path_dict)
for audio_name in audio_name_list:
    print(audio_name)
    current_name = audio_name[:-10]
    audio_path = audio_dataset_path + "/" + audio_name
    #print(name_path_dict[current_name])
    #print(1)
    name_path_dict[current_name]["audio_path"] = audio_path
    #print(video_path, current_name)
#print(count)
#print(len(video_name_list))
#print(len(audio_name_list))
