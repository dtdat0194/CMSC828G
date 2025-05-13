import math
from typing import Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Backbones import VisionClassifier, AudioClassifier

#Might not use this, but going to test it out.
from Contrastive import ModalityEncoder

class MultiModalMultiGPUAudioVideoEncoder(nn.module):

    def __init__(
        self,
        video_gpu_devices: Sequence[int],
        audio_gpu_device: int = 0,
        proj_dim: int = 128,
        temerature: float = 0.07,

    ):
        super().__init__()

        #Going to try and build the Encoders onto their dedicated GPUs

        self.audio_gpu_device = torch.device(f"cuda:{audio_gpu_device}")
        self.video_gpu_devices = [torch.device(f"cuda:{d}") for d in video_gpu_devices]

        #audio single GPU
        self.aduio_encoder = ModalityEncoder("audio", proj_dim).to(self.audio_data)

        #One GPU per Video
        self.video_encoder = nn.ModuleList(
            [ModalityEncoder("vision", proj_dim).to(dev) for dev in self.video_data]
        )

        self.register_buffer(

            "temperature", torch.tensor(temerature, dtype=torch.float32)

        )

    #helper functions

    def _chunk_video(self, video: torch.Tensor) -> List[torch.Tensor]:

        chunks = []

        n_chucks = len(self.video_gpu_devices)
        base_value = math.ceil(video.size(0) / n_chucks)

        for i in range(n_chucks):
            start_chunk, end_chunk = i * base_value, min((i+1) * base_value, video.size(0))
            if start_chunk < end_chunk:
                chunks.append(video[start_chunk:end_chunk])
        return chunks
    
    @staticmethod
    def _info_nce(z1, z2, tempature):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / tempature
        labels = torch.arange(z1.size(0), device=logits.device)

        mean_loss = F.cross_entropy(logits, labels)
        return mean_loss
    
    def forward(self, batch):


        #doing forward on audios own GPU
        audio_branch = batch["audio"].to(self.audio_gpu_device, non_blocking=True)
        z_audio = self.aduio_encoder(audio_branch)

        #doing video along with a scatter, encode, and gather

        video_chunks = self._chunk_video(batch["video"])
        zv_list, start = [], 0

        for chunk, encode, device in zip(video_chunks, self.video_encoder, self.video_gpu_devices):
            chunk = chunk.to(device, non_blocking=True)
            zv = encode(chunk)
            zv_list.append(zv.to(self.audio_gpu_device))

        z_video = torch.cat(zv_list, dim=0)


        #InfoONCE

        t = self.temperature

        loss = 0.5 * (

            self._info_nce(z_audio, t) + self._info_nce(z_audio,z_video, t)

        )

        return loss, z_audio, z_video