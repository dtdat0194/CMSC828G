import math
from typing import Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from Backbones import VisionClassifier, AudioClassifier

#Might not use this, but going to test it out.
from Contrastive import ModalityEncoder

class MultiModalMultiGPUAudioVideoEncoder(nn.module):

    def __init__(
        self,
        video_gpu_devices: Sequence[int],
        audio_gpu_device: int = 0,
        proj_dim: int = 256,
        temerature: float = 0.07,

    ):
        super().__init__()

        #Going to try and build the Encoders onto their dedicated GPUs

        self.audio_gpu_device = torch.device(f"cuda:{audio_gpu_device}")
        
        #incase we wanna do vision in the future
        self.video_encoder = ModalityEncoder("vision", proj_dim)

        self.audio_encoder = ModalityEncoder("audio", proj_dim).to(self.audio_gpu_device)

        #self.video_gpu_devices = [torch.device(f"cuda:{d}") for d in video_gpu_devices]

        #audio single GPU
        #self.audio_encoder = ModalityEncoder("audio", proj_dim).to(self.audio_data)

        #One GPU per Video
        '''
        self.video_encoder = nn.ModuleList(
            [ModalityEncoder("vision", proj_dim).to(dev) for dev in self.video_data]
        )
        '''

        self.register_buffer(
            "temperature", torch.tensor(temerature, dtype=torch.float32)
        )

        self.video_gpu_devices = video_gpu_devices

    #calling the physical GPU id that our process owns.
    def ddp_wrapper_video(self, gpu_ids):
        #moving and wrapping the vision encoder if we want to do vision in the future.
        self.video_encoder = self.video_encoder.cuda(gpu_ids)
        self.video_encoder = DDP(self.video_encoder, device_ids=[gpu_ids], output_device=gpu_ids)

    
    #helper functions
    """
    def _chunk_video(self, video: torch.Tensor) -> List[torch.Tensor]:

        chunks = []

        n_chucks = len(self.video_gpu_devices)
        base_value = math.ceil(video.size(0) / n_chucks)

        for i in range(n_chucks):
            start_chunk, end_chunk = i * base_value, min((i+1) * base_value, video.size(0))
            if start_chunk < end_chunk:
                chunks.append(video[start_chunk:end_chunk])
        return chunks
    """
    @staticmethod
    def _info_nce(z1, z2, tempature):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / tempature
        labels = torch.arange(z1.size(0), device=logits.device)

        mean_loss = F.cross_entropy(logits, labels)
        return mean_loss
    
    #gathering all video embeddings from 1-k, computing the audio embeddings, and evaluating the InfoNCE loss. 
    def forward(self, video, audio_cpu):

        #doing forward on audios own GPU
        z_video_local_gpu = self.video_encoder(video)
        
        world = distributed.get_world_size()
        rank = distributed.get_rank()
        '''
        audio_branch = batch["audio"].to(self.audio_gpu_device, non_blocking=True)
        z_audio = self.aduio_encoder(audio_branch)
        '''
        
        #gathering raw audio tensors batching them to GPU0 and getting gradient info.
        #This is also where I'm making it so that the audio lives on GPU0 but it also communicates
        #with the other GPUs. All gather on GPU 0 shouldn't need to run Video encoder or the loss on every GPU.
        sizes = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world)]
        local_audio = audio_cpu.cuda(rank, non_blocking=True)
        local_size = torch.tensor([local_audio.size(0)], device='cuda')
        distributed.all_gather(sizes, local_size)

        max_size = int(max(s.item() for s in sizes))
        pad = (0,) * local_audio.dim()*2-2 + (0, max_size - local_audio.size(0))
        padded = F.pad(local_audio, pad)
        gathered = [torch.empty_like(padded) for _ in range(world)]
        distributed.all_gather(gathered, padded)
        
        if rank == 0:
            audio_full = torch.cat([g[:s.item()] for g, s in zip(gathered,sizes)], 0)
            z_audio_full = self.audio_gpu_device(audio_full)
        #Video going to all other devices.
        else:
            z_audio_full = torch.empty((sum(int(s) for s in sizes), z_video_local_gpu.size(1)), device='cuda')

        distributed.broadcast(z_video_local_gpu, src=0)

        start = sum(int(sizes[r].item()) for r in range(rank))
        end = start + sizes[rank].item()
        z_video_local_gpu = z_audio_full[start:end]

        #doing video along with a scatter, encode, and gather
        '''
        video_chunks = self._chunk_video(batch["video"])
        #zv_list, start = [], 0
        zv_list = []
        
        for chunk, encode, device in zip(video_chunks, self.video_encoder, self.video_gpu_devices):
            chunk = chunk.to(device, non_blocking=True)
            zv = encode(chunk)
            zv_list.append(zv.to(self.audio_gpu_device))
        
        z_video = torch.cat(zv_list, dim=0)
        #InfoONCE

        temperature = self.temperature
        '''
        loss = 0.5 * (

            self._info_nce(z_video_local_gpu, z_audio_full, self.temperature) + self._info_nce(z_audio_full, z_video_local_gpu, self.temperature)

        )

        return loss