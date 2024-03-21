import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from fairseq.models.text_to_speech.codehifigan import CodeGenerator as CodeHiFiGANModel
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

import torchvision
import pickle
import numpy as np
import cv2

class UnitAVRenderer(CodeHiFiGANVocoder):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], lang: str, fp16: bool = False
    ) -> None:
        super(CodeHiFiGANVocoder, self).__init__()
        self.model = CodeHiFiGANModel_spk(model_cfg)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["audio"][lang])
        self.model.eval()

        self.face_model = FaceRenderer(unit_num=model_cfg["num_embeddings"])
        self.face_model.load_state_dict(state_dict["video"])
        self.face_model.eval()

        if fp16:
            self.model.half()
            self.face_model.half()
        self.model.remove_weight_norm()

        units_per_second = 50
        frames_per_second = 25
        self.num_frames = 10
        self.code_frame_ratio = units_per_second // frames_per_second
        self.num_units = self.num_frames * self.code_frame_ratio

    def get_crops(self, bbox_path):
        bbs = pickle.load(open(bbox_path, 'rb'))
        prev_val = None
        for i in range(len(bbs)):
            if bbs[i] is None:
                bbs[i] = prev_val
            else:
                prev_val = bbs[i]
        return np.array(bbs)

    def read_window(self, frames, crops):
        window = []
        for img, (x1, y1, x2, y2) in zip(frames, crops):
            img = img[max(int(y1), 0): int(y2), max(int(x1), 0):int(x2)]
            img = cv2.resize(img, (96, 96))
            window.append(img)
        return window 

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def forward(self, x: Dict[str, torch.Tensor], video_path: str, bbox_path: str, dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        if dur_prediction:
            x["code"] = torch.unique_consecutive(x["code"])

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        gen_wav, dedup_code = self.model(**x)
        gen_wav = gen_wav.detach().squeeze().cpu().numpy()

        tgt_len = len(dedup_code) // self.code_frame_ratio
        remain = len(dedup_code) % self.num_units
        if remain != 0:
            repeat_num = self.num_units - remain
            dedup_code = torch.cat([dedup_code, dedup_code[-1].repeat(repeat_num)])
        padded_tgt_len = len(dedup_code) // self.code_frame_ratio
        
        frames = torchvision.io.read_video(video_path, pts_unit="sec")[0]
        len_frames = len(frames)
        reverse_frames = frames.flip(0)
        repeated_frames = torch.cat((reverse_frames[1:], frames[1:]))
        while len(frames) < padded_tgt_len:
            frames = torch.cat([frames, repeated_frames])
        frames = frames[:padded_tgt_len]
        frames = frames.flip(-1)
        
        crops = self.get_crops(bbox_path)
        assert len(crops) == len_frames
        reverse_crops = crops[::-1]
        repeated_crops = np.concatenate([reverse_crops[1:], crops[1:]])
        while len(crops) < padded_tgt_len:
            crops = np.concatenate([crops, repeated_crops])
        crops = crops[:padded_tgt_len]

        frames_numpy = np.array(frames)
        window = self.read_window(frames_numpy, crops)
        wrong_window = window.copy()

        dedup_code_seq = dedup_code.view(-1, self.num_units)

        window = self.prepare_window(window)
        window[:, :, window.shape[2] // 2:] = 0.
        wrong_window = self.prepare_window(wrong_window)
        windows = np.concatenate([window, wrong_window], axis=0)
        windows = torch.FloatTensor(windows).to(dedup_code_seq.device)
        windows = windows.transpose(1,0)

        gen_vid = self.face_model(dedup_code_seq, windows)
        gen_vid = (gen_vid.detach().cpu().numpy().transpose(0,2,3,1)* 255.).astype(np.uint8)

        return gen_wav, gen_vid[:tgt_len], frames_numpy[:tgt_len], crops[:tgt_len]


class CodeHiFiGANModel_spk(CodeHiFiGANModel):
    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)

        if self.dur_predictor and kwargs.get("dur_prediction", False):
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.f0:
            if self.f0_quant_embed:
                kwargs["f0"] = self.f0_quant_embed(kwargs["f0"].long()).transpose(1, 2)
            else:
                kwargs["f0"] = kwargs["f0"].unsqueeze(1)

            if x.shape[-1] < kwargs["f0"].shape[-1]:
                x = self._upsample(x, kwargs["f0"].shape[-1])
            elif x.shape[-1] > kwargs["f0"].shape[-1]:
                kwargs["f0"] = self._upsample(kwargs["f0"], x.shape[-1])
            x = torch.cat([x, kwargs["f0"]], dim=1)

        if self.multispkr:
            assert (
                "spkr" in kwargs
            ), 'require "spkr" input for multispeaker CodeHiFiGAN vocoder'
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ["spkr", "code", "f0", "dur_prediction"]:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super(CodeHiFiGANModel, self).forward(x), torch.repeat_interleave(kwargs["code"], dur_out.view(-1))


class FaceRenderer(nn.Module):
    def __init__(self, unit_num):
        super(FaceRenderer, self).__init__()
        self.unit_num = unit_num
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.unit_embed = nn.Embedding(self.unit_num, 512)
        self.unit2lip = nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=1024, dropout=0.1, activation='relu')

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  
                            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                        Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ), 

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ]) 

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        audio_sequences = self.unit_embed(audio_sequences) # B,20,512 / T/10,20,512 
        audio_sequences = F.interpolate(audio_sequences.permute(0, 2, 1), scale_factor=0.5, mode='linear')  # B,512,10 / T/10,512,10
        audio_sequences = audio_sequences.permute(2, 0, 1) # 10,B,512 / 10,T/10,512
        audio_embedding = self.unit2lip(audio_sequences).permute(1,0,2)  # B,10,512
        audio_embedding = audio_embedding.contiguous().view(-1, 512).unsqueeze(-1).unsqueeze(-1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        outputs = self.output_block(x)
        return outputs
    
class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
    
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)
    