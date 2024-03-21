# This code is from https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/generate_waveform_from_code.py

import argparse
import os
import json
import torch

from fairseq import utils
from unit2av.model import UnitAVRenderer
from unit2av.model_speaker_encoder import SpeakerEncoder
from util import save_video, extract_audio_from_video

def load_model(model_path, cfg_path, lang, use_cuda=False):
    with open(cfg_path) as f:
        vocoder_cfg = json.load(f)
    vocoder = UnitAVRenderer(model_path, vocoder_cfg, lang)
    if use_cuda:
        vocoder = vocoder.cuda()
    return vocoder

def load_speaker_encoder_model(model_path, use_cuda=False):
    speaker_encoder = SpeakerEncoder(model_path)
    if use_cuda:
        speaker_encoder = speaker_encoder.cuda()
    return speaker_encoder

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    vocoder = load_model(args.unit2av_path, cfg_path, args.tgt_lang, use_cuda=use_cuda)
    speaker_encoder = load_speaker_encoder_model(os.path.join(os.path.dirname(__file__), "encoder.pt"), use_cuda=use_cuda)

    temp_audio_path = os.path.splitext(args.in_vid_path)[0]+".temp.wav"
    bbox_path = os.path.splitext(args.in_vid_path)[0]+".bbox.pkl"
    extract_audio_from_video(args.in_vid_path, temp_audio_path)

    with open(args.in_unit_path) as f:
        unit = list(map(int, f.readline().strip().split()))

    sample = {
        "code": torch.LongTensor(unit).view(1,-1),
        "spkr": torch.from_numpy(speaker_encoder.get_embed(args.in_vid_path)).view(1,1,-1),
    }
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    wav, video, full_video, bbox = vocoder(sample, args.in_vid_path, bbox_path, dur_prediction=True)

    save_video(wav, video, full_video, bbox, args.out_vid_path)

    os.remove(temp_audio_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit-path", type=str, required=True, help="File path of unit input"
    )
    parser.add_argument(
        "--in-vid-path", type=str, required=True, help="File path of video input"
    )
    parser.add_argument(
        "--in-bbox-path", type=str, required=True, help="File path of bounding box"
    )
    parser.add_argument(
        "--out-vid-path", type=str, required=True, help="File path of video output"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument(
        "--unit2av-path", type=str, required=True, help="path to the Unit AV Renderer"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()