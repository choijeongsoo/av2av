import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from av2unit.inference import load_model as load_av2unit_model
from unit2unit.inference import load_model as load_unit2unit_model
from unit2av.inference import load_model as load_unit2av_model, load_speaker_encoder_model

from util import process_units, extract_audio_from_video, save_video

class AVSpeechToAVSpeechPipeline:
    def __init__(self,
        av2unit_model, av2unit_task,
        unit2unit_task, unit2unit_generator,
        unit2av_model, speaker_encoder,
        use_cuda=False
    ):
        self.av2unit_model = av2unit_model
        self.av2unit_task = av2unit_task
        self.unit2unit_task = unit2unit_task
        self.unit2unit_generator = unit2unit_generator
        self.unit2av_model = unit2av_model
        self.speaker_encoder = speaker_encoder
        self.use_cuda = use_cuda

    def process_av2unit(self, lip_video_path, audio_path):
        task = self.av2unit_task
        video_feats, audio_feats = task.dataset.load_feature((lip_video_path, audio_path))
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if task.dataset.normalize and 'audio' in task.dataset.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
        collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))

        sample = {"source": {
            "audio": collated_audios, "video": collated_videos,
        }}
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        pred = task.inference(
            self.av2unit_model,
            sample,
        )
        pred_str = task.dictionaries[0].string(pred.int().cpu())

        return pred_str

    def process_unit2unit(self, unit):
        task = self.unit2unit_task
        unit = list(map(int, unit.strip().split()))
        unit = task.source_dictionary.encode_line(
            " ".join(map(lambda x: str(x), process_units(unit, reduce=True))),
            add_if_not_exist=False,
            append_eos=True,
        ).long()
        unit = torch.cat([
            unit.new([task.source_dictionary.bos()]),
            unit,
            unit.new([task.source_dictionary.index("[{}]".format(task.source_language))])
        ])

        sample = {"net_input": {
            "src_tokens": torch.LongTensor(unit).view(1,-1),
        }}
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        pred = task.inference_step(
            self.unit2unit_generator,
            None,
            sample,
        )[0][0]

        pred_str = task.target_dictionary.string(
            pred["tokens"].int().cpu(),
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.unit2unit_generator)
        )

        return pred_str

    def process_unit2av(self, unit, audio_path, video_path, bbox_path):
        unit = list(map(int, unit.strip().split()))

        sample = {
            "code": torch.LongTensor(unit).view(1,-1),
            "spkr": torch.from_numpy(self.speaker_encoder.get_embed(audio_path)).view(1,1,-1),
        }
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        wav, video, full_video, bbox = self.unit2av_model(sample, video_path, bbox_path, dur_prediction=True)

        return wav, video, full_video, bbox

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    av2unit_model, av2unit_task = load_av2unit_model(args.av2unit_path, args.modalities, use_cuda=use_cuda)
    unit2unit_task, unit2unit_generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)
    cfg_path = os.path.join("unit2av", "config.json")
    unit2av_model = load_unit2av_model(args.unit2av_path, cfg_path, args.tgt_lang, use_cuda=use_cuda)
    speaker_encoder_model = load_speaker_encoder_model(os.path.join("unit2av", "encoder.pt"), use_cuda=use_cuda)

    pipeline = AVSpeechToAVSpeechPipeline(
        av2unit_model, av2unit_task,
        unit2unit_task, unit2unit_generator,
        unit2av_model, speaker_encoder_model,
        use_cuda=use_cuda
    )

    temp_audio_path = os.path.splitext(args.in_vid_path)[0]+".temp.wav"
    lip_video_path = os.path.splitext(args.in_vid_path)[0]+".lip.mp4"
    bbox_path = os.path.splitext(args.in_vid_path)[0]+".bbox.pkl"
    extract_audio_from_video(args.in_vid_path, temp_audio_path)

    src_unit = pipeline.process_av2unit(lip_video_path, temp_audio_path)
    tgt_unit = pipeline.process_unit2unit(src_unit)
    tgt_audio, tgt_video, full_video, bbox = pipeline.process_unit2av(tgt_unit, temp_audio_path, args.in_vid_path, bbox_path)

    save_video(tgt_audio, tgt_video, full_video, bbox, args.out_vid_path)

    os.remove(temp_audio_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-vid-path", type=str, required=True, help="File path of source video input"
    )
    parser.add_argument(
        "--out-vid-path", type=str, required=True, help="File path of translated video output"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument(
        "--modalities", type=str, default="audio,video", help="input modalities",
        choices=["audio,video","audio","video"],
    )
    parser.add_argument(
        "--av2unit-path", type=str, required=True, help="path to the mAV-HuBERT pre-trained model"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
    )
    parser.add_argument(
        "--unit2av-path", type=str, required=True, help="path to the Unit AV Renderer"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
