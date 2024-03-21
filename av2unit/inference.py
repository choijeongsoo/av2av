import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils

from util import process_units, save_unit, extract_audio_from_video
from av2unit.task import AVHubertUnitPretrainingTask

def load_model(model_path, modalities, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])

    for model in models:
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()            
        model.prepare_for_inference_(cfg)

    task.cfg.modalities = modalities.split(",")
    task.load_dataset()

    return models[0], task

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    model, task = load_model(args.av2unit_path, args.modalities, use_cuda=use_cuda)

    temp_audio_path = os.path.splitext(args.in_vid_path)[0]+".temp.wav"
    lip_video_path = os.path.splitext(args.in_vid_path)[0]+".lip.mp4"
    extract_audio_from_video(args.in_vid_path, temp_audio_path)

    video_feats, audio_feats = task.dataset.load_feature((lip_video_path, temp_audio_path))
    audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
    if task.dataset.normalize and 'audio' in task.dataset.modalities:
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

    collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
    collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))

    sample = {"source": {
        "audio": collated_audios, "video": collated_videos,
    }}
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    pred = task.inference(
        model,
        sample,
    )
    pred_str = task.dictionaries[0].string(pred.int().cpu())

    save_unit(pred_str, args.out_unit_path)
    os.remove(temp_audio_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-vid-path", type=str, required=True, help="File path of source video input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="File path of target unit output"
    )
    parser.add_argument(
        "--av2unit-path", type=str, required=True, help="path to the mAV-HuBERT pre-trained model"
    )
    parser.add_argument(
        "--modalities", type=str, default="audio,video", help="input modalities",
        choices=["audio,video","audio","video"],
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
