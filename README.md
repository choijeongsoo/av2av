# AV2AV

Official PyTorch implementation for the following paper:
> **AV2AV: Direct Audio-Visual Speech to Audio-Visual Speech Translation with Unified Audio-Visual Speech Representation**<br>
> [Jeongsoo Choi](https://choijeongsoo.github.io)\*, [Se Jin Park](https://sites.google.com/view/sejinpark)\*, [Minsu Kim](https://sites.google.com/view/ms-dot-k)\*, [Yong Man Ro](https://www.ivllab.kaist.ac.kr/people/professor)<br>
> CVPR 2024<br>
> \[[Paper](https://arxiv.org/abs/2312.02512)\] \[[Demo](https://choijeongsoo.github.io/av2av)\]

<div align="center"><img width="60%" src="imgs/fig1.png?raw=true"/></div>

## Method
<div align="center"><img width="100%" src="imgs/fig2.png?raw=true"/></div>

## Setup
- Python >=3.7,<3.11
```
git clone -b main --single-branch https://github.com/choijeongsoo/av2av
cd av2av
git submodule init
git submodule update
pip install -e fairseq
pip install -r requirements.txt
conda install "ffmpeg<5" -c conda-forge
```

## Dataset
  Name | Language | Link
  |---|---|---
  LRS3 | English | [here](https://mmai.io/datasets/lip_reading)
  mTEDx | Spanish, French, Italian, and Portuguese | [here](https://www.openslr.org/100)
- We use curated lists of [this work](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) for filtering mTEDx.
- For more details, please refer to the 'Dataset' section in our paper.

### Data Preprocessing
- We follow [Auto-AVSR](https://github.com/mpc001/auto_avsr) to preprocess audio-visual data.

## Model Checkpoints
  Stage | Download Link
  |---|---
  AV Speech Unit Extraction | [mavhubert_large_noise.pt](https://drive.google.com/file/d/18qiZkzrhDtaaleUPiOXf8WDKS9ZumUTa/view?usp=sharing)
  Multilingual AV2AV Translation | [utut_sts_ft.pt](https://drive.google.com/file/d/1iT3EItPn8uqM5vMTJSCGztzCzNouHzNb/view?usp=sharing)
  Zero-shot AV-Renderer | [unit_av_renderer.pt](https://drive.google.com/file/d/1KhtkV8TkdE_GBhLQy-CQKJpfJThQr1P4/view?usp=sharing)

## Inference

### Pipeline for Audio-Visual Speech to Audio-Visual Speech Translation (AV2AV)
```
$ cd av2av
$ PYTHONPATH=fairseq python inference.py \
  --in-vid-path samples/en/TRajLqEaWhQ_00002.mp4 \
  --out-vid-path samples/es/TRajLqEaWhQ_00002.mp4 \
  --src-lang en --tgt-lang es \
  --av2unit-path /path/to/mavhubert_large_noise.pt \
  --utut-path /path/to/utut_sts_ft.pt \
  --unit2av-path /path/to/unit_av_renderer.pt \
```
- Our model supports 5 languages: en (English), es (Spanish), fr (French), it (Italian), pt (Portuguese)

## Acknowledgement
This repository is built upon [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [UTUT](https://github.com/choijeongsoo/utut), [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), and [Fairseq](https://github.com/pytorch/fairseq). We appreciate the open-source of the projects.

## Citation

If our work is useful for your research, please consider citing the following papers:
```bibtex
@article{choi2023av2av,
  title={AV2AV: Direct Audio-Visual Speech to Audio-Visual Speech Translation with Unified Audio-Visual Speech Representation},
  author={Jeongsoo Choi and Se Jin Park and Minsu Kim and Yong Man Ro},
  journal={arXiv preprint arXiv:2312.02512},
  year={2023}
}
@article{kim2023many,
    title={Many-to-Many Spoken Language Translation via Unified Speech and Text Representation Learning with Unit-to-Unit Translation},
    author={Minsu Kim and Jeongsoo Choi and Dahun Kim and Yong Man Ro},
    journal={arXiv preprint arXiv:2308.01831},
    year={2023}
}
