"""
Modified from https://github.com/CorentinJ/Real-Time-Voice-Cloning
"""

import torch
from torch import nn
from dataclasses import dataclass
from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

@dataclass
class SpeakerEncoderConfig:

    ## Model parameters
    model_hidden_size = 256
    model_embedding_size = 256
    model_num_layers = 3

    ## Mel-filterbank
    mel_window_length = 25  # In milliseconds
    mel_window_step = 10    # In milliseconds
    mel_n_channels = 40

    ## Audio
    sampling_rate = 16000
    # Number of spectrogram frames in a partial utterance
    partials_n_frames = 160     # 1600 ms
    # Number of spectrogram frames at inference
    inference_n_frames = 80     #  800 ms

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out. 
    vad_moving_average_width = 8
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length = 6

    ## Audio volume normalization
    audio_norm_target_dBFS = -30

    int16_max = (2 ** 15) - 1


class SpeakerEncoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        
        self.cfg = SpeakerEncoderConfig()

        # Network defition
        self.lstm = nn.LSTM(input_size=self.cfg.mel_n_channels,
                            hidden_size=self.cfg.model_hidden_size, 
                            num_layers=self.cfg.model_num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.cfg.model_hidden_size, 
                                out_features=self.cfg.model_embedding_size)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict["model_state"])
        self.eval()

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds


    def preprocess_wav(
                    self,
                    fpath_or_wav: Union[str, Path, np.ndarray],
                    source_sr: Optional[int] = None,
                    normalize: Optional[bool] = True,
                    trim_silence: Optional[bool] = True):
        """
        Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
        either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

        :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
        just .wav), either the waveform as a numpy array of floats.
        :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
        preprocessing. After preprocessing, the waveform's sampling rate will match the data 
        hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
        this argument will be ignored.
        """
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
        else:
            wav = fpath_or_wav
        
        # Resample the wav if needed
        if source_sr is not None and source_sr != self.cfg.sampling_rate:
            wav = librosa.resample(wav, source_sr, self.cfg.sampling_rate)
            
        # Apply the preprocessing: normalize volume and shorten long silences 
        if normalize:
            wav = self.normalize_volume(wav, self.cfg.audio_norm_target_dBFS, increase_only=True)
        if webrtcvad and trim_silence:
            wav = self.trim_long_silences(wav)
        
        return wav


    def wav_to_mel_spectrogram(self, wav):
        """
        Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
        Note: this not a log-mel spectrogram.
        """
        frames = librosa.feature.melspectrogram(
            wav,
            self.cfg.sampling_rate,
            n_fft=int(self.cfg.sampling_rate * self.cfg.mel_window_length / 1000),
            hop_length=int(self.cfg.sampling_rate * self.cfg.mel_window_step / 1000),
            n_mels=self.cfg.mel_n_channels
        )
        return frames.astype(np.float32).T


    def trim_long_silences(self, wav):
        """
        Ensures that segments without voice in the waveform remain no longer than a 
        threshold determined by the VAD parameters in params.py.

        :param wav: the raw waveform as a numpy array of floats 
        :return: the same waveform with silences trimmed away (length <= original wav length)
        """
        # Compute the voice detection window size
        samples_per_window = (self.cfg.vad_window_length * self.cfg.sampling_rate) // 1000
        
        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        
        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * self.cfg.int16_max)).astype(np.int16))
        
        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                            sample_rate=self.cfg.sampling_rate))
        voice_flags = np.array(voice_flags)
        
        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        
        audio_mask = moving_average(voice_flags, self.cfg.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)
        
        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.cfg.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        
        return wav[audio_mask == True]


    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
        if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
            return wav
        return wav * (10 ** (dBFS_change / 20))


    def embed_frames_batch(self, frames_batch):
        """
        Computes embeddings for a batch of mel spectrogram.

        :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """
        frames = torch.from_numpy(frames_batch).to(next(self.parameters()).device)
        embed = self.forward(frames).detach().cpu().numpy()
        return embed


    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=None,
                            min_pad_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        if partial_utterance_n_frames is None:
            partial_utterance_n_frames = self.cfg.inference_n_frames

        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((self.cfg.sampling_rate * self.cfg.mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices


    def embed_utterance(self, wav, **kwargs):
        """
        Computes an embedding for a single utterance.

        # TODO: handle multiple wavs to benefit from batching on GPU
        :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
        :param using_partials: if True, then the utterance is split in partial utterances of
        <partial_utterance_n_frames> frames and the utterance embedding is computed from their
        normalized average. If False, the utterance is instead computed from feeding the entire
        spectogram to the network.
        :param return_partials: if True, the partial embeddings will also be returned along with the
        wav slices that correspond to the partial embeddings.
        :param kwargs: additional arguments to compute_partial_splits()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned. If <using_partials> is simultaneously set to False, both these values will be None
        instead.
        """

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        frames = self.wav_to_mel_spectrogram(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        partial_embeds = self.embed_frames_batch(frames_batch)

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        return embed

    def get_embed(self, wav_path):
        wav_preprocessed = self.preprocess_wav(wav_path)
        embed = self.embed_utterance(wav_preprocessed)
        return embed