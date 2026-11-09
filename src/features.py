"""Audio loading, windowing and log-mel features.

Follows the original one-pass detector feature pipeline (github.com/satvik-venkatesh/you-only-hear-once,
MIT): mono downmix, fixed-length windows with zero padding at the end, per-window
peak normalisation, librosa mel power spectrogram, `power_to_db`.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


@dataclass
class FeatureParams:
    sr: int = 44100
    win_len: float = 2.56
    hop_len: float = 1.96
    n_fft: int = 2048
    stft_hop: int = 441
    stft_win: int = 1764
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = 22050.0
    causal_norm: bool = False

    @property
    def n_frames(self) -> int:
        return 1 + int(self.sr * self.win_len) // self.stft_hop

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "tut2017": FeatureParams(),
    "urbansed": FeatureParams(win_len=10.0, hop_len=10.0),
    "music_speech": FeatureParams(
        sr=16000,
        win_len=8.0,
        hop_len=8.0,
        n_fft=512,
        stft_hop=160,
        stft_win=400,
        n_mels=64,
        fmin=125.0,
        fmax=7500.0,
    ),
    "voice": FeatureParams(),
    "urbansed_causal": FeatureParams(win_len=10.0, hop_len=10.0, causal_norm=True),
    "voice_causal": FeatureParams(win_len=10.0, hop_len=10.0, causal_norm=True),
    "desed_causal": FeatureParams(
        sr=16000,
        win_len=10.0,
        hop_len=10.0,
        n_fft=1024,
        stft_hop=160,
        stft_win=400,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
        causal_norm=True,
    ),
    "desed": FeatureParams(
        sr=16000,
        win_len=10.0,
        hop_len=10.0,
        n_fft=1024,
        stft_hop=160,
        stft_win=400,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
    ),
}


def load_mono(
    path: Union[str, Path], expected_sr: int, resample: bool = False
) -> np.ndarray:
    """Float64 mono signal; multichannel audio is averaged (as `sox ... channels 1`).

    With `resample=True`, mismatched sample rates are converted instead of
    raising. Off by default so a wrong preset cannot silently pass.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    if sr != expected_sr:
        if not resample:
            raise ValueError(
                f"{path}: sample rate {sr}, expected {expected_sr} "
                "(pass resample=True to convert)"
            )
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr)
    return audio


def window_audio(
    audio: np.ndarray, sr: int, win_len: float, hop_len: float
) -> Tuple[List[np.ndarray], List[float]]:
    """Zero-pad the tail so the windows cover the whole signal (`construct_examples`).

    Returns (windows, window start times in s).
    """
    win = int(sr * win_len)
    hop = int(sr * hop_len)
    if audio.shape[0] < win:
        padded = np.zeros(win)
    else:
        n_hops = math.ceil((audio.shape[0] - win) / hop)
        padded = np.zeros(win + hop * n_hops)
    padded[: audio.shape[0]] = audio
    ends = range(win, padded.shape[0] + 1, hop)
    return [padded[e - win : e] for e in ends], [(e - win) / sr for e in ends]


def log_mel(window: np.ndarray, p: FeatureParams) -> np.ndarray:
    """(n_frames, n_mels) float32 log-mel of one window."""
    import librosa

    if p.causal_norm:
        gain = np.maximum.accumulate(np.abs(window))
        y = window / np.maximum(gain, 1e-6)
    else:
        y = librosa.util.normalize(window)
    bands = librosa.feature.melspectrogram(
        y=y,
        sr=p.sr,
        n_fft=p.n_fft,
        hop_length=p.stft_hop,
        win_length=p.stft_win,
        n_mels=p.n_mels,
        fmin=p.fmin,
        fmax=p.fmax,
    )
    return librosa.power_to_db(bands).T.astype(np.float32)
