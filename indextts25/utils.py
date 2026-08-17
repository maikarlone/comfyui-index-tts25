import os
import tempfile
import threading
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# Shared lock for model load / unload / CUDA inference across all nodes.
TTS_INFER_LOCK = threading.Lock()


def process_comfy_audio(audio: Any, *, allow_none: bool = False) -> Optional[Tuple[np.ndarray, int]]:
    """
    Convert ComfyUI AUDIO dict or (wave, sr) into float32 mono (wave, sr).
    When allow_none=True, None returns None; otherwise raises.
    """
    if audio is None:
        if allow_none:
            return None
        raise ValueError("AUDIO input is None")

    if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
        wave = audio["waveform"]
        sr = int(audio["sample_rate"])
        if torch is not None and isinstance(wave, torch.Tensor):
            if wave.dim() == 3:
                wave = wave[0, 0].detach().cpu().numpy()
            elif wave.dim() == 1:
                wave = wave.detach().cpu().numpy()
            else:
                wave = wave.flatten().detach().cpu().numpy()
        elif isinstance(wave, np.ndarray):
            if wave.ndim == 3:
                wave = wave[0, 0]
            elif wave.ndim == 2:
                wave = wave[0]
        else:
            raise ValueError("AUDIO waveform must be torch.Tensor or numpy.ndarray")
        return wave.astype(np.float32), sr

    if isinstance(audio, tuple) and len(audio) == 2:
        wave, sr = audio
        if torch is not None and isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()
        return np.asarray(wave, dtype=np.float32), int(sr)

    raise ValueError("AUDIO input must be ComfyUI dict or (wave, sr)")


def save_temp_wav(wave_sr: Tuple[np.ndarray, int]) -> str:
    """
    Save (wave, sr) to a temporary mono WAV file and return the path.
    Wave is expected in float32 [-1, 1] range or int16.
    Caller should delete the path when done (see cleanup_temp_paths).
    """
    wave, sr = wave_sr
    if wave is None:
        raise ValueError("wave is None")
    wave = np.asarray(wave)
    if wave.ndim > 1:
        wave = wave.reshape(-1)
    if wave.dtype != np.float32:
        if wave.dtype == np.int16:
            wave = (wave.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        else:
            wave = wave.astype(np.float32)

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="indextts25_")
    os.close(fd)
    try:
        sf.write(path, wave, int(sr))
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


def cleanup_temp_paths(*paths: Optional[str]) -> None:
    """Best-effort delete of temporary WAV paths."""
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            warnings.warn(f"[IndexTTS-2.5] Failed to remove temp file {path}: {e}")


def normalize_emotion_vector(values) -> List[float]:
    """
    Normalize an 8-D emotion vector for IndexTTS-2.5.

    - Clamp each dim to >= 0
    - Soft-cap sum to 1.5 (UI guidance)
    - Then match upstream semantics: sum <= 0.8
    - All zeros -> Neutral=1.0 (then scaled to 0.8)
    """
    vec = [float(max(0.0, float(x))) for x in values]
    if len(vec) != 8:
        raise ValueError(f"emotion vector must have 8 dims, got {len(vec)}")

    s = float(sum(vec))
    if s <= 0.0:
        vec = [0.0] * 7 + [1.0]
        s = 1.0
    elif s > 1.5:
        scale = 1.5 / s
        vec = [v * scale for v in vec]
        s = 1.5
        print(f"[IndexTTS-2.5] Emotion vector sum capped to 1.5 -> {[round(v, 4) for v in vec]}")

    if s > 0.8:
        scale = 0.8 / s
        vec = [v * scale for v in vec]
        print(f"[IndexTTS-2.5] Emotion vector scaled to sum<=0.8 -> {[round(v, 4) for v in vec]}")

    return vec
