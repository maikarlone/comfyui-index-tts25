import torch
import numpy as np
from typing import Any, Tuple

from .indextts25 import IndexTTS25Loader, IndexTTS25Engine

_GLOBAL_LOADER = IndexTTS25Loader(use_qwen_emo=False)
_GLOBAL_ENGINE = IndexTTS25Engine(_GLOBAL_LOADER)

_LANG_CHOICES = ["ZH", "EN", "JA", "ES", "AR"]


class _IndexTTS25BaseMixin:
    @staticmethod
    def _process_audio_input(audio: Any) -> Tuple[np.ndarray, int]:
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            wave = audio["waveform"]
            sr = int(audio["sample_rate"])
            if isinstance(wave, torch.Tensor):
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
            return wave.astype(np.float32), sr
        elif isinstance(audio, tuple) and len(audio) == 2:
            wave, sr = audio
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            return wave.astype(np.float32), int(sr)
        else:
            raise ValueError("AUDIO input must be ComfyUI dict or (wave, sr)")

    @classmethod
    def _base_inputs(cls):
        return {
            "text": ("STRING", {"multiline": True, "default": "Hello, this is IndexTTS 2.5."}),
            "reference_audio": ("AUDIO", {"tooltip": "声纹/音色参考（决定说谁的声音）"}),
            "lang": (_LANG_CHOICES, {"default": "ZH"}),
            "duration_factor": ("FLOAT", {
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "step": 0.05,
                "tooltip": "语速：>1 更慢，<1 更快",
            }),
            "mode": (["Auto", "Duration", "Tokens"], {"default": "Auto"}),
        }

    @classmethod
    def _common_optional(cls):
        return {
            "do_sample_mode": (["off", "on"], {"default": "on"}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
            "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            "max_mel_tokens": ("INT", {"default": 1815, "min": 50, "max": 1815, "step": 5}),
            "max_tokens_per_sentence": ("INT", {"default": 120, "min": 0, "max": 600, "step": 5}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            "cache_control": ("DICT", {"default": None}),
        }

    def _do_generate(self, engine: IndexTTS25Engine, **kwargs):
        sr, wave, sub = engine.generate(**kwargs)
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio = {"waveform": wave_t, "sample_rate": int(sr)}
        return audio, kwargs.get("seed", 0), (sub or "")

    def _maybe_unload(self, cache_control):
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass


class IndexTTS25BaseNode(_IndexTTS25BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": cls._base_inputs(), "optional": cls._common_optional()}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(
        self,
        text,
        reference_audio,
        lang="ZH",
        duration_factor=1.0,
        mode="Auto",
        do_sample_mode="off",
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1815,
        max_tokens_per_sentence=120,
        seed=0,
        cache_control=None,
    ):
        ref = self._process_audio_input(reference_audio)
        out = self._do_generate(
            self.engine,
            text=text,
            reference_audio=ref,
            lang=lang,
            duration_factor=duration_factor,
            mode=mode,
            do_sample=(do_sample_mode == "on"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None,
            emo_ref_audio=None,
            emo_vector=None,
            emo_weight=0.8,
            seed=seed,
            return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionAudioNode(_IndexTTS25BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emo_ref_audio": ("AUDIO", {
                "tooltip": "情绪/节奏参考音频（对应 2.0 的 emo_ref_audio；不影响音色，音色由 reference_audio 决定）",
            }),
            "emotion_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.4,
                "step": 0.05,
                "tooltip": "情绪参考强度；配音常用 0.4~0.65，过高可能把参考音频的语速也带进来",
            }),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS25Loader(use_qwen_emo=False)
        self.engine = IndexTTS25Engine(self.loader)

    def generate(
        self,
        text,
        reference_audio,
        lang="ZH",
        duration_factor=1.0,
        mode="Auto",
        emo_ref_audio=None,
        emotion_weight=0.8,
        do_sample_mode="off",
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1815,
        max_tokens_per_sentence=120,
        seed=0,
        cache_control=None,
    ):
        ref = self._process_audio_input(reference_audio)
        emo_ref = self._process_audio_input(emo_ref_audio) if emo_ref_audio is not None else None
        out = self._do_generate(
            self.engine,
            text=text,
            reference_audio=ref,
            lang=lang,
            duration_factor=duration_factor,
            mode=mode,
            do_sample=(do_sample_mode == "on"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None,
            emo_ref_audio=emo_ref,
            emo_vector=None,
            emo_weight=float(emotion_weight),
            seed=seed,
            return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionVectorNode(_IndexTTS25BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "Happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Love": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
            "Neutral": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.4, "step": 0.01}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS25Loader(use_qwen_emo=False)
        self.engine = IndexTTS25Engine(self.loader)

    def generate(
        self,
        text,
        reference_audio,
        lang="ZH",
        duration_factor=1.0,
        mode="Auto",
        Happy=0.0,
        Angry=0.0,
        Sad=0.0,
        Fear=0.0,
        Hate=0.0,
        Love=0.0,
        Surprise=0.0,
        Neutral=0.0,
        do_sample_mode="off",
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1815,
        max_tokens_per_sentence=120,
        seed=0,
        cache_control=None,
    ):
        ref = self._process_audio_input(reference_audio)
        vec = [Happy, Angry, Sad, Fear, Hate, Love, Surprise, Neutral]
        s = float(sum(max(0.0, float(x)) for x in vec))
        emo_vec = ([float(max(0.0, float(x))) / s for x in vec] if s > 0 else [0.0] * 7 + [1.0])
        out = self._do_generate(
            self.engine,
            text=text,
            reference_audio=ref,
            lang=lang,
            duration_factor=duration_factor,
            mode=mode,
            do_sample=(do_sample_mode == "on"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None,
            emo_ref_audio=None,
            emo_vector=emo_vec,
            emo_weight=0.8,
            seed=seed,
            return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionTextNode(_IndexTTS25BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emotion_description": ("STRING", {"multiline": True, "default": ""}),
            "emotion_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.4, "step": 0.05}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS25Loader(use_qwen_emo=True)
        self.engine = IndexTTS25Engine(self.loader)

    def generate(
        self,
        text,
        reference_audio,
        lang="ZH",
        duration_factor=1.0,
        mode="Auto",
        emotion_description="",
        emotion_weight=0.6,
        do_sample_mode="off",
        temperature=0.8,
        top_p=0.9,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1815,
        max_tokens_per_sentence=120,
        seed=0,
        cache_control=None,
    ):
        ref = self._process_audio_input(reference_audio)
        emo_text = emotion_description.strip() if isinstance(emotion_description, str) else ""
        emo_text = emo_text if emo_text else None
        out = self._do_generate(
            self.engine,
            text=text,
            reference_audio=ref,
            lang=lang,
            duration_factor=duration_factor,
            mode=mode,
            do_sample=(do_sample_mode == "on"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=emo_text,
            emo_ref_audio=None,
            emo_vector=None,
            emo_weight=float(emotion_weight),
            use_qwen=True,
            verbose=True,
            seed=seed,
            return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out
