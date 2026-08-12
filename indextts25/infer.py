import json
import numpy as np
from typing import Optional, Tuple, List

from .model_loader import IndexTTS25Loader
from .utils import save_temp_wav


class IndexTTS25Engine:
    """
    Thin wrapper calling vendored indextts.infer_v2_5.IndexTTS2.infer.
    Converts ComfyUI audio to temp WAVs and returns float32 mono waveform.
    """

    def __init__(self, loader: Optional[IndexTTS25Loader] = None):
        self.loader = loader or IndexTTS25Loader()

    def generate(
        self,
        text: str,
        reference_audio: Optional[Tuple[np.ndarray, int]] = None,
        lang: str = "ZH",
        duration_factor: float = 1.0,
        style_text: Optional[str] = None,
        style_audio: Optional[Tuple[np.ndarray, int]] = None,
        mode: str = "Auto",
        duration_sec: Optional[float] = None,
        token_count: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_tokens_per_sentence: int = 120,
        emotion_control_method: Optional[str] = None,
        emo_text: Optional[str] = None,
        emo_ref_audio: Optional[Tuple[np.ndarray, int]] = None,
        emo_vector: Optional[List[float]] = None,
        emo_weight: float = 0.8,
        seed: int = 0,
        use_qwen: bool = False,
        verbose: bool = False,
        return_subtitles: bool = True,
    ) -> Tuple[int, np.ndarray, Optional[str]]:
        # If emotion-text is requested but loader has no Qwen, rebuild with Qwen enabled
        need_qwen = bool(use_qwen) or (
            emo_text is not None and str(emo_text).strip() != "" and emo_ref_audio is None and not emo_vector
        )
        if need_qwen and not self.loader.use_qwen_emo:
            self.loader.unload_tts()
            self.loader = IndexTTS25Loader(
                models_root=self.loader._models_root,
                device=str(self.loader.device),
                use_qwen_emo=True,
                use_bf16=self.loader._use_bf16,
            )

        tts = self.loader.get_tts()

        if reference_audio is None:
            raise ValueError("reference_audio is required for IndexTTS-2.5")
        spk_wav_path = save_temp_wav(reference_audio)

        emo_wav_path = None
        if emo_ref_audio is not None:
            emo_wav_path = save_temp_wav(emo_ref_audio)
        elif style_audio is not None:
            emo_wav_path = save_temp_wav(style_audio)

        _max_mel_tokens = int(max_mel_tokens) if max_mel_tokens else 1500
        gen_kwargs = dict(
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(_max_mel_tokens),
        )

        use_emo_text = False
        _emo_text = None
        if emo_wav_path is None and (emo_vector is None or len(emo_vector) == 0):
            if (use_qwen or need_qwen) and emo_text and str(emo_text).strip():
                use_emo_text = True
                _emo_text = emo_text
            elif use_qwen and not (emo_text and str(emo_text).strip()):
                # use text itself for emotion when use_qwen without explicit emo_text
                use_emo_text = True
                _emo_text = None

        lang = (lang or "ZH").upper().strip()
        duration_factor = float(max(0.5, min(2.0, float(duration_factor))))

        result = tts.infer(
            spk_audio_prompt=spk_wav_path,
            text=text,
            output_path=None,
            lang=lang,
            emo_audio_prompt=emo_wav_path,
            emo_alpha=float(emo_weight),
            emo_vector=emo_vector if (emo_wav_path is None and emo_vector) else None,
            use_emo_text=bool(use_emo_text),
            emo_text=_emo_text,
            use_random=False,
            interval_silence=200,
            verbose=bool(verbose),
            max_text_tokens_per_segment=int(max_tokens_per_sentence) if max_tokens_per_sentence else 120,
            duration_factor=duration_factor,
            **gen_kwargs,
        )

        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(f"Unexpected return from IndexTTS2.5.infer: {type(result)}")

        sr, wav = result
        wav = np.asarray(wav)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = (wav.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

        subtitle = None
        if return_subtitles:
            duration = len(wav) / float(sr)
            subtitle = json.dumps([
                {"id": "Narrator", "字幕": text, "start": 0.0, "end": round(duration, 2)}
            ], ensure_ascii=False)

        return int(sr), wav, subtitle
