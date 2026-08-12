import os
import sys
import gc
import torch
from typing import Optional, Dict, Any


class IndexTTS25Loader:
    """
    Lightweight model manager for IndexTTS-2.5.
    - Resolves model root: <ComfyUI>/models/IndexTTS-2.5
    - Validates required files
    - Lazy loads vendored indextts.infer_v2_5.IndexTTS2
    """

    DEFAULT_DIRNAME = "IndexTTS-2.5"

    def __init__(
        self,
        models_root: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        use_qwen_emo: bool = False,
        use_bf16: Optional[bool] = None,
    ):
        self._models_root = models_root or self._default_models_root()
        self._model_dir = os.path.join(self._models_root, self.DEFAULT_DIRNAME)
        self._device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # IndexTTS-2.5 prefers bf16 on CUDA
        if use_bf16 is None:
            use_bf16 = self._device.type == "cuda"
        self._use_bf16 = bool(use_bf16) and self._device.type == "cuda"
        if dtype is None:
            dtype = "bf16" if self._use_bf16 else "fp32"
        self._dtype = self._resolve_dtype(dtype)
        self._use_qwen_emo = bool(use_qwen_emo)
        self._cache: Dict[str, Any] = {}

        self._vendor_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        self._vendor_pkg_root = os.path.join(self._vendor_root, "indextts")
        if os.path.isdir(self._vendor_root):
            try:
                sys.path.remove(self._vendor_root)
            except ValueError:
                pass
            sys.path.insert(0, self._vendor_root)

    @staticmethod
    def _default_models_root() -> str:
        # .../ComfyUI/custom_nodes/ComfyUI-Index-TTS/indextts25/model_loader.py
        # Go up 4 levels to .../ComfyUI/
        here = os.path.abspath(__file__)
        for _ in range(4):
            here = os.path.dirname(here)
        return os.path.join(here, "models")

    @staticmethod
    def _resolve_dtype(dtype: Optional[str]):
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def use_qwen_emo(self):
        return self._use_qwen_emo

    def validate(self) -> None:
        required = [
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "codec.pth",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
        ]
        # bpe.model / pinyin.vocab are commonly present in the HF snapshot; warn only if missing
        recommended = ["bpe.model", "pinyin.vocab"]
        missing = [f for f in required if not os.path.exists(os.path.join(self._model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"IndexTTS-2.5 missing files in {self._model_dir}: {', '.join(missing)}. "
                "Download from IndexTeam/IndexTTS-2.5 (see TTS25_download.py)."
            )
        for f in recommended:
            if not os.path.exists(os.path.join(self._model_dir, f)):
                print(f"[IndexTTS-2.5] Warning: recommended file missing: {f}")

    def get_tts(self):
        """
        Return a cached instance of indextts.infer_v2_5.IndexTTS2.
        """
        cache_key = f"tts_qwen{int(self._use_qwen_emo)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self.validate()

        # Ensure our vendor is first on path (may have been displaced by TTS2)
        if os.path.isdir(self._vendor_root):
            try:
                sys.path.remove(self._vendor_root)
            except ValueError:
                pass
            sys.path.insert(0, self._vendor_root)

        try:
            for k in list(sys.modules.keys()):
                if k == "indextts" or k.startswith("indextts."):
                    sys.modules.pop(k, None)
            from indextts.infer_v2_5 import IndexTTS2
        except Exception as e:
            try:
                import importlib.util
                infer_path = os.path.join(self._vendor_pkg_root, "infer_v2_5.py")
                spec = importlib.util.spec_from_file_location("indextts_infer_v2_5", infer_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"spec load failed for {infer_path}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["indextts_infer_v2_5"] = mod
                spec.loader.exec_module(mod)
                IndexTTS2 = getattr(mod, "IndexTTS2")
            except Exception as e2:
                raise ImportError(
                    f"Failed to import IndexTTS2 (2.5) from vendored source at {self._vendor_pkg_root}. "
                    f"Error: {e}. Fallback failed: {e2}. "
                    "Ensure dependencies (transformers, modelscope, huggingface_hub, torchaudio, "
                    "safetensors, omegaconf, fugashi, unidic-lite, g2p-en, cn2an, sentencepiece, etc.) are installed."
                )

        cfg_path = os.path.join(self._model_dir, "config.yaml")
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=self._model_dir,
            use_bf16=self._use_bf16,
            device=str(self._device),
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=self._use_qwen_emo,
        )
        self._cache[cache_key] = tts
        self._cache["tts"] = tts  # alias for unload
        return tts

    def unload_tts(self) -> None:
        """Best-effort unload of cached TTS instance and free GPU cache."""
        try:
            for k in list(self._cache.keys()):
                if k.startswith("tts"):
                    self._cache.pop(k, None)
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
