import os
import sys
import gc
import torch
from typing import List, Optional, Dict, Any


class IndexTTS25Loader:
    """
    Lightweight model manager for IndexTTS-2.5.
    - Resolves model root: <ComfyUI>/models/IndexTTS-2.5
    - Auto-downloads missing main weights via HuggingFace / ModelScope
    - Validates required files
    - Lazy loads vendored indextts.infer_v2_5.IndexTTS2
    """

    DEFAULT_DIRNAME = "IndexTTS-2.5"
    MODEL_REPO = "IndexTeam/IndexTTS-2.5"
    REQUIRED_FILES = (
        "config.yaml",
        "feat1.pt",
        "feat2.pt",
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "wav2vec2bert_stats.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    )
    # IndexTTS-2.5 uses multilingual tiktoken; bpe.model/pinyin.vocab are IndexTTS-2 leftovers.
    RECOMMENDED_FILES = ()
    QWEN_EMO_DIRNAME = "qwen0.6bemo4-merge"

    def __init__(
        self,
        models_root: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        use_qwen_emo: bool = False,
        use_bf16: Optional[bool] = None,
        auto_download: Optional[bool] = None,
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
        if auto_download is None:
            auto_download = os.environ.get("INDEXTTS_NO_AUTO_DOWNLOAD", "").strip().lower() not in (
                "1", "true", "yes", "on",
            )
        self._auto_download = bool(auto_download)
        self._cache: Dict[str, Any] = {}

        self._vendor_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        self._vendor_pkg_root = os.path.join(self._vendor_root, "indextts")
        self._ensure_vendor_on_path()

    def _ensure_vendor_on_path(self) -> None:
        if not os.path.isdir(self._vendor_root):
            return
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

    def _missing_files(self, filenames) -> List[str]:
        return [
            f for f in filenames
            if not os.path.exists(os.path.join(self._model_dir, f))
        ]

    def _qwen_emo_ready(self) -> bool:
        qwen_dir = os.path.join(self._model_dir, self.QWEN_EMO_DIRNAME)
        if not os.path.isdir(qwen_dir):
            return False
        try:
            return any(os.scandir(qwen_dir))
        except OSError:
            return False

    def _download_main_model(self, reason: str) -> None:
        """Download IndexTeam/IndexTTS-2.5 via HuggingFace Hub or ModelScope."""
        self._ensure_vendor_on_path()
        os.makedirs(self._model_dir, exist_ok=True)
        print(f"[IndexTTS-2.5] {reason}")
        print(f"[IndexTTS-2.5] Auto-downloading {self.MODEL_REPO} -> {self._model_dir}")
        print("[IndexTTS-2.5] Source: HuggingFace / ModelScope (network auto-detect)")
        try:
            from indextts.utils.model_download import snapshot_download
            snapshot_download(self.MODEL_REPO, local_dir=self._model_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to auto-download {self.MODEL_REPO} into {self._model_dir}: {e}. "
                "You can also run TTS25_download.py manually, or set HF_ENDPOINT="
                "https://hf-mirror.com / use ModelScope. "
                "Disable auto-download with INDEXTTS_NO_AUTO_DOWNLOAD=1."
            ) from e

    def ensure_main_model(self) -> None:
        """
        Ensure required main weights exist. If missing and auto_download is enabled,
        pull IndexTeam/IndexTTS-2.5 with the vendored HF/ModelScope helper.
        """
        missing = self._missing_files(self.REQUIRED_FILES)
        need_qwen = self._use_qwen_emo and not self._qwen_emo_ready()

        if missing or need_qwen:
            if not self._auto_download:
                parts = []
                if missing:
                    parts.append(f"missing files: {', '.join(missing)}")
                if need_qwen:
                    parts.append(f"missing {self.QWEN_EMO_DIRNAME}/ (Emotion Text)")
                raise FileNotFoundError(
                    f"IndexTTS-2.5 incomplete in {self._model_dir} ({'; '.join(parts)}). "
                    "Auto-download is disabled (INDEXTTS_NO_AUTO_DOWNLOAD). "
                    f"Download from {self.MODEL_REPO} (see TTS25_download.py)."
                )
            reason = []
            if missing:
                reason.append(f"missing required files: {', '.join(missing)}")
            if need_qwen:
                reason.append(f"missing {self.QWEN_EMO_DIRNAME}/ for Emotion Text")
            self._download_main_model("; ".join(reason))

        still_missing = self._missing_files(self.REQUIRED_FILES)
        if still_missing:
            raise FileNotFoundError(
                f"IndexTTS-2.5 still missing after download in {self._model_dir}: "
                f"{', '.join(still_missing)}. Check network / HF_ENDPOINT / ModelScope access."
            )
        if self._use_qwen_emo and not self._qwen_emo_ready():
            raise FileNotFoundError(
                f"Emotion Text requires {self.QWEN_EMO_DIRNAME}/ under {self._model_dir}, "
                "but it is still missing after download."
            )

    def validate(self) -> None:
        missing = self._missing_files(self.REQUIRED_FILES)
        if missing:
            raise FileNotFoundError(
                f"IndexTTS-2.5 missing files in {self._model_dir}: {', '.join(missing)}. "
                f"Download from {self.MODEL_REPO} (see TTS25_download.py)."
            )
        for f in self.RECOMMENDED_FILES:
            if not os.path.exists(os.path.join(self._model_dir, f)):
                print(f"[IndexTTS-2.5] Warning: recommended file missing: {f}")

    def get_tts(self):
        """
        Return a cached instance of indextts.infer_v2_5.IndexTTS2.
        """
        cache_key = f"tts_qwen{int(self._use_qwen_emo)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Ensure our vendor is first on path (may have been displaced by TTS2)
        self._ensure_vendor_on_path()
        self.ensure_main_model()
        self.validate()

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
