"""
IndexTTS-2.5 ComfyUI integration package.
Isolated from IndexTTS-2.0 (indextts2/) so both can coexist.
"""

from .model_loader import IndexTTS25Loader
from .infer import IndexTTS25Engine

__all__ = ["IndexTTS25Loader", "IndexTTS25Engine"]
