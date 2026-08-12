"""
@title: ComfyUI-Index-TTS25
@author: ComfyUI-Index-TTS25
@description: ComfyUI nodes for IndexTTS-2.5 zero-shot multilingual TTS
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .audio_enhancement import AudioCleanupNode
from .timbre_audio_loader import TimbreAudioLoader
from .novel_text_parser import NovelTextStructureNode
from .index_tts25_mode_nodes import (
    IndexTTS25BaseNode,
    IndexTTS25EmotionAudioNode,
    IndexTTS25EmotionVectorNode,
    IndexTTS25EmotionTextNode,
)
from .index_tts25_cache_nodes import IndexTTS25CacheControlNode
from .index_tts25_pro import IndexTTS25ProNode

NODE_CLASS_MAPPINGS = {
    "AudioCleanupNode": AudioCleanupNode,
    "TimbreAudioLoader": TimbreAudioLoader,
    "NovelTextStructureNode": NovelTextStructureNode,
    "IndexTTS25BaseNode": IndexTTS25BaseNode,
    "IndexTTS25EmotionAudioNode": IndexTTS25EmotionAudioNode,
    "IndexTTS25EmotionVectorNode": IndexTTS25EmotionVectorNode,
    "IndexTTS25EmotionTextNode": IndexTTS25EmotionTextNode,
    "IndexTTS25CacheControlNode": IndexTTS25CacheControlNode,
    "IndexTTS25ProNode": IndexTTS25ProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioCleanupNode": "Audio Cleaner",
    "TimbreAudioLoader": "Timbre Audio Loader",
    "NovelTextStructureNode": "Novel Text Structure",
    "IndexTTS25BaseNode": "Index TTS 2.5 - Base",
    "IndexTTS25EmotionAudioNode": "Index TTS 2.5 - Emotion Audio (声纹+情绪节奏)",
    "IndexTTS25EmotionVectorNode": "Index TTS 2.5 - Emotion Vector",
    "IndexTTS25EmotionTextNode": "Index TTS 2.5 - Emotion Text",
    "IndexTTS25CacheControlNode": "Index TTS 2.5 - Cache Control",
    "IndexTTS25ProNode": "Index TTS 2.5 Pro (Multi-Character)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
