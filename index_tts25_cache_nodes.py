from typing import Dict, Any


class IndexTTS25CacheControlNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keep_models_cached": ("BOOLEAN", {"default": False}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("cache_control",)
    FUNCTION = "build"
    CATEGORY = "audio"

    def build(self, keep_models_cached: bool = False) -> Dict[str, Any]:
        ctrl = {
            "keep_cached": bool(keep_models_cached),
        }
        return (ctrl,)
