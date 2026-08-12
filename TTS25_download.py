#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS-2.5 Model Download Script
下载 IndexTeam/IndexTTS-2.5 到 ComfyUI/models/IndexTTS-2.5
辅助模型（w2v-bert / campplus / bigvgan）可在首次推理时由官方 ensure_models_available 自动下载到 hf_cache/
"""

import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


class ModelDownloader:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        # custom_nodes/ComfyUI-Index-TTS -> ComfyUI/models/IndexTTS-2.5
        self.models_dir = self.script_dir.parent.parent / "models" / "IndexTTS-2.5"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.endpoint_official = "https://huggingface.co"
        self.endpoint_mirror = "https://hf-mirror.com"
        self.current_endpoint = self.endpoint_official
        self.hf_home = self.models_dir / "hf_cache"
        os.environ.setdefault("HF_HOME", str(self.hf_home))

    def ask_mirror_preference(self):
        print("检测到您可能在中国大陆地区访问，是否使用国内镜像加速下载？")
        print("1. 使用官方地址 (huggingface.co)")
        print("2. 使用国内镜像 (hf-mirror.com) - 推荐")

        while True:
            choice = input("请选择 (1/2，默认为2): ").strip()
            if choice == "1":
                self.current_endpoint = self.endpoint_official
                print("已选择官方地址")
                break
            elif choice == "2" or choice == "":
                self.current_endpoint = self.endpoint_mirror
                print("已选择国内镜像")
                break
            else:
                print("请输入1或2")

        os.environ["HF_ENDPOINT"] = self.current_endpoint
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    def _snapshot(self, repo_id: str, allow_patterns: List[str], local_dir: Path):
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision="main",
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    def _download_file(self, repo_id: str, filename: str, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision="main",
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            return True
        except EntryNotFoundError:
            print(f"✗ 远端未找到文件: {repo_id}:{filename}")
            return False

    def download_all(self):
        print(f"\n{'='*50}")
        print("开始下载 IndexTTS-2.5 模型...")
        print(f"{'='*50}")
        success = True

        print("\n[1/3] 下载基础模型 (IndexTeam/IndexTTS-2.5)...")
        base_files = [
            "bpe.model",
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "codec.pth",
            "wav2vec2bert_stats.pt",
            "pinyin.vocab",
        ]
        try:
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2.5",
                allow_patterns=base_files + ["*.yaml", "*.json", "*.txt", "*.model", "*.vocab"],
                local_dir=self.models_dir,
            )
            # Prefer full repo snapshot for completeness
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2.5",
                allow_patterns=["*"],
                local_dir=self.models_dir,
            )
            print("✓ IndexTTS-2.5 基础/完整仓库下载完成")
        except Exception as e:
            print(f"✗ IndexTTS-2.5 下载失败: {e}")
            success = False

        print("\n[2/3] 确认 qwen0.6bemo4-merge（情绪文本节点需要）...")
        qwen_dir = self.models_dir / "qwen0.6bemo4-merge"
        if qwen_dir.exists():
            print("✓ qwen0.6bemo4-merge 已存在")
        else:
            try:
                self._snapshot(
                    repo_id="IndexTeam/IndexTTS-2.5",
                    allow_patterns=["qwen0.6bemo4-merge/*"],
                    local_dir=self.models_dir,
                )
                if qwen_dir.exists():
                    print("✓ qwen0.6bemo4-merge 下载完成")
                else:
                    print("⚠ qwen 目录未找到（若仅用 Base/Audio/Vector 可稍后补齐）")
            except Exception as e:
                print(f"⚠ qwen0.6bemo4-merge 下载失败: {e}")

        print("\n[3/3] 预下载辅助模型到 hf_cache（可选，首次推理也会自动下载）...")
        try:
            hf_cache = self.models_dir / "hf_cache"
            hf_cache.mkdir(parents=True, exist_ok=True)
            # campplus
            camp = hf_cache / "campplus_cn_common.bin"
            if not camp.exists():
                self._download_file("funasr/campplus", "campplus_cn_common.bin", camp)
            # w2v-bert
            w2v = hf_cache / "w2v-bert-2.0"
            need_w2v = (not w2v.exists()) or (not any(w2v.iterdir()))
            if need_w2v:
                self._snapshot("facebook/w2v-bert-2.0", ["*"], w2v)
            # bigvgan
            bv = hf_cache / "bigvgan"
            bv.mkdir(parents=True, exist_ok=True)
            for fname in ("config.json", "bigvgan_generator.pt"):
                target = bv / fname
                if not target.exists():
                    self._download_file(
                        "nvidia/bigvgan_v2_22khz_80band_256x",
                        fname,
                        target,
                    )
            print("✓ 辅助模型预下载完成（或已存在）")
        except Exception as e:
            print(f"⚠ 辅助模型预下载部分失败（首次推理仍可自动补齐）: {e}")

        return success

    def verify_downloads(self):
        print(f"\n{'='*50}")
        print("验证下载的文件...")
        print(f"{'='*50}")
        required_files = [
            "config.yaml",
            "gpt.pth",
            "s2mel.pth",
            "codec.pth",
            "feat1.pt",
            "feat2.pt",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
        ]
        missing = []
        for file_path in required_files:
            full_path = self.models_dir / file_path
            if not full_path.exists():
                missing.append(file_path)
            else:
                print(f"✓ {file_path}")
        if missing:
            print("\n缺少以下文件:")
            for file_path in missing:
                print(f"✗ {file_path}")
            return False
        print("\n✓ 所有必需文件都已下载完成!")
        return True

    def run(self):
        print("IndexTTS-2.5 模型下载脚本")
        print("=" * 50)
        print(f"模型将下载到: {self.models_dir.absolute()}")
        self.ask_mirror_preference()
        try:
            ok = self.download_all()
        except KeyboardInterrupt:
            print("\n用户中断下载")
            sys.exit(1)
        except Exception as e:
            print(f"下载过程中出错: {e}")
            ok = False

        print(f"\n{'='*50}")
        print("下载完成报告")
        print(f"{'='*50}")
        if self.verify_downloads() and ok:
            print(f"\n所有模型下载完成! 模型路径: {self.models_dir.absolute()}")
        else:
            print("\n部分文件可能缺失，请重新运行脚本或检查网络/镜像设置")


if __name__ == "__main__":
    try:
        ModelDownloader().run()
    except KeyboardInterrupt:
        print("\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"脚本运行出错: {e}")
        sys.exit(1)
