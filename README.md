# ComfyUI-Index-TTS25

ComfyUI custom nodes for **[IndexTTS-2.5](https://github.com/index-tts/index-tts)** — an industrial-grade, multilingual zero-shot text-to-speech system with emotion control and speaking-speed control.

This repository contains **only IndexTTS-2.5** integration (IndexTTS 1.x / 2.0 are not included).

Upstream model & paper:

- Code: [index-tts/index-tts](https://github.com/index-tts/index-tts)
- Weights: [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) · [ModelScope](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- Paper: [IndexTTS 2.5 Technical Report](https://arxiv.org/abs/2601.03888)

## Features

- Zero-shot voice cloning from a short reference clip
- Languages: **Chinese / English / Japanese / Spanish / Arabic** (`lang`)
- Speaking speed via **`duration_factor`** (0.5–2.0; `>1` slower, `<1` faster)
- Emotion control:
  - reference emotion audio
  - 8-D emotion vector
  - emotion text (Qwen)
- Pronunciation hints in text (Chinese Pinyin / English CMU phonemes / Japanese Kana)
- Multi-character novel reading (Pro node)
- Optional cache control to free VRAM after each run

## Nodes

| Node | Description |
|------|-------------|
| **Index TTS 2.5 - Base** | Basic synthesis |
| **Index TTS 2.5 - Emotion Audio** | Emotion from a second reference audio |
| **Index TTS 2.5 - Emotion Vector** | Emotion from 8 sliders (happy/angry/sad/…) |
| **Index TTS 2.5 - Emotion Text** | Emotion from text description (loads Qwen) |
| **Index TTS 2.5 - Cache Control** | Keep or unload models after generation |
| **Index TTS 2.5 Pro (Multi-Character)** | Structured multi-speaker novel TTS |
| Timbre Audio Loader | Load reference wav from `TimbreModel/` |
| Novel Text Structure | Parse novel text into role-tagged segments |
| Audio Cleaner | Lightweight audio cleanup helper |

## Install

1. Clone into ComfyUI custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maikarlone/comfyui-index-tts25.git
```

2. Install Python dependencies (use the same Python / venv as ComfyUI):

```bash
cd comfyui-index-tts25
pip install -r requirements.txt
```

> **Note:** Official IndexTTS pins `transformers==4.52.1`. This plugin requires `transformers>=4.50`. Do **not** force `torch==2.8` if your ComfyUI environment already has a working CUDA torch build.

3. Download models (see next section).

4. Restart ComfyUI. Search for `Index TTS 2.5` in the node menu.

## Models

Place weights under:

```text
ComfyUI/models/IndexTTS-2.5/
```

### One-click download

From this plugin directory:

```bash
python TTS25_download.py
```

Or with Hugging Face / ModelScope CLI:

```bash
hf download IndexTeam/IndexTTS-2.5 --local-dir ComfyUI/models/IndexTTS-2.5

# or
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir ComfyUI/models/IndexTTS-2.5
```

### Required files (validated at load time)

| File | Role |
|------|------|
| `config.yaml` | Model config |
| `gpt.pth` | GPT / T2S weights |
| `s2mel.pth` | S2Mel weights |
| `codec.pth` | Semantic codec (2.5 native; **not** Amphion MaskGCT) |
| `feat1.pt` / `feat2.pt` | Speaker / emotion matrices |
| `wav2vec2bert_stats.pt` | Feature normalization stats |
| `multilingual_zh_ja_yue_char_del.tiktoken` | Multilingual tokenizer vocab |

### Recommended

- `bpe.model`, `pinyin.vocab`
- `qwen0.6bemo4-merge/` — required for **Emotion Text** node
- `glossary.yaml` — optional pronunciation glossary

### Auto-downloaded auxiliaries (`hf_cache/`)

On first run, IndexTTS may download into `IndexTTS-2.5/hf_cache/`:

- `w2v-bert-2.0/`
- `campplus_cn_common.bin`
- `bigvgan/` (`config.json`, `bigvgan_generator.pt`)

If Hugging Face is slow, set:

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### Example layout

```text
ComfyUI/models/IndexTTS-2.5/
├── config.yaml
├── gpt.pth
├── s2mel.pth
├── codec.pth
├── feat1.pt
├── feat2.pt
├── wav2vec2bert_stats.pt
├── multilingual_zh_ja_yue_char_del.tiktoken
├── bpe.model
├── pinyin.vocab
├── qwen0.6bemo4-merge/
└── hf_cache/
    ├── w2v-bert-2.0/
    ├── campplus_cn_common.bin
    └── bigvgan/
```

See also [`MODEL_PATHS.txt`](MODEL_PATHS.txt).

## Quick usage

1. Load a reference voice with **Load Audio** or **Timbre Audio Loader**.
2. Add **Index TTS 2.5 - Base**.
3. Set `text`, `lang` (e.g. `ZH` / `EN`), and optional `duration_factor`.
4. Connect reference audio → generate.

### Emotion examples

- **Emotion Audio**: connect a second clip that carries the desired emotion; tune `emotion_weight`.
- **Emotion Vector**: raise one or more of Happy / Angry / Sad / … (normalized internally).
- **Emotion Text**: describe emotion in words; keep `emotion_weight` around `0.6` for natural results.

### Speaking speed

- `duration_factor=1.0` — normal
- `duration_factor=1.2` — slower (~1.2× duration)
- `duration_factor=0.8` — faster

### Pronunciation control (in the text itself)

```text
他在银<行|XING2>里<行|HANG2>走了半天。

He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.

彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

Full lists: `pinyin.vocab` in the checkpoint folder; English CMU dictionary for phonemes.

### Multi-character (Pro)

Use structured tags:

```text
<Narrator>旁白内容。<Character1>角色台词。<Narrator>他又说。
```

Connect narrator + character reference audios to **Index TTS 2.5 Pro**.

## Project layout

```text
comfyui-index-tts25/
├── __init__.py                 # Node registration
├── index_tts25_mode_nodes.py   # Base / Emotion* nodes
├── index_tts25_pro.py          # Multi-character Pro
├── index_tts25_cache_nodes.py
├── indextts25/                 # Loader + Engine + vendored IndexTTS code
│   ├── model_loader.py
│   ├── infer.py
│   └── vendor/indextts/        # Synced from official index-tts
├── TTS25_download.py
├── requirements.txt
└── README.md
```

## Acknowledgements

- [IndexTTS Team / Bilibili](https://github.com/index-tts/index-tts) for IndexTTS-2.5
- Earlier ComfyUI IndexTTS ports that inspired the node UX

## Disclaimer

This project is for research, learning, and legitimate creative use only. **Do not** use it for illegal purposes, copyright infringement, fraud, or any activity that violates applicable laws. Users are solely responsible for how they use the software and generated audio.

## License

- This ComfyUI wrapper is released under the [MIT License](LICENSE).
- IndexTTS model weights and upstream / vendored `indextts` code remain under the licenses of the [IndexTTS authors](https://github.com/index-tts/index-tts).
