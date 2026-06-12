# 🤖 Social Media Automation Suite

> **AI-powered post automation for Facebook, Instagram, and LinkedIn** — generates captions with Groq LLM and images with local Stable Diffusion, then publishes them autonomously using Playwright browser automation.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Image Generation](#-image-generation)
- [Session Management](#-session-management)
- [Logging & Screenshots](#-logging--screenshots)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations](#-known-limitations)

---

## 🌟 Overview

This suite automates daily social media posting across three major platforms. You provide a **domain/topic**, an optional **custom prompt**, and an **image description** — the agent does the rest:

1. Generates a platform-optimized post caption using **Groq's LLaMA 3.3 70B** model
2. Generates a high-quality image using **Stable Diffusion 1.5** (runs locally on your GPU)
3. Opens the platform in a real browser (Playwright), logs in via saved session, uploads the image and posts — fully autonomously

---

## ✨ Features

| Feature | Details |
|---|---|
| **3 Platforms** | Facebook (`fb.py`), Instagram (`insta.py`), LinkedIn (`linkedin_automation.py`) |
| **AI Captions** | Groq API — LLaMA 3.3 70B, context-aware, emoji-rich posts |
| **Local AI Images** | Stable Diffusion 1.5 via Diffusers — no cloud cost, works on 4GB VRAM |
| **Persistent Sessions** | Browser login state saved — no repeated logins |
| **Multi-day Posting** | Schedule 1–N posts in sequence with configurable delay |
| **Draft Recovery** | Auto-detects and dismisses Facebook's "Save as draft?" dialog |
| **Retry Logic** | 3-strategy image upload with full exception logging |
| **Screenshot Audit** | Full screenshot trail at every step for debugging |
| **Log Files** | Per-platform rotating log files with timestamps |

---

## 📁 Project Structure

```
agenticai-main/
│
├── fb.py                    # Facebook automation script
├── insta.py                 # Instagram automation script
├── linkedin_automation.py   # LinkedIn automation script
├── local_image_generator.py # Shared Stable Diffusion image generator
│
├── .env                     # API keys (NOT committed to git)
├── .gitignore               # Excludes sessions, cache, logs
├── requirements.txt         # Python dependencies
│
├── fb_session/              # Saved Facebook browser session (auto-created)
├── insta_session/           # Saved Instagram browser session (auto-created)
├── linkedin_session/        # Saved LinkedIn browser session (auto-created)
│
├── image_cache/             # Generated images cached here (auto-created)
├── screenshots/
│   ├── facebook/            # Step-by-step screenshots from fb.py
│   ├── instagram/           # Step-by-step screenshots from insta.py
│   └── linkedin/            # Step-by-step screenshots from linkedin_automation.py
│
├── logs/
│   ├── facebook_automation.log
│   ├── instagram_automation.log
│   └── linkedin_automation.log
│
└── static/                  # Reserved for static assets
```

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  1. Caption Generation (Groq LLM)   │
│     Model: llama-3.3-70b-versatile  │
│     Platform-specific prompt tuning │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  2. Image Generation (Local SD 1.5) │
│     local_image_generator.py        │
│     Stable Diffusion 1.5 (fp16)     │
│     4GB VRAM optimized              │
│     25 inference steps              │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  3. Browser Automation (Playwright) │
│     Persistent session context      │
│     ① Open composer                 │
│     ② Upload image (3-strategy)     │
│     ③ Type caption after upload     │
│     ④ Click Post button             │
└─────────────────────────────────────┘
```

> **Why image before caption?**
> Facebook's UI transitions to a new state when the Photo/video button is clicked, which **wipes the textbox**. Uploading the image first, then typing the caption last, is the only reliable order.

---

## ✅ Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher |
| **CUDA GPU** | NVIDIA GPU with **≥ 4GB VRAM** (for Stable Diffusion) |
| **CUDA Toolkit** | Compatible with PyTorch (tested with CUDA 12.4) |
| **Groq API Key** | Free tier available at [console.groq.com](https://console.groq.com) |
| **Social Media Accounts** | Facebook, Instagram, LinkedIn accounts |

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agenticai.git
cd agenticai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support

> ⚠️ **Do this BEFORE installing requirements.txt** — the CUDA version must match your GPU.

```bash
# For CUDA 12.4 (most modern NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Playwright browsers

```bash
playwright install chromium
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required: Groq API key for caption generation
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: HuggingFace token (only needed if using HF API — not required for local SD)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Set to "true" to run browser without visible window
HEADLESS=false
```

> 🔒 The `.env` file is listed in `.gitignore` and will **never** be committed to git.

---

## 🚀 Usage

### Facebook

```bash
python fb.py
```

```
Domain Name: cooking
Custom prompt (press Enter to skip): Share tips about healthy meal prep
Do you want an image? (y/n): y
Image prompt: A vibrant flat-lay of colorful vegetables and spices on a wooden table
Number of days (default 1): 3
```

### Instagram

```bash
python insta.py
```

Same interactive prompts as above.

### LinkedIn

```bash
python linkedin_automation.py
```

Same interactive prompts — posts are tuned for professional/business tone.

---

### First Run — Login

On the **first run**, the browser will open a real Chromium window. You need to:

1. Log in manually to your account
2. The script detects login success automatically and continues

Your session is saved to `fb_session/`, `insta_session/`, or `linkedin_session/`. All subsequent runs will be **fully automatic** — no manual login needed.

---

## 🔍 How It Works

### Step-by-step flow (per post)

```
1. Navigate to platform home page
2. Dismiss any leftover dialogs (e.g. Facebook "Save as draft?")
3. Open the post composer:
   - Strategy 1: Click "What's on your mind" / "Start a post" button
   - Strategy 2: If textbox is already visible, use it directly
   - Strategy 3: Reload page and retry
4. Upload image (if provided):
   - Priority 1: Inject directly into existing file <input> element
   - Priority 2: Click Photo/video → wait for file input → inject
   - Priority 3: Intercept native file chooser dialog
5. Wait for image preview to appear in dialog
6. Type caption into textbox (AFTER image upload settles)
7. Click the blue "Post" / "Share" button
8. Confirm success, save screenshot
```

### Key Design Decisions

| Decision | Reason |
|---|---|
| **Upload image before typing caption** | Facebook wipes the textbox when switching to media mode |
| **Coordinate-based mouse click for Post button** | Most reliable method; bypasses click interception |
| **Scan Post buttons last-to-first** | The bottommost `div[aria-label='Post']` is always the composer button |
| **3 upload strategies with full logging** | Facebook's DOM changes between sessions; fallbacks ensure reliability |
| **Persistent browser context** | Saves login session — avoids repeated CAPTCHA / 2FA challenges |

---

## 🎨 Image Generation

Images are generated **locally** using **Stable Diffusion 1.5** via HuggingFace Diffusers.

**Model:** `runwayml/stable-diffusion-v1-5`  
**Format:** FP16 (float16) — optimized for VRAM efficiency  
**Inference steps:** 25 (fast + good quality balance)  
**VRAM optimizations applied:**
- `enable_model_cpu_offload()` — offloads unused layers to RAM
- `enable_attention_slicing()` — reduces peak VRAM usage

Generated images are cached in `image_cache/` with timestamped filenames. The cache is automatically cleaned — only the 5 most recent images are kept.

> 💡 The model is downloaded from HuggingFace on the **first run** (~4GB). Subsequent runs load from cache in seconds.

---

## 💾 Session Management

Each platform saves its Playwright browser session to a dedicated folder:

| Platform | Session folder |
|---|---|
| Facebook | `./fb_session/` |
| Instagram | `./insta_session/` |
| LinkedIn | `./linkedin_session/` |

These folders contain cookies, local storage, and browser state. They are excluded from git (see `.gitignore`).

**To reset a session** (force re-login):
```bash
# Example: reset Facebook session
Remove-Item -Recurse -Force fb_session
```

---

## 📸 Logging & Screenshots

### Log Files

| File | Contents |
|---|---|
| `logs/facebook_automation.log` | All Facebook run logs with timestamps |
| `logs/instagram_automation.log` | All Instagram run logs |
| `logs/linkedin_automation.log` | All LinkedIn run logs |

### Screenshots

Each automation run saves screenshots at every key step inside `screenshots/<platform>/`:

| Screenshot | When taken |
|---|---|
| `00_page_before_compose.png` | Before opening composer |
| `01_page_after_reload.png` | After page reload (if needed) |
| `before_image_upload.png` | Before clicking Photo/video |
| `after_image_upload.png` | After image is confirmed in dialog |
| `after_text_fill.png` | After caption is typed |
| `before_post_click.png` | Immediately before clicking Post |
| `post_success.png` | After successful publish |
| `editor_button_not_found.png` | ❌ If composer couldn't be opened |
| `post_click_failed.png` | ❌ If Post button click failed |
| `upload_retry_N_error.png` | ❌ If upload retry N failed |

---

## 🔧 Troubleshooting

### ❌ CUDA not available / CPU-only PyTorch
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### ❌ "Could not open composer"
- Check `screenshots/facebook/editor_button_not_found.png`
- The script will dump all visible buttons in the log — look for `[DIAGNOSTIC]` lines
- Facebook may have updated its UI — check if "What's on your mind" text changed

### ❌ "All upload retries failed"
- Check `screenshots/facebook/before_image_upload.png` — is the composer open?
- Check `screenshots/facebook/upload_retry_1_error.png` for what went wrong
- Ensure the image file path exists and is a valid PNG/JPG

### ❌ "Save as draft?" dialog blocking automation
- The script now auto-dismisses this dialog
- If it persists, manually delete the browser session folder and log in fresh

### ❌ Playwright timeout errors
- Increase timeouts in the script for slow connections
- Set `HEADLESS=false` in `.env` to watch the browser in real time

### ❌ Model download fails on first run
- Ensure you have a stable internet connection (~4GB download)
- If HuggingFace is blocked, set a mirror: `HF_ENDPOINT=https://hf-mirror.com`

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| **GPU required** | Stable Diffusion requires an NVIDIA GPU with ≥ 4GB VRAM and CUDA. CPU-only mode is not supported |
| **Platform UI changes** | Facebook/Instagram/LinkedIn may update their DOM — selectors may need updating |
| **Rate limits** | Posting too frequently may trigger platform anti-spam detection |
| **2FA / CAPTCHA** | Must be handled manually on first login; subsequent runs are session-based |
| **Session expiry** | Sessions last weeks/months but may expire — delete session folder to force re-login |
| **Single account** | Each script supports one account per run (one session folder) |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `playwright` | Browser automation (Chromium) |
| `groq` | LLaMA 3.3 70B API for caption generation |
| `diffusers` | Stable Diffusion 1.5 image generation pipeline |
| `torch` | PyTorch — GPU compute backend for SD |
| `transformers` | HuggingFace model loading |
| `accelerate` | Model acceleration / CPU offload support |
| `pillow` | Image processing and placeholder generation |
| `python-dotenv` | Load `.env` configuration |
| `huggingface_hub` | HuggingFace model download client |

---

## 📄 License

This project is for educational and personal use. Ensure your usage complies with the Terms of Service of Facebook, Instagram, and LinkedIn.

---

*Built with ❤️ using Playwright, Groq, and Stable Diffusion.*
