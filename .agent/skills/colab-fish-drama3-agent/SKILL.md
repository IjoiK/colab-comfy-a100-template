---
name: colab-fish-drama3-agent
description: Automates or interacts with Google Colab A100 instances running Fish Audio Drama 3 (35B-A3B) and Fish Speech API/WebUI for multi-speaker TTS, emotional speech synthesis, and voice cloning.
---

# Colab Fish Audio Drama 3 Agent Skill

This skill enables AI agents to synthesize single-speaker or multi-speaker emotional speech and clone voices using a Google Colab A100 instance running **Fish Audio Drama 3** (`fishaudio/drama-3`) via **Fish Speech** (`fishaudio/fish-speech`).

## 1. Drama 3 Prompting & Tag Guidelines

- **Speaker Tags**: Use `<|speaker:0|>` through `<|speaker:15|>` to voice up to 16 distinct characters in a single scene:
  ```text
  <|speaker:0|> [sighs] Мы наконец-то добрались. <|speaker:1|> [laughs] Я же говорил, что успеем!
  ```
- **Inline Paralinguistic & Emotion Tags**: Place tags directly where the expression should occur:
  `[sighs]`, `[laughs]`, `[whisper]`, `[angry]`, `[crying]`, `[excited]`, `[pause]`.

---

## 2. Execution Patterns

### Pattern 1: Direct API Call via Pinggy / Cloudflare Tunnel

When the user provides a live tunnel URL from Step 4 of `fish-audio-drama-3/colab_fish_drama3_a100.ipynb`:

```python
import sys
sys.path.insert(0, "fish-audio-drama-3")
from client import FishSpeechColabClient

client = FishSpeechColabClient(base_url="https://your-tunnel.a.free.pinggy.link")
assert client.check_health(), "Fish Speech server is not reachable!"

client.generate_speech(
    text="<|speaker:0|> [sighs] Привет! Это проверка модели Fish Audio Drama 3 на A100.",
    output_path="drama3_output.wav",
    audio_format="wav"
)
```

### Pattern 2: Browser Automation

1. Open the notebook in Google Colab:
   `https://colab.research.google.com/github/IjoiK/colab-comfy-a100-template/blob/main/fish-audio-drama-3/colab_fish_drama3_g4_96gb.ipynb`
2. Select runtime **G4 GPU (96 GB)** and run all cells (`Ctrl+F9`).
3. Extract the Pinggy or Cloudflare tunnel URL from Step 4 and use `fish-audio-drama-3/client.py` to generate audio.
