import urllib.request
import json

BASE_URL = "https://cab-county-deferred-optional.trycloudflare.com"

# MiniMax-H3 公式プロンプティングガイド準拠:
# 1. タイムコード別ショットリスト [0-2s] [2-4s] [4-5s]
# 2. カメラワークとレンズ演出 (Cinematic push-in, shallow depth of field)
# 3. キャラクター造形の厳密な固定 (Preserve features)
# 4. ネイティブステレオ音響の演出 (Audio direction: room tone, rain, soft sigh)
# 5. 明確な除外制約 (State what not to show)
KENMOU_PROMPT_FAL_SPEC = (
    "A 5-second 16:9 cinematic anime short of Kenmou-kun `( ´ん\\` )`. "
    "Preserve character identity: completely round, bald pale white head, droopy resigned eyes with heavy dark bags, "
    "sparse faint stubble along the jawline, slouching posture in a worn gray long-sleeve shirt. "
    "Setting: a dimly lit Japanese 4.5-tatami apartment room at night. An old CRT monitor on a low table, "
    "an open can of cheap beer, messy futon in the corner, and heavy rain tapping against the window pane. "
    "[0 to 2 seconds] Medium-close shot. Kenmou-kun sits hunched over the low table, staring lifelessly at the 2ch poverty board text on the screen. "
    "Soft green phosphor reflections illuminate his pale face. "
    "[2 to 4 seconds] Smooth dramatic push-in to a close-up of his face. The screen suddenly flashes with warm golden light showing: 'A100 GPU Video Generation Successful!'. "
    "His tired eyes widen subtly and fill with glistening tears. His lip quivers as he breathes out in quiet disbelief. "
    "[4 to 5 seconds] A tear rolls down his pale cheek. He lets out a faint, trembling whisper. "
    "Audio: gentle rain hitting the window, low electrical 60Hz hum of the old CRT monitor, clothing friction as he shifts, "
    "and a quiet, emotional sigh with soft piano chords rising delicately. "
    "Visual style: retro 1990s Japanese cel anime look, 35mm film grain, subdued desaturated color palette with warm monitor light, 24fps. "
    "Do not show handsome bishonen features, sharp hair, glossy modern 3D CGI, jump scares, or bright cartoon expressions."
)

prompt_data = {
    # 1. UNET Loader (MiniMax-H3)
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default"
        }
    },
    # 2. CLIP Loader (Qwen3-VL 32B)
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax"
        }
    },
    # 3. Video VAE
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"
        }
    },
    # 4. Audio VAE
    "4": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
        }
    },
    # 5. Sigma Shift for MiniMax-H3
    "5": {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": ["1", 0],
            "shift_video": 12.0,
            "shift_audio": 3.0
        }
    },
    # 6. MiniMaxH3 Conditioning
    "6": {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": KENMOU_PROMPT_FAL_SPEC,
            "width": 1344,
            "height": 768,
            "length": 124
        }
    },
    # 7. Negative Conditioning (Prompt Guide Rule 4: State what you do not want)
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "handsome bishonen, spiky anime hair, western 3D cartoon, glossy modern CGI render, saturated cheerful colors, blurry, distorted anatomy, fast jump cuts, watermark",
            "clip": ["2", 0]
        }
    },
    # 8. KSampler
    "8": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["5", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["6", 1],
            "seed": 20260924,
            "steps": 25,
            "cfg": 6.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    },
    # 9. VAEDecode (Latent -> Video Frames)
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],
            "vae": ["3", 0]
        }
    },
    # 10. CreateVideo (Frames -> Video Stream at 24fps)
    "10": {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["9", 0],
            "fps": 24.0
        }
    },
    # 11. SaveVideo (Output strictly to MP4)
    "11": {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["10", 0],
            "filename_prefix": "kenmou_official_h3_spec",
            "format": "mp4"
        }
    }
}

def queue():
    data = json.dumps({"prompt": prompt_data, "client_id": "agent_kenmou"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            res_json = json.loads(res.read().decode())
            prompt_id = res_json.get("prompt_id")
            print(f"[SUCCESS] Fal-spec MiniMax-H3 MP4 job queued on Colab A100! Prompt ID: {prompt_id}")
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    queue()
