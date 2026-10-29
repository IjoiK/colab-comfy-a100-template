"""
ComfyUI Client & Agent Bridge for Google Colab + MiniMax-H3
Allows AI agents (or CLI scripts) to trigger video generation via Cloudflare Tunnel URL.
"""

import json
import time
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any


class ComfyUIColabClient:
    def __init__(self, base_url: str):
        """
        :param base_url: e.g. "https://xxxx.trycloudflare.com"
        """
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        """Checks if the ComfyUI endpoint is alive and responsive."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            print(f"[Error] Failed to connect to {self.base_url}: {e}")
            return False

    def queue_prompt(self, workflow_prompt: Dict[str, Any], client_id: str = "agent-runner") -> Optional[str]:
        """
        Queues a ComfyUI prompt dictionary.
        Returns the prompt_id if successful.
        """
        data = json.dumps({"prompt": workflow_prompt, "client_id": client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                result = json.loads(res.read().decode("utf-8"))
                return result.get("prompt_id")
        except urllib.error.HTTPError as e:
            print(f"[HTTP Error {e.code}] Failed to queue prompt: {e.read().decode('utf-8', errors='replace')}")
            return None
        except Exception as e:
            print(f"[Error] Failed to queue prompt: {e}")
            return None

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Fetches history for a given prompt_id."""
        url = f"{self.base_url}/history/{prompt_id}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data.get(prompt_id)
        except Exception:
            return None

    def wait_for_completion(self, prompt_id: str, poll_interval: int = 5, timeout: int = 1800) -> Optional[Dict[str, Any]]:
        """Polls until the video generation is completed and returns outputs."""
        print(f"⏳ Waiting for generation job {prompt_id} to finish...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history and "outputs" in history:
                print("🎉 Generation completed successfully!")
                return history["outputs"]
            time.sleep(poll_interval)
        print("❌ Generation timed out.")
        return None

    def get_output_file_url(self, filename: str, subfolder: str = "", folder_type: str = "output") -> str:
        """Returns the public download URL for a generated video file."""
        query = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        })
        return f"{self.base_url}/view?{query}"

    def download_output(self, filename: str, dest_path: str, subfolder: str = "", folder_type: str = "output") -> str:
        """Downloads a generated video/image file from ComfyUI to a local path."""
        url = self.get_output_file_url(filename, subfolder=subfolder, folder_type=folder_type)
        urllib.request.urlretrieve(url, dest_path)
        print(f"💾 Saved output to: {dest_path}")
        return dest_path


def create_minimax_t2v_prompt(
    positive_prompt: str,
    negative_prompt: str = "blurry, distorted anatomy, glitch, low quality, fast jump cuts, watermark",
    width: int = 1344,
    height: int = 768,
    frames: int = 124,
    steps: int = 25,
    cfg: float = 6.0,
    seed: int = 42,
    use_turbo_lora: bool = False,
    filename_prefix: str = "minimax_h3_output"
) -> Dict[str, Any]:
    """
    Constructs an official MiniMax-H3 API workflow structure for ComfyUI.
    Supports optional 8-step Turbo LoRA (`use_turbo_lora=True`, recommended steps=8).
    """
    if use_turbo_lora and steps == 25:
        steps = 8

    workflow: Dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_video_vae_fp16.safetensors"
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
            }
        }
    }

    model_source = ["1", 0]
    if use_turbo_lora:
        workflow["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
                "strength_model": 1.0
            }
        }
        model_source = ["12", 0]

    workflow.update({
        "5": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": model_source,
                "shift_video": 12.0,
                "shift_audio": 3.0
            }
        },
        "6": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": positive_prompt,
                "width": width,
                "height": height,
                "length": frames
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["2", 0]
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["6", 1],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0]
            }
        },
        "10": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["9", 0],
                "fps": 24.0
            }
        },
        "11": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["10", 0],
                "filename_prefix": filename_prefix,
                "format": "mp4"
            }
        }
    })
    return workflow
