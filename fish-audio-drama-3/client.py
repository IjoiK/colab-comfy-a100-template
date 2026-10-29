"""
Fish Speech (Drama 3) Client & Agent Bridge for Google Colab A100
Allows AI agents or CLI scripts to trigger single/multi-speaker TTS and voice cloning
via Pinggy / Cloudflare Tunnel URL using standard library only (zero external dependencies).
"""

import base64
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class FishSpeechColabClient:
    def __init__(self, base_url: str):
        """
        :param base_url: e.g. "https://xxxx.a.free.pinggy.link" or "https://xxxx.trycloudflare.com"
        """
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/ui"):
            cleaned = cleaned[:-3]
        self.base_url = cleaned

    def check_health(self) -> bool:
        """Checks if the Fish Speech API server is alive and ready."""
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/health")
            with urllib.request.urlopen(req, timeout=10) as res:
                data = json.loads(res.read().decode("utf-8"))
                return res.status == 200 and data.get("status") == "ok"
        except Exception as e:
            print(f"[Error] Failed to connect to {self.base_url}: {e}")
            return False

    def list_references(self) -> List[str]:
        """Returns the list of saved reference voice IDs on the server."""
        req = urllib.request.Request(
            f"{self.base_url}/v1/references/list",
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("reference_ids", [])

    def generate_speech(
        self,
        text: str,
        output_path: str = "drama3_output.wav",
        references: Optional[List[Dict[str, Any]]] = None,
        reference_id: Optional[str] = None,
        audio_format: str = "wav",
        single_chunk: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.7,
        seed: Optional[int] = None,
        timeout: int = 600
    ) -> Optional[str]:
        """
        Synthesizes speech using Fish Audio Drama 3.

        :param text: Text to synthesize. Supports inline tags like [sighs], [laughs], [whisper]
                     and multi-speaker tags <|speaker:0|> ... <|speaker:15|>.
        :param output_path: Local path to save the generated audio file.
        :param references: Optional list of reference dicts for zero-shot voice cloning:
                           [{"audio_path": "ref.wav", "text": "Reference transcript", "speaker": 0}]
        :param reference_id: Optional saved voice ID on the server.
        """
        encoded_refs = []
        for ref in references or []:
            audio_bytes = Path(ref["audio_path"]).read_bytes()
            ref_entry: Dict[str, Any] = {
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "text": ref["text"],
            }
            if "speaker" in ref and ref["speaker"] is not None:
                ref_entry["speaker"] = int(ref["speaker"])
            encoded_refs.append(ref_entry)

        payload: Dict[str, Any] = {
            "text": text,
            "format": audio_format,
            "single_chunk": single_chunk,
            "references": encoded_refs,
            "reference_id": reference_id,
            "temperature": temperature,
            "top_p": top_p,
            "streaming": False,
        }
        if seed is not None:
            payload["seed"] = seed

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/tts",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "*/*"
            }
        )
        print(f"🎙️ Sending Drama 3 TTS request to {self.base_url}/v1/tts ...")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                audio_data = res.read()
                Path(output_path).write_bytes(audio_data)
                print(f"💾 Saved generated audio ({len(audio_data)} bytes) to: {output_path}")
                return output_path
        except urllib.error.HTTPError as e:
            print(f"[HTTP Error {e.code}] {e.read().decode('utf-8', errors='replace')}")
            return None
        except Exception as e:
            print(f"[Error] TTS generation failed: {e}")
            return None
