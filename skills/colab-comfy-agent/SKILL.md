---
name: colab-comfy-agent
description: Automates or interacts with Google Colab ComfyUI A100 instances to trigger MiniMax-H3 (Hailuo) video and audio generation via API or headless browser.
---

# Colab ComfyUI Video Agent Skill (MiniMax-H3 on A100)

This skill enables AI agents to autonomously compose prompts, queue jobs, monitor rendering, and download generated `.mp4` videos from a Google Colab instance running ComfyUI on an A100 GPU.

## 1. MiniMax-H3 Official Prompting Guidelines

When the user asks for a video in natural language (e.g., in Russian or English), translate and expand their request into a structured English prompt following the **MiniMax-H3 Official Spec**:

1. **Character & Setting Lock**: Describe character appearance and environment first (`Preserve character identity: ...`, `Setting: ...`).
2. **Timecoded Shot List**: Break down the motion into time segments (e.g. `[0 to 2 seconds] ...`, `[2 to 4 seconds] ...`, `[4 to 5 seconds] ...`).
3. **Camera & Lens Direction**: Specify camera movement (`Smooth dramatic push-in`, `medium-close shot`, `shallow depth of field`, `24fps`).
4. **Native Stereo Audio Direction**: Include an `Audio:` section describing ambient sound, sound effects, voice/sighs, and music (`Audio: gentle rain hitting the window, soft piano chords...`).
5. **Explicit Negative Constraints**: End the positive prompt with `Do not show ...` and also populate `negative_prompt`.

---

## 2. Execution Patterns

### Pattern 1: Direct API Call via Cloudflare Tunnel (Recommended)

When the Colab notebook is already running and the user provides a `https://xxxx.trycloudflare.com` URL:

1. Verify endpoint health and submit the job using `minimax-h3-comfyui/client.py`:
   ```python
   import sys
   sys.path.insert(0, "minimax-h3-comfyui")
   from client import ComfyUIColabClient, create_minimax_t2v_prompt

   client = ComfyUIColabClient(base_url="https://your-tunnel.trycloudflare.com")
   assert client.check_health(), "ComfyUI tunnel is not reachable!"

   # Set use_turbo_lora=True for fast 8-step generation (~6-8 min),
   # or use_turbo_lora=False for standard 25-step generation (~25-30 min).
   workflow = create_minimax_t2v_prompt(
       positive_prompt=(
           "A 5-second 16:9 cinematic shot... "
           "[0 to 2 seconds] ... [2 to 5 seconds] ... "
           "Audio: ... "
           "Do not show ..."
       ),
       negative_prompt="blurry, distorted anatomy, glitch, low quality, fast jump cuts, watermark",
       width=1344,
       height=768,
       frames=124,
       use_turbo_lora=True,
       seed=42,
       filename_prefix="minimax_h3_agent"
   )

   prompt_id = client.queue_prompt(workflow)
   outputs = client.wait_for_completion(prompt_id, poll_interval=5, timeout=1800)
   print("Outputs:", outputs)
   ```

2. Extract the output filename from `outputs` and download the `.mp4` locally:
   ```python
   for node_id, node_output in (outputs or {}).items():
       for item in node_output.get("videos", []) + node_output.get("gifs", []) + node_output.get("images", []):
           filename = item.get("filename")
           subfolder = item.get("subfolder", "")
           if filename:
               client.download_output(filename, dest_path=f"./{filename}", subfolder=subfolder)
   ```

### Pattern 2: Full Browser Automation (Headless / Browser Agent)

If the user asks to launch the Colab notebook from scratch:
1. Open the Google Colab notebook URL:
   `https://colab.research.google.com/github/IjoiK/colab-comfy-a100-template/blob/main/minimax-h3-comfyui/colab_comfy_minimax_h3_a100.ipynb`
2. Select runtime: **Runtime** -> **Change runtime type** -> **A100 GPU**.
3. Trigger **Run all cells** (`Ctrl+F9`).
4. Monitor the stdout of **Step 4** until `ComfyUI is LIVE: https://xxxx.trycloudflare.com` appears.
5. Extract the `trycloudflare.com` tunnel URL and proceed with **Pattern 1**.
6. Once all requested videos are generated and saved, remind the user (or use the Colab menu **Runtime -> Disconnect and delete runtime**) to stop Compute Unit (CU) consumption immediately.
