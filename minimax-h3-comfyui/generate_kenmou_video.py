"""
Workflow script to generate Kenmou-kun 15-second crying video on Colab ComfyUI A100.
MiniMax-H3 supports long generation (up to 15s at 24fps -> ~360 frames).
"""

from client import ComfyUIColabClient, create_minimax_t2v_prompt

# 1. 接続先 Cloudflare Tunnel URL を指定 (ColabのStep 4で表示されたURL)
TUNNEL_URL = "https://your-tunnel-id.trycloudflare.com"

client = ComfyUIColabClient(base_url=TUNNEL_URL)

# 2. けんもうくん「Google Colabで動画生成ができる！」プロンプト構成
PROMPT = (
    "A funny anime character Kenmou-kun, crying comical tears of joy and screaming passionately with open mouth: "
    "'Google Colab can generate videos on A100 GPU!'. "
    "He is holding a glowing laptop showing high GPU utilization and progress bar. "
    "Dynamic camera zoom-in, vibrant high-octane anime motion, emotional dramatic lighting, 24fps, cinematic animation."
)

NEGATIVE_PROMPT = (
    "blurry, static, glitch, low resolution, deformed face, bad anatomy, watermark"
)

# 15秒動画設定 (24fps * 15s ≈ 361 frames)
workflow = create_minimax_t2v_prompt(
    positive_prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=1280,
    height=720,
    frames=361,   # 15秒分
    steps=30,     # A100の計算力を活かして高品質サンプリング
    seed=114514
)

print("🚀 A100 で 15秒のアニメ動画生成ジョブをキューに投入します...")
# prompt_id = client.queue_prompt(workflow)
# print(f"Job ID: {prompt_id}")
# outputs = client.wait_for_completion(prompt_id)
# print("Finished:", outputs)
