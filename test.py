import subprocess
import os

prompts = [
    "two giraffes in an enclosure at a zoo",
]

num_images_per_prompt = 100  # 每個 prompt 生成的圖片數量
out = "outputs/txt2img-samples/test"
os.makedirs(out, exist_ok=True)

for j, prompt in enumerate(prompts):
    for i in range(num_images_per_prompt):
        seed = i  # 設定隨機種子
        # Stable Diffusion txt2img 命令
        command = [
            "python", "scripts/txt2img.py",
            "--prompt", prompt,  # 確保 prompt 正確處理
            "--plms",
            "--precision", "autocast",
            "--H", "512", "--W", "512",
            "--n_samples", "1",
            "--skip_grid",
            "--outdir",out
            
        ]

        # 執行命令生成圖片
        subprocess.run(command)
        print(f"[{j+1}/{len(prompts)}] Prompt {j+1}: Generated image {i+1}/{num_images_per_prompt}")
