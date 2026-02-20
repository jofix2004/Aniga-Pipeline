"""
ImgCraft Boot — Gọi từ notebook Cell 2 để init/re-init FluxProcessor an toàn.
Tự động unload model cũ, giải phóng VRAM trước khi load mới.

Usage trong notebook Cell 2:
    import imgcraft_boot
    imgcraft_boot.start()
"""

import sys
import os
import gc
import time
import importlib
import subprocess
import threading

def start(port=8001):
    """Init FluxProcessor + Server + Cloudflare Tunnel (gọi 1 lần từ notebook)."""
    
    os.chdir('/content/Aniga-Pipeline/aniga_imgcraft')
    if '/content/Aniga-Pipeline/aniga_imgcraft' not in sys.path:
        sys.path.append('/content/Aniga-Pipeline/aniga_imgcraft')
    
    import torch
    import imgcraft_core
    import imgcraft_server
    
    # ============================================================
    # BƯỚC 1: Dọn dẹp cũ
    # ============================================================
    print("🧹 Dọn dẹp tiến trình cũ...")
    subprocess.run(f"fuser -k {port}/tcp", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("pkill -f uvicorn", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("pkill -f cloudflared", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Unload FluxProcessor cũ nếu có
    if hasattr(imgcraft_server.state, 'flux_processor') and imgcraft_server.state.flux_processor is not None:
        print("🗑️ Unloading FluxProcessor cũ khỏi VRAM...")
        imgcraft_server.state.flux_processor.unload()
        imgcraft_server.state.flux_processor = None
    
    # Dọn VRAM toàn bộ
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 VRAM sau dọn: {vram_used:.2f}GB used / {vram_reserved:.2f}GB reserved")
    
    # Reload modules để lấy code mới nhất
    importlib.reload(imgcraft_core)
    importlib.reload(imgcraft_server)
    # Re-import sau reload
    import imgcraft_core
    import imgcraft_server
    
    # ============================================================
    # BƯỚC 2: Init FluxProcessor mới
    # ============================================================
    print("📦 Khởi tạo FluxProcessor mới...")
    imgcraft_server.state.flux_processor = imgcraft_core.FluxProcessor()
    
    # ============================================================
    # BƯỚC 3: Start Server
    # ============================================================
    import uvicorn
    print(f"🚀 Bật ImgCraft Server (Port {port})...")
    server_config = uvicorn.Config(imgcraft_server.app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(server_config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(5)
    
    # ============================================================
    # BƯỚC 4: Cloudflare Tunnel
    # ============================================================
    import re
    print("🌐 Bật Cloudflare Tunnel...")
    cmd = f"cloudflared tunnel --url http://localhost:{port}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print("=" * 50)
    print("👇👇👇 LINK TRUY CẬP IMGCRAFT: 👇👇👇")
    try:
        url_printed = False
        for line in iter(process.stdout.readline, ''):
            if not url_printed and "trycloudflare.com" in line:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    print(f"\n🔗 URL: {match.group(0)}\n")
                    print("=" * 50)
                    url_printed = True
                elif "https://" in line:
                    url_part = line.split("https://")[1].split(" ")[0].strip()
                    print(f"\n🔗 URL: https://{url_part}\n")
                    print("=" * 50)
                    url_printed = True
            # Giữ vòng lặp để giữ cell chạy mãi mãi
    except KeyboardInterrupt:
        process.kill()
        print("\n🛑 Tunnel stopped.")
