# imgcraft_server.py — ImgCraft Chapter Processor (Colab)
# FastAPI server + UI cho xử lý clean manga page-by-page
# Dùng core FluxProcessor từ pipeline_v2.py

import os
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import chapter_manager as cm
import imgcraft_core
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("ANIGA_WORK_DIR", "/content/drive/MyDrive/Aniga_Work/imgcraft")
UPLOAD_DIR = os.path.join(WORK_DIR, "_uploads")
SAVE_DIR = os.environ.get("ANIGA_SAVE_DIR", "/content/drive/MyDrive/Aniga_Work/imgcraft/_output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI(title="ImgCraft Chapter Processor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# STATE
# ============================================================
class ProcessorState:
    def __init__(self):
        self.queue = []              # [{"id": str, "bundle_path": str, "manifest": dict, "working_dir": str}]
        self.active_id = None        # ID dự án đang chạy
        self.should_pause = False    # Flag pause
        self.is_running = False      # Đang xử lý?
        self.current_page = None     # Page đang xử lý
        self.lock = threading.Lock()
        self.worker_thread = None
        self.flux_processor = None   # Sẽ được set từ notebook

state = ProcessorState()


def _find_queue_item(project_id):
    for item in state.queue:
        if item["id"] == project_id:
            return item
    return None


# ============================================================
# CORE: Worker thread — xử lý tuần tự
# ============================================================
def _worker_loop():
    """Main worker loop: xử lý trang tuần tự cho dự án active."""
    while state.is_running and state.active_id:
        item = _find_queue_item(state.active_id)
        if not item:
            break

        manifest = cm.read_manifest_from_dir(item["working_dir"])
        next_page = cm.get_next_pending(manifest, "clean")

        if next_page is None:
            # Xong hết
            print(f"✅ Xong tất cả trang cho: {manifest['project_name']}")
            state.is_running = False
            state.active_id = None
            state.current_page = None
            # Auto-pack
            _auto_pack(item)
            break

        if state.should_pause:
            print(f"⏸ Đã pause sau khi xong trang trước.")
            state.is_running = False
            state.should_pause = False
            state.current_page = None
            # Pack trước khi pause
            _auto_pack(item)
            break

        hidden_id = next_page["hidden_id"]
        state.current_page = next_page
        page_order = next_page["display_order"] + 1
        total = len(manifest["pages"])
        print(f"🎨 Xử lý trang {page_order}/{total}: {hidden_id}")

        try:
            # === CORE PROCESSING ===
            raw_path = cm.get_page_from_dir(item["working_dir"], hidden_id, "raw")
            if raw_path is None:
                raise Exception("Không tìm thấy raw.png")

            from PIL import Image
            raw_img = Image.open(raw_path)

            # Đọc flux_size từ manifest config
            imgcraft_config = manifest.get("imgcraft_config", {})
            flux_size = imgcraft_config.get("flux_size", 1280)
            content_bbox = next_page.get("content_bbox")

            # Gọi core FluxProcessor
            clean_img = _process_single_page(raw_img, flux_size=flux_size, content_bbox=content_bbox)

            # Transactional save
            cm.save_page_to_dir(item["working_dir"], hidden_id, "clean", clean_img)
            manifest = cm.read_manifest_from_dir(item["working_dir"])
            cm.mark_page_done(manifest, hidden_id, "clean")
            cm.update_manifest_in_dir(item["working_dir"], manifest)

            done, total = cm.get_progress(manifest, "clean")
            print(f"✅ Trang {page_order}/{total} done ({done}/{total} tổng)")

        except Exception as e:
            print(f"❌ Lỗi trang {hidden_id}: {e}")
            manifest = cm.read_manifest_from_dir(item["working_dir"])
            cm.mark_page_done(manifest, hidden_id, "clean", error=str(e))
            cm.update_manifest_in_dir(item["working_dir"], manifest)

    state.is_running = False
    state.current_page = None


def _process_single_page(raw_pil, flux_size=1280, content_bbox=None):
    """
    Xử lý 1 trang raw → clean sử dụng core algorithm.
    flux_size: kích thước resize tile trước khi đưa vào Flux (1024/1280/1536)
    """
    if state.flux_processor is None:
        raise Exception("FluxProcessor chưa được init. Chạy cell khởi tạo trên notebook trước.")

    try:
        # 1. Master Data (Giả định ảnh raw_pil đã được resize 2048x2048 từ Local)
        master_img_pil = raw_pil
        master_cv = cv2.cvtColor(np.array(master_img_pil), cv2.COLOR_RGB2BGR)
        master_data = imgcraft_core.prepare_master_data(master_cv)

        # 2. Cắt 4 tiles trên RAM (Kích thước gốc 1024)
        # Ý đồ: Cắt ảnh dựa vào nội dung thật (content_bbox). Phù hợp khổ ngang lẫn dọc
        size = 1024
        if content_bbox:
            x_left, y_top, x_right_raw, y_bottom_raw = content_bbox
            # Tính toán 2 góc còn lại để quét trọn ảnh, tối đa 1024, không vọt ra ngoài
            x_right = max(x_left, x_right_raw - size)
            y_bottom = max(y_top, y_bottom_raw - size)
        else:
            # Fallback cũ lấy trục dọc trọng tâm nếu ảnh cũ chưa có bbox
            x_left = 250
            x_right = 774
            y_top = 60
            y_bottom = 964

        coords = [
            (x_left, y_top),         # Tile 0: Top-Left
            (x_right, y_top),        # Tile 1: Top-Right
            (x_left, y_bottom),      # Tile 2: Bottom-Left
            (x_right, y_bottom)      # Tile 3: Bottom-Right
        ]
        tiles_pil = []
        for (x, y) in coords:
            box = (x, y, x + size, y + size)
            cropped_tile = master_img_pil.crop(box)
            # Resize tile lên flux_size x flux_size trước khi cho vào Flux
            tiles_pil.append(cropped_tile.resize((flux_size, flux_size), imgcraft_core.Image.Resampling.LANCZOS))

        # 3. Clean bằng FluxProcessor (GPU)
        clean_tiles_pil = state.flux_processor.process(tiles_pil)

        # 4. Gom & Alignment (CPU)
        tile_results = []
        for i, clean_tile in enumerate(clean_tiles_pil):
            clean_tile_cv = cv2.cvtColor(np.array(clean_tile), cv2.COLOR_RGB2BGR)
            
            res = imgcraft_core.process_tile_core(clean_tile_cv, master_cv, master_data, refine=True)
            res["tile_index"] = i
            tile_results.append(res)
            
        # 5. Smart Stitch → trả ảnh clean 2048x2048
        final_pil = imgcraft_core.stack_images(master_cv, tile_results)

        return final_pil

    except Exception as e:
        print(f"❌ Core processing error: {e}")
        raise e


def _auto_pack(item):
    """Đóng gói working dir → .aniga (overwrite)."""
    try:
        cm.pack_to_bundle(item["working_dir"], item["bundle_path"])
    except Exception as e:
        print(f"⚠️ Pack error: {e}")


# ============================================================
# API: Upload .aniga
# ============================================================
@app.post("/api/upload")
async def upload_bundle(file: UploadFile = File(...)):
    if not file.filename.endswith(".aniga"):
        raise HTTPException(400, "Chỉ nhận file .aniga")

    # Lưu file (đảm bảo folder tồn tại — Drive có thể lag)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    bundle_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(bundle_path, 'wb') as out:
        content = await file.read()
        out.write(content)

    # Đọc manifest
    manifest = cm.read_manifest(bundle_path)
    project_id = manifest["project_id"]

    # Kiểm tra trùng
    existing = _find_queue_item(project_id)
    if existing:
        raise HTTPException(409, f"Dự án {manifest['project_name']} đã có trong hàng chờ")

    # Giải nén vào working dir
    working_dir = os.path.join(WORK_DIR, project_id)
    manifest = cm.extract_to_working_dir(bundle_path, working_dir)

    state.queue.append({
        "id": project_id,
        "bundle_path": bundle_path,
        "manifest": manifest,
        "working_dir": working_dir,
    })

    done, total = cm.get_progress(manifest, "clean")
    return {
        "status": "ok",
        "project_id": project_id,
        "project_name": manifest["project_name"],
        "progress": f"{done}/{total}"
    }


# ============================================================
# API: Danh sách queue
# ============================================================
@app.get("/api/queue")
def get_queue():
    items = []
    for item in state.queue:
        manifest = cm.read_manifest_from_dir(item["working_dir"])
        done, total = cm.get_progress(manifest, "clean")
        is_active = item["id"] == state.active_id
        items.append({
            "id": item["id"],
            "project_name": manifest["project_name"],
            "done": done,
            "total": total,
            "is_active": is_active,
            "is_running": is_active and state.is_running,
            "is_pausing": is_active and state.should_pause,
            "current_page": state.current_page["display_order"] + 1 if is_active and state.current_page else None,
            "pages": [{
                "hidden_id": p["hidden_id"],
                "display_order": p["display_order"],
                "display_name": cm.get_display_name(manifest["project_name"], p["display_order"]),
                "has_clean": p.get("has_clean", False),
                "error": p.get("error"),
            } for p in sorted(manifest["pages"], key=lambda x: x["display_order"])]
        })
    return {"queue": items, "is_any_running": state.is_running}


# ============================================================
# API: Start
# ============================================================
@app.post("/api/start/{project_id}")
def start_project(project_id: str):
    if state.is_running:
        raise HTTPException(409, "Đang chạy dự án khác. Pause trước!")

    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy dự án trong hàng chờ")

    state.active_id = project_id
    state.is_running = True
    state.should_pause = False

    state.worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    state.worker_thread.start()

    return {"status": "started"}


# ============================================================
# API: Pause
# ============================================================
@app.post("/api/pause")
def pause():
    if not state.is_running:
        raise HTTPException(400, "Không có dự án nào đang chạy")

    state.should_pause = True
    return {"status": "pausing", "message": "Sẽ dừng sau khi xong trang hiện tại"}


# ============================================================
# API: Download .aniga
# ============================================================
@app.get("/api/download/{project_id}")
def download(project_id: str):
    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")

    # Pack latest từ working dir
    snapshot_path = os.path.join(UPLOAD_DIR, f"_snapshot_{project_id}.aniga")
    cm.pack_to_bundle(item["working_dir"], snapshot_path)
    manifest = cm.read_manifest_from_dir(item["working_dir"])
    filename = f"{manifest['project_name'].replace(' ', '_')}.aniga"
    return FileResponse(snapshot_path, media_type="application/octet-stream", filename=filename)


# ============================================================
# API: Save to Google Drive
# ============================================================
@app.post("/api/save-drive/{project_id}")
def save_to_drive(project_id: str):
    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")

    manifest = cm.read_manifest_from_dir(item["working_dir"])
    filename = f"{manifest['project_name'].replace(' ', '_')}.aniga"
    save_path = os.path.join(SAVE_DIR, filename)

    # Pack latest từ working dir → save thẳng vào Drive
    cm.pack_to_bundle(item["working_dir"], save_path)
    print(f"💾 Saved to Drive: {save_path}")
    return {"status": "ok", "path": save_path, "filename": filename}


# ============================================================
# API: Remove from queue
# ============================================================
@app.delete("/api/queue/{project_id}")
def remove_from_queue(project_id: str):
    if state.active_id == project_id and state.is_running:
        raise HTTPException(409, "Đang chạy! Pause trước!")

    state.queue = [item for item in state.queue if item["id"] != project_id]
    if state.active_id == project_id:
        state.active_id = None
    return {"status": "removed"}


# ============================================================
# HTML SPA
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImgCraft Chapter Processor</title>
    <style>
        :root { --bg:#0f0f14; --bg2:#1a1a24; --bg3:#252535; --text:#e0e0ef; --text2:#8888aa;
                --accent:#f59e42; --accent2:#d68530; --green:#4caf50; --red:#e74c3c;
                --border:#2a2a3a; --radius:10px; }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
        .topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;gap:16px;}
        .topbar h1{font-size:20px;font-weight:700;}
        .container{max-width:900px;margin:0 auto;padding:24px;}

        .upload-area{background:var(--bg2);border:2px dashed var(--border);border-radius:var(--radius);padding:24px;text-align:center;margin-bottom:24px;cursor:pointer;transition:border-color 0.2s;}
        .upload-area:hover{border-color:var(--accent);}
        .upload-area input{display:none;}

        .queue-item{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:12px;}
        .queue-item .header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
        .queue-item .name{font-size:16px;font-weight:600;}
        .queue-item .progress{color:var(--text2);font-size:13px;}
        .queue-item .actions{display:flex;gap:8px;}
        .queue-item .actions button{padding:6px 14px;border-radius:6px;border:none;cursor:pointer;font-size:13px;font-weight:600;}
        .btn-start{background:var(--green);color:#fff;}
        .btn-start:disabled{background:var(--bg3);color:var(--text2);cursor:not-allowed;}
        .btn-pause{background:var(--accent);color:#fff;}
        .btn-dl{background:var(--bg3);color:var(--text);border:1px solid var(--border) !important;}
        .btn-rm{background:var(--bg3);color:var(--red);border:1px solid var(--border) !important;}

        .pages-bar{display:flex;gap:3px;flex-wrap:wrap;margin-top:8px;}
        .pages-bar .pg{width:24px;height:24px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;}
        .pg-done{background:#1b5e20;color:#a5d6a7;}
        .pg-pending{background:var(--bg3);color:var(--text2);}
        .pg-active{background:var(--accent);color:#fff;animation:pulse 1s infinite;}
        .pg-error{background:#b71c1c;color:#ef9a9a;}
        @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.5;}}

        .status-banner{background:var(--bg2);border:1px solid var(--accent);border-radius:var(--radius);padding:12px 16px;margin-bottom:24px;display:none;font-size:14px;}
        .status-banner.active{display:block;}

        .toast{position:fixed;bottom:24px;right:24px;background:var(--bg2);border:1px solid var(--accent);color:var(--text);padding:12px 20px;border-radius:8px;font-size:14px;z-index:200;display:none;}
    </style>
</head>
<body>
    <div class="topbar"><h1>🏭 ImgCraft Chapter Processor</h1></div>
    <div class="container">
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept=".aniga" onchange="uploadBundle(this)">
            <div style="font-size:24px;margin-bottom:8px;">📤</div>
            <div>Click để upload file .aniga</div>
            <div style="color:var(--text2);font-size:13px;margin-top:4px;">Hoặc kéo thả vào đây</div>
        </div>

        <div class="status-banner" id="statusBanner"></div>

        <div id="queueList"></div>
    </div>
    <div class="toast" id="toast"></div>

    <script>
    function toast(msg,d=3000){const t=document.getElementById('toast');t.textContent=msg;t.style.display='block';setTimeout(()=>t.style.display='none',d);}

    async function uploadBundle(input){
        const file=input.files[0]; if(!file)return;
        const fd=new FormData(); fd.append('file',file);
        toast('⏳ Đang upload...');
        try{
            const res=await fetch('/api/upload',{method:'POST',body:fd});
            const data=await res.json();
            if(!res.ok){toast('❌ '+data.detail,5000);return;}
            toast(`✅ Đã upload: ${data.project_name} (${data.progress})`);
            refreshQueue();
        }catch(e){toast('❌ Lỗi upload: '+e);}
        input.value='';
    }

    async function refreshQueue(){
        const res=await fetch('/api/queue');
        const data=await res.json();
        const el=document.getElementById('queueList');
        const banner=document.getElementById('statusBanner');

        if(!data.queue.length){
            el.innerHTML='<div style="text-align:center;color:var(--text2);padding:40px;">Chưa có dự án nào. Upload file .aniga để bắt đầu.</div>';
            banner.classList.remove('active');
            return;
        }

        // Status banner
        const running=data.queue.find(q=>q.is_running);
        if(running){
            banner.classList.add('active');
            banner.innerHTML=`🎨 Đang xử lý: <b>${running.project_name}</b> — Trang ${running.current_page||'?'}/${running.total} (${running.done}/${running.total} done)`;
        }else{
            const pausing=data.queue.find(q=>q.is_pausing);
            if(pausing){banner.classList.add('active');banner.innerHTML='⏳ Đang pause (chờ xong trang hiện tại)...';}
            else{banner.classList.remove('active');}
        }

        el.innerHTML=data.queue.map(q=>{
            const canStart=!data.is_any_running;
            const isActive=q.is_active;
            let actionsHtml='';
            if(q.is_running){
                actionsHtml=`<button class="btn-pause" onclick="pauseProject()">⏸ Pause</button>`;
            }else if(q.is_pausing){
                actionsHtml=`<button class="btn-pause" disabled>⏳ Đang dừng...</button>`;
            }else{
                actionsHtml=`<button class="btn-start" ${canStart?'':'disabled'} onclick="startProject('${q.id}')">▶ Start</button>`;
            }
            actionsHtml+=`<button class="btn-dl" onclick="saveToDrive('${q.id}')">💾 Drive</button>`;
            actionsHtml+=`<button class="btn-dl" onclick="downloadProject('${q.id}')">📥</button>`;
            if(!q.is_running)actionsHtml+=`<button class="btn-rm" onclick="removeProject('${q.id}')">🗑️</button>`;

            const pagesHtml=q.pages.map(p=>{
                let cls='pg-pending';
                if(p.error)cls='pg-error';
                else if(p.has_clean)cls='pg-done';
                else if(q.is_running && q.current_page===p.display_order+1)cls='pg-active';
                return `<div class="pg ${cls}" title="${p.display_name}">${p.display_order+1}</div>`;
            }).join('');

            return `<div class="queue-item">
                <div class="header">
                    <div>
                        <span class="name">${q.is_running?'🟢':'⚪'} ${q.project_name}</span>
                        <span class="progress">&nbsp; ${q.done}/${q.total} pages</span>
                    </div>
                    <div class="actions">${actionsHtml}</div>
                </div>
                <div class="pages-bar">${pagesHtml}</div>
            </div>`;
        }).join('');
    }

    async function startProject(id){
        const res=await fetch(`/api/start/${id}`,{method:'POST'});
        if(!res.ok){const d=await res.json();toast('❌ '+d.detail);return;}
        toast('▶ Đã start!');refreshQueue();
    }
    async function pauseProject(){
        await fetch('/api/pause',{method:'POST'});
        toast('⏸ Đang pause (chờ xong trang hiện tại)...');refreshQueue();
    }
    function downloadProject(id){window.location.href=`/api/download/${id}`;}
    async function saveToDrive(id){
        toast('💾 Đang lưu vào Drive...');
        try{
            const res=await fetch(`/api/save-drive/${id}`,{method:'POST'});
            const data=await res.json();
            if(!res.ok){toast('❌ '+data.detail,5000);return;}
            toast(`✅ Đã lưu: ${data.filename}`,5000);
        }catch(e){toast('❌ Lỗi: '+e);}
    }
    async function removeProject(id){
        if(!confirm('Xóa dự án khỏi hàng chờ?'))return;
        await fetch(`/api/queue/${id}`,{method:'DELETE'});
        toast('🗑️ Đã xóa');refreshQueue();
    }

    // Auto-refresh
    refreshQueue();
    setInterval(refreshQueue,3000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🏭 ImgCraft Chapter Processor — http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
