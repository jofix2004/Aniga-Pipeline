# aniga3_server.py — Aniga3 Chapter Processor (Colab)
# FastAPI server + UI cho detection + masking page-by-page
# Dùng core run_full_pipeline() từ Aniga3-main

import os
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import chapter_manager as cm
import core as aniga3_core
import config as aniga3_config

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("ANIGA_WORK_DIR", "/content/drive/MyDrive/Aniga_Work/aniga3")
UPLOAD_DIR = os.path.join(WORK_DIR, "_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Default Aniga3 config
DEFAULT_CONFIG = {
    "mask_classes": ["text", "sfx", "b1"],
    "expand": 31,
    "feather": 21,
    "blur": 10,
    "min_area": 100,
    "cleanup": 7,
    "overlap_threshold": 0.1,
    "confidence_threshold": 0.25,
    "ocr_mode": "Không bật",
}

app = FastAPI(title="Aniga3 Chapter Processor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# STATE
# ============================================================
class ProcessorState:
    def __init__(self):
        self.queue = []
        self.active_id = None
        self.should_pause = False
        self.is_running = False
        self.current_page = None
        self.lock = threading.Lock()
        self.worker_thread = None
        self.pipeline_func = None     # Sẽ được set từ notebook

state = ProcessorState()


def _find_queue_item(project_id):
    for item in state.queue:
        if item["id"] == project_id:
            return item
    return None


# ============================================================
# CORE: Worker thread
# ============================================================
def _worker_loop():
    """Xử lý tuần tự: detect + mask cho từng page."""
    while state.is_running and state.active_id:
        item = _find_queue_item(state.active_id)
        if not item:
            break

        manifest = cm.read_manifest_from_dir(item["working_dir"])

        # Tìm trang pending theo mode
        mode = item.get("run_mode", "all")  # "all" | "errors" | "range"

        if mode == "errors":
            # Chỉ chạy trang lỗi
            pending = [p for p in manifest["pages"]
                       if p.get("error") and not p.get("has_mask", False)]
            pending = sorted(pending, key=lambda p: p["display_order"])
            next_page = pending[0] if pending else None
        elif mode == "range":
            range_start = item.get("range_start", 0)
            range_end = item.get("range_end", len(manifest["pages"]))
            pending = [p for p in manifest["pages"]
                       if range_start <= p["display_order"] < range_end
                       and not p.get("has_mask", False)
                       and p.get("error") is None]
            pending = sorted(pending, key=lambda p: p["display_order"])
            next_page = pending[0] if pending else None
        else:
            next_page = cm.get_next_pending(manifest, "mask")

        if next_page is None:
            print(f"✅ Xong tất cả trang cho: {manifest['project_name']}")
            state.is_running = False
            state.active_id = None
            state.current_page = None
            _auto_pack(item)
            break

        if state.should_pause:
            print(f"⏸ Đã pause.")
            state.is_running = False
            state.should_pause = False
            state.current_page = None
            _auto_pack(item)
            break

        hidden_id = next_page["hidden_id"]
        state.current_page = next_page
        page_order = next_page["display_order"] + 1
        total = len(manifest["pages"])
        print(f"🔍 Xử lý trang {page_order}/{total}: {hidden_id}")

        try:
            # Cần clean image
            clean_path = cm.get_page_from_dir(item["working_dir"], hidden_id, "clean")
            if clean_path is None:
                raise Exception("Chưa có clean.png — skip")

            raw_path = cm.get_page_from_dir(item["working_dir"], hidden_id, "raw")

            from PIL import Image
            clean_img = Image.open(clean_path)
            raw_img = Image.open(raw_path) if raw_path else None

            # Gọi core pipeline
            config = item.get("config", DEFAULT_CONFIG)
            mask_img, detections = _process_single_page(raw_img, clean_img, config)

            # Transactional save
            cm.save_page_to_dir(item["working_dir"], hidden_id, "mask", mask_img)
            cm.save_detections_to_dir(item["working_dir"], hidden_id, detections)

            manifest = cm.read_manifest_from_dir(item["working_dir"])
            cm.mark_page_done(manifest, hidden_id, "mask")
            cm.mark_page_done(manifest, hidden_id, "detections")
            cm.update_manifest_in_dir(item["working_dir"], manifest)

            done, total = cm.get_progress(manifest, "mask")
            print(f"✅ Trang {page_order}/{total} mask done ({done}/{total} tổng)")

        except Exception as e:
            print(f"❌ Lỗi trang {hidden_id}: {e}")
            manifest = cm.read_manifest_from_dir(item["working_dir"])
            cm.mark_page_done(manifest, hidden_id, "mask", error=str(e))
            cm.update_manifest_in_dir(item["working_dir"], manifest)

    state.is_running = False
    state.current_page = None


def _process_single_page(raw_pil, clean_pil, config):
    """
    Xử lý 1 trang: detect + mask.
    Sử dụng core Aniga3 pipeline.
    
    Returns: (mask_pil, detections_dict)
    """
    # Xây dựng tham số mask_params từ config
    mask_params = {
        "blur": config.get("blur", 10),
        "min_area": config.get("min_area", 100),
        "cleanup": config.get("cleanup", 7),
        "overlap_threshold": config.get("overlap_threshold", 0.1),
        "expand": config.get("expand", 31),
        "feather": config.get("feather", 21),
    }

    # Override confidence threshold vào runtime config nếu cần
    conf_thr = config.get("confidence_threshold", 0.25)
    aniga3_config.CONFIG['single_model_defaults']['conf_threshold'] = conf_thr

    # Phân giải classes
    mask_classes = config.get("mask_classes", ["text", "sfx", "b1"])
    ocr_mode = config.get("ocr_mode", "Không bật")

    result = aniga3_core.run_full_pipeline(
        raw_image_pil=raw_pil,
        clean_image_pil=clean_pil,
        device_mode="Auto",
        ocr_mode=ocr_mode,
        mask_classes=mask_classes,
        mask_params=mask_params,
    )

    final_mask_np = result["final_mask"]
    bbox_json = result["bbox_json"]
    logs = result["logs"]

    from PIL import Image
    mask_pil = Image.fromarray(final_mask_np)
    
    # Pack lại thành dict
    detections_dict = {
        "boxes": bbox_json,
        "logs": logs
    }
    
    return mask_pil, detections_dict


def _auto_pack(item):
    try:
        cm.pack_to_bundle(item["working_dir"], item["bundle_path"])
    except Exception as e:
        print(f"⚠️ Pack error: {e}")


# ============================================================
# API: Upload
# ============================================================
@app.post("/api/upload")
async def upload_bundle(file: UploadFile = File(...)):
    if not file.filename.endswith(".aniga"):
        raise HTTPException(400, "Chỉ nhận file .aniga")

    bundle_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(bundle_path, 'wb') as out:
        content = await file.read()
        out.write(content)

    manifest = cm.read_manifest(bundle_path)
    project_id = manifest["project_id"]

    existing = _find_queue_item(project_id)
    if existing:
        raise HTTPException(409, f"Dự án {manifest['project_name']} đã có trong hàng chờ")

    working_dir = os.path.join(WORK_DIR, project_id)
    manifest = cm.extract_to_working_dir(bundle_path, working_dir)

    config = manifest.get("aniga3_config") or DEFAULT_CONFIG.copy()

    state.queue.append({
        "id": project_id,
        "bundle_path": bundle_path,
        "manifest": manifest,
        "working_dir": working_dir,
        "config": config,
        "run_mode": "all",
    })

    done, total = cm.get_progress(manifest, "mask")
    return {
        "status": "ok",
        "project_id": project_id,
        "project_name": manifest["project_name"],
        "progress": f"{done}/{total}"
    }


# ============================================================
# API: Queue
# ============================================================
@app.get("/api/queue")
def get_queue():
    items = []
    for item in state.queue:
        manifest = cm.read_manifest_from_dir(item["working_dir"])
        done, total = cm.get_progress(manifest, "mask")
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
            "config": item.get("config", DEFAULT_CONFIG),
            "run_mode": item.get("run_mode", "all"),
            "pages": [{
                "hidden_id": p["hidden_id"],
                "display_order": p["display_order"],
                "display_name": cm.get_display_name(manifest["project_name"], p["display_order"]),
                "has_clean": p.get("has_clean", False),
                "has_mask": p.get("has_mask", False),
                "has_detections": p.get("has_detections", False),
                "error": p.get("error"),
            } for p in sorted(manifest["pages"], key=lambda x: x["display_order"])]
        })
    return {"queue": items, "is_any_running": state.is_running}


# ============================================================
# API: Config
# ============================================================
@app.post("/api/config/{project_id}")
async def set_config(project_id: str, request: Request):
    """Cập nhật config cho dự án trước khi Start."""
    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")
    if state.is_running and state.active_id == project_id:
        raise HTTPException(409, "Đang chạy — không thể đổi config!")

    body = await request.json()
    config = {**DEFAULT_CONFIG, **body.get("config", {})}
    item["config"] = config

    # Lưu vào manifest trên drive
    manifest = cm.read_manifest_from_dir(item["working_dir"])
    manifest["aniga3_config"] = config
    cm.update_manifest_in_dir(item["working_dir"], manifest)

    return {"status": "ok", "config": config}


# ============================================================
# API: Start (với mode)
# ============================================================
@app.post("/api/start/{project_id}")
async def start_project(project_id: str, request: Request):
    if state.is_running:
        raise HTTPException(409, "Đang chạy dự án khác!")

    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")

    body = await request.json() if True else {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    mode = body.get("mode", "all")     # "all" | "errors" | "range"
    item["run_mode"] = mode
    if mode == "range":
        item["range_start"] = body.get("range_start", 0)
        item["range_end"] = body.get("range_end", 999)

    # Reset lỗi nếu chạy lại trang lỗi
    if mode == "errors":
        manifest = cm.read_manifest_from_dir(item["working_dir"])
        for page in manifest["pages"]:
            if page.get("error"):
                page["error"] = None
        cm.update_manifest_in_dir(item["working_dir"], manifest)

    state.active_id = project_id
    state.is_running = True
    state.should_pause = False

    state.worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    state.worker_thread.start()

    return {"status": "started", "mode": mode}


# ============================================================
# API: Pause / Download / Remove / Clean
# ============================================================
@app.post("/api/pause")
def pause():
    if not state.is_running:
        raise HTTPException(400, "Không có gì đang chạy")
    state.should_pause = True
    return {"status": "pausing"}


@app.get("/api/download/{project_id}")
def download(project_id: str):
    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")
    snapshot_path = os.path.join(UPLOAD_DIR, f"_snapshot_{project_id}.aniga")
    cm.pack_to_bundle(item["working_dir"], snapshot_path)
    manifest = cm.read_manifest_from_dir(item["working_dir"])
    filename = f"{manifest['project_name'].replace(' ', '_')}.aniga"
    return FileResponse(snapshot_path, media_type="application/octet-stream", filename=filename)


@app.delete("/api/queue/{project_id}")
def remove_from_queue(project_id: str):
    if state.active_id == project_id and state.is_running:
        raise HTTPException(409, "Đang chạy! Pause trước!")
    state.queue = [item for item in state.queue if item["id"] != project_id]
    if state.active_id == project_id:
        state.active_id = None
    return {"status": "removed"}


@app.post("/api/clean/{project_id}")
def clean_data(project_id: str):
    """Xóa toàn bộ mask + detections (clean data)."""
    item = _find_queue_item(project_id)
    if not item:
        raise HTTPException(404, "Không tìm thấy")
    if state.is_running and state.active_id == project_id:
        raise HTTPException(409, "Đang chạy!")

    manifest = cm.clean_detection_data_in_dir(item["working_dir"])
    return {"status": "ok", "message": "Đã xóa toàn bộ mask + detections"}


# ============================================================
# HTML SPA
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aniga3 Chapter Processor</title>
    <style>
        :root { --bg:#0f0f14; --bg2:#1a1a24; --bg3:#252535; --text:#e0e0ef; --text2:#8888aa;
                --accent:#7c6ff7; --accent2:#5a4fd4; --green:#4caf50; --red:#e74c3c; --yellow:#f39c12;
                --border:#2a2a3a; --radius:10px; }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
        .topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;gap:16px;}
        .topbar h1{font-size:20px;font-weight:700;}
        .container{max-width:960px;margin:0 auto;padding:24px;}

        .upload-area{background:var(--bg2);border:2px dashed var(--border);border-radius:var(--radius);padding:24px;text-align:center;margin-bottom:24px;cursor:pointer;transition:border-color 0.2s;}
        .upload-area:hover{border-color:var(--accent);}
        .upload-area input{display:none;}

        .queue-item{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:16px;}
        .qi-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
        .qi-header .name{font-size:16px;font-weight:600;}
        .qi-actions{display:flex;gap:6px;flex-wrap:wrap;}
        .qi-actions button{padding:6px 12px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:600;}
        .btn-start{background:var(--green);color:#fff;}
        .btn-start:disabled{background:var(--bg3);color:var(--text2);cursor:not-allowed;}
        .btn-pause{background:var(--yellow);color:#fff;}
        .btn-dl{background:var(--bg3);color:var(--text);border:1px solid var(--border)!important;}
        .btn-rm{background:var(--bg3);color:var(--red);border:1px solid var(--border)!important;}
        .btn-clean{background:var(--bg3);color:var(--yellow);border:1px solid var(--border)!important;}
        .btn-err{background:#b71c1c;color:#ef9a9a;}

        /* Config panel */
        .config-panel{background:var(--bg3);border-radius:8px;padding:12px;margin:10px 0;font-size:13px;}
        .config-panel label{display:inline-block;margin-right:12px;color:var(--text2);}
        .config-panel input[type=number]{width:60px;padding:4px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);text-align:center;}
        .config-panel .cb{display:inline-flex;align-items:center;gap:4px;margin-right:10px;}

        .pages-bar{display:flex;gap:3px;flex-wrap:wrap;margin-top:8px;}
        .pg{width:24px;height:24px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;}
        .pg-done{background:#1b5e20;color:#a5d6a7;}
        .pg-pending{background:var(--bg3);color:var(--text2);}
        .pg-active{background:var(--accent);color:#fff;animation:pulse 1s infinite;}
        .pg-error{background:#b71c1c;color:#ef9a9a;}
        .pg-noclean{background:#37474f;color:#78909c;}
        @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.5;}}

        .status-banner{background:var(--bg2);border:1px solid var(--accent);border-radius:var(--radius);padding:12px 16px;margin-bottom:24px;display:none;font-size:14px;}
        .status-banner.active{display:block;}

        .mode-row{display:flex;gap:6px;margin:8px 0;flex-wrap:wrap;}
        .mode-row button{padding:4px 10px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);color:var(--text2);cursor:pointer;font-size:12px;}
        .mode-row button.sel{border-color:var(--accent);color:var(--accent);}

        .toast{position:fixed;bottom:24px;right:24px;background:var(--bg2);border:1px solid var(--accent);color:var(--text);padding:12px 20px;border-radius:8px;font-size:14px;z-index:200;display:none;}
    </style>
</head>
<body>
    <div class="topbar"><h1>🔍 Aniga3 Chapter Processor</h1></div>
    <div class="container">
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept=".aniga" onchange="uploadBundle(this)">
            <div style="font-size:24px;margin-bottom:8px;">📤</div>
            <div>Click để upload file .aniga</div>
        </div>
        <div class="status-banner" id="statusBanner"></div>
        <div id="queueList"></div>
    </div>
    <div class="toast" id="toast"></div>

    <script>
    function toast(msg,d=3000){const t=document.getElementById('toast');t.textContent=msg;t.style.display='block';setTimeout(()=>t.style.display='none',d);}

    // Track selected modes per project
    const projectModes = {};

    async function uploadBundle(input){
        const file=input.files[0]; if(!file)return;
        const fd=new FormData(); fd.append('file',file);
        toast('⏳ Đang upload...');
        try{
            const res=await fetch('/api/upload',{method:'POST',body:fd});
            const data=await res.json();
            if(!res.ok){toast('❌ '+data.detail,5000);return;}
            toast(`✅ ${data.project_name} (${data.progress})`);
            refreshQueue();
        }catch(e){toast('❌ '+e);}
        input.value='';
    }

    async function refreshQueue(){
        const res=await fetch('/api/queue');
        const data=await res.json();
        const el=document.getElementById('queueList');
        const banner=document.getElementById('statusBanner');

        if(!data.queue.length){
            el.innerHTML='<div style="text-align:center;color:var(--text2);padding:40px;">Chưa có dự án. Upload .aniga để bắt đầu.</div>';
            banner.classList.remove('active');
            return;
        }

        const running=data.queue.find(q=>q.is_running);
        if(running){
            banner.classList.add('active');
            banner.innerHTML=`🔍 Đang xử lý: <b>${running.project_name}</b> — Trang ${running.current_page||'?'}/${running.total} (${running.done}/${running.total} mask done)`;
        }else{banner.classList.remove('active');}

        el.innerHTML=data.queue.map(q=>{
            const canStart=!data.is_any_running;
            const mode=projectModes[q.id]||'all';
            const errCount=q.pages.filter(p=>p.error).length;
            const noClean=q.pages.filter(p=>!p.has_clean).length;

            let actHtml='';
            if(q.is_running){actHtml=`<button class="btn-pause" onclick="doPause()">⏸ Pause</button>`;}
            else if(q.is_pausing){actHtml=`<button disabled style="opacity:0.5">⏳ Đang dừng...</button>`;}
            else{actHtml=`<button class="btn-start" ${canStart?'':'disabled'} onclick="doStart('${q.id}')">▶ Start</button>`;}
            actHtml+=`<button class="btn-dl" onclick="location.href='/api/download/${q.id}'">📥</button>`;
            if(!q.is_running){
                actHtml+=`<button class="btn-clean" onclick="doClean('${q.id}')">🗑️ Clean data</button>`;
                actHtml+=`<button class="btn-rm" onclick="doRemove('${q.id}')">❌</button>`;
            }
            if(errCount>0&&!q.is_running){
                actHtml+=`<button class="btn-err" onclick="setMode('${q.id}','errors')">🔄 Retry ${errCount} lỗi</button>`;
            }

            // Config panel
            const c=q.config;
            const cfgHtml=q.is_running?'':`
                <div class="config-panel">
                    <b>⚙️ Config:</b><br>
                    <span class="cb"><input type="checkbox" ${c.mask_classes.includes('text')?'checked':''} onchange="toggleClass('${q.id}','text',this.checked)">text</span>
                    <span class="cb"><input type="checkbox" ${c.mask_classes.includes('sfx')?'checked':''} onchange="toggleClass('${q.id}','sfx',this.checked)">sfx</span>
                    <span class="cb"><input type="checkbox" ${c.mask_classes.includes('b1')?'checked':''} onchange="toggleClass('${q.id}','b1',this.checked)">b1</span>
                    &nbsp;
                    <label>Expand <input type="number" value="${c.expand}" onchange="setCfg('${q.id}','expand',+this.value)"></label>
                    <label>Feather <input type="number" value="${c.feather}" onchange="setCfg('${q.id}','feather',+this.value)"></label>
                    <label>Blur <input type="number" value="${c.blur}" onchange="setCfg('${q.id}','blur',+this.value)"></label>
                    <div class="mode-row" style="margin-top:8px;">
                        <span style="color:var(--text2)">Mode:</span>
                        <button class="${mode==='all'?'sel':''}" onclick="setMode('${q.id}','all')">Tất cả</button>
                        <button class="${mode==='errors'?'sel':''}" onclick="setMode('${q.id}','errors')">Chỉ lỗi (${errCount})</button>
                        <button class="${mode==='range'?'sel':''}" onclick="setMode('${q.id}','range')">Khoảng</button>
                    </div>
                </div>`;

            const pagesHtml=q.pages.map(p=>{
                let cls='pg-pending';
                if(p.error)cls='pg-error';
                else if(p.has_mask)cls='pg-done';
                else if(!p.has_clean)cls='pg-noclean';
                else if(q.is_running&&q.current_page===p.display_order+1)cls='pg-active';
                return `<div class="pg ${cls}" title="${p.display_name}${p.error?' ⚠️'+p.error:''}">${p.display_order+1}</div>`;
            }).join('');

            return `<div class="queue-item">
                <div class="qi-header">
                    <div>
                        <span class="name">${q.is_running?'🟢':'⚪'} ${q.project_name}</span>
                        <span style="color:var(--text2);font-size:13px;"> ${q.done}/${q.total} masks</span>
                        ${noClean?`<span style="color:var(--yellow);font-size:12px;"> (${noClean} chưa clean)</span>`:''}
                    </div>
                    <div class="qi-actions">${actHtml}</div>
                </div>
                ${cfgHtml}
                <div class="pages-bar">${pagesHtml}</div>
            </div>`;
        }).join('');
    }

    function setMode(id,mode){projectModes[id]=mode;refreshQueue();}

    // Config updates (debounced)
    const cfgCache={};
    function toggleClass(id,cls,on){
        if(!cfgCache[id])cfgCache[id]={};
        // Get current from DOM... just send full config
        fetch(`/api/queue`).then(r=>r.json()).then(d=>{
            const item=d.queue.find(q=>q.id===id);
            if(!item)return;
            let mc=[...item.config.mask_classes];
            if(on&&!mc.includes(cls))mc.push(cls);
            if(!on)mc=mc.filter(c=>c!==cls);
            const cfg={...item.config,mask_classes:mc};
            fetch(`/api/config/${id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:cfg})}).then(()=>refreshQueue());
        });
    }
    function setCfg(id,key,val){
        fetch(`/api/queue`).then(r=>r.json()).then(d=>{
            const item=d.queue.find(q=>q.id===id);
            if(!item)return;
            const cfg={...item.config,[key]:val};
            fetch(`/api/config/${id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:cfg})}).then(()=>refreshQueue());
        });
    }

    async function doStart(id){
        const mode=projectModes[id]||'all';
        const body={mode};
        if(mode==='range'){
            const s=prompt('Trang bắt đầu (1-indexed):','1'); if(!s)return;
            const e=prompt('Trang kết thúc (1-indexed):','999'); if(!e)return;
            body.range_start=parseInt(s)-1; body.range_end=parseInt(e);
        }
        const res=await fetch(`/api/start/${id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
        if(!res.ok){const d=await res.json();toast('❌ '+d.detail);return;}
        toast('▶ Started!');refreshQueue();
    }
    async function doPause(){await fetch('/api/pause',{method:'POST'});toast('⏸ Đang pause...');refreshQueue();}
    async function doClean(id){
        if(!confirm('Xóa toàn bộ mask + detections? Không thể hoàn tác!'))return;
        await fetch(`/api/clean/${id}`,{method:'POST'});
        toast('🗑️ Đã clean data');refreshQueue();
    }
    async function doRemove(id){
        if(!confirm('Xóa khỏi hàng chờ?'))return;
        await fetch(`/api/queue/${id}`,{method:'DELETE'});
        toast('❌ Đã xóa');refreshQueue();
    }

    refreshQueue();
    setInterval(refreshQueue,3000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


if __name__ == "__main__":
    print("🔍 Aniga3 Chapter Processor — http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
