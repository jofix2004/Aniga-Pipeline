# local_app.py — Aniga Local Manager (FastAPI + Embedded SPA)
# Quản lý dự án .aniga: tạo/mở/update/resolve/preview

import os
import io
import json
import uuid
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from functools import lru_cache

import chapter_manager as cm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)

app = FastAPI(title="Aniga Project Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# HELPERS
# ============================================================

def _get_project_list():
    """Quét thư mục projects/ để liệt kê các file .aniga."""
    projects = []
    for f in sorted(os.listdir(PROJECTS_DIR)):
        if f.endswith(".aniga"):
            fpath = os.path.join(PROJECTS_DIR, f)
            try:
                manifest = cm.read_manifest(fpath)
                clean_done, total = cm.get_progress(manifest, "clean")
                mask_done, _ = cm.get_progress(manifest, "mask")
                projects.append({
                    "filename": f,
                    "project_id": manifest["project_id"],
                    "project_name": manifest["project_name"],
                    "page_count": total,
                    "clean_progress": f"{clean_done}/{total}",
                    "mask_progress": f"{mask_done}/{total}",
                    "updated_at": manifest.get("updated_at", ""),
                })
            except Exception as e:
                projects.append({"filename": f, "error": str(e)})
    return projects


def _get_project_path(filename):
    fpath = os.path.join(PROJECTS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy: {filename}")
    return fpath

# ============================================================
# API: Danh sách dự án
# ============================================================

@app.get("/api/projects")
def list_projects():
    return _get_project_list()


# ============================================================
# API: Tạo dự án mới
# ============================================================

@app.post("/api/projects/create")
async def create_project(
    project_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    if not project_name.strip():
        raise HTTPException(400, "Tên dự án không được trống")
    if not files:
        raise HTTPException(400, "Cần ít nhất 1 ảnh")

    tmp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        for f in files:
            tmp_path = os.path.join(tmp_dir, f.filename)
            with open(tmp_path, 'wb') as out:
                content = await f.read()
                out.write(content)
            image_paths.append(tmp_path)

        safe_name = "".join(c for c in project_name if c.isalnum() or c in " _-").strip()
        output_filename = f"{safe_name}.aniga"
        output_path = os.path.join(PROJECTS_DIR, output_filename)

        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{safe_name}_{counter}.aniga"
            output_path = os.path.join(PROJECTS_DIR, output_filename)
            counter += 1

        manifest = cm.create_bundle(image_paths, project_name, output_path)

        return {
            "status": "ok",
            "filename": output_filename,
            "project_id": manifest["project_id"],
            "page_count": len(manifest["pages"])
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# API: Chi tiết dự án
# ============================================================

@app.get("/api/projects/{filename}")
def get_project_detail(filename: str):
    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)

    pages_info = []
    for page in sorted(manifest["pages"], key=lambda p: p["display_order"]):
        display_name = cm.get_display_name(manifest["project_name"], page["display_order"])
        pages_info.append({
            **page,
            "display_name": display_name,
        })

    return {
        "filename": filename,
        "project_id": manifest["project_id"],
        "project_name": manifest["project_name"],
        "page_count": len(manifest["pages"]),
        "pages": pages_info,
        "imgcraft_config": manifest.get("imgcraft_config", {"flux_size": 1280}),
        "aniga3_config": manifest.get("aniga3_config"),
        "updated_at": manifest.get("updated_at", ""),
    }


# ============================================================
# API: Preview ảnh — Cache & Preload
# ============================================================

# Server-side cache: key = "filepath:hidden_id:layer" → bytes
_image_cache = {}
_detections_cache = {}

def _preload_bundle(fpath):
    """Mở ZIP 1 lần, đọc TẤT CẢ ảnh + detections vào cache."""
    try:
        manifest = cm.read_manifest(fpath)
        with zipfile.ZipFile(fpath, 'r') as zf:
            namelist = set(zf.namelist())
            for page in manifest["pages"]:
                hid = page["hidden_id"]
                for layer in ("raw", "clean", "mask"):
                    arcname = f"pages/{hid}/{layer}.png"
                    if arcname in namelist:
                        key = f"{fpath}:{hid}:{layer}"
                        _image_cache[key] = zf.read(arcname)
                # Detections
                det_name = f"pages/{hid}/detections.json"
                if det_name in namelist:
                    _detections_cache[f"{fpath}:{hid}"] = json.loads(zf.read(det_name))
        return len(manifest["pages"])
    except Exception as e:
        print(f"⚠️ Preload error: {e}")
        return 0


@app.post("/api/projects/{filename}/preload")
def preload_project(filename: str):
    """Nạp toàn bộ ảnh vào RAM cache — gọi 1 lần khi mở dự án."""
    fpath = _get_project_path(filename)
    count = _preload_bundle(fpath)
    return {"status": "ok", "cached_pages": count}


# QUAN TRỌNG: route detections phải nằm TRƯỚC route {layer}
@app.get("/api/projects/{filename}/pages/{hidden_id}/detections")
def get_page_detections(filename: str, hidden_id: str):
    fpath = _get_project_path(filename)
    # Thử cache trước
    det_key = f"{fpath}:{hidden_id}"
    if det_key in _detections_cache:
        return _detections_cache[det_key]
    data = cm.get_detections_from_bundle(fpath, hidden_id)
    if data is None:
        raise HTTPException(404, "Không có detections cho page này")
    _detections_cache[det_key] = data
    return data


@app.get("/api/projects/{filename}/pages/{hidden_id}/{layer}")
def get_page_image(filename: str, hidden_id: str, layer: str):
    if layer not in ("raw", "clean", "mask"):
        raise HTTPException(400, "Layer phải là raw, clean, hoặc mask")

    fpath = _get_project_path(filename)
    key = f"{fpath}:{hidden_id}:{layer}"
    data = _image_cache.get(key)
    if data is None:
        # Fallback: đọc từ ZIP nếu chưa preload
        data = cm.get_page_from_bundle(fpath, hidden_id, layer)
        if data is not None:
            _image_cache[key] = data
    if data is None:
        raise HTTPException(404, f"Không tìm thấy {layer} cho page {hidden_id}")

    return StreamingResponse(io.BytesIO(data), media_type="image/png")


# ============================================================
# API: Thêm trang
# ============================================================

@app.post("/api/projects/{filename}/add-pages")
async def add_pages(filename: str, files: list[UploadFile] = File(...)):
    fpath = _get_project_path(filename)
    tmp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        for f in files:
            tmp_path = os.path.join(tmp_dir, f.filename)
            with open(tmp_path, 'wb') as out:
                content = await f.read()
                out.write(content)
            image_paths.append(tmp_path)

        manifest = cm.add_pages_to_bundle(fpath, image_paths)
        return {"status": "ok", "page_count": len(manifest["pages"])}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# API: Xóa trang
# ============================================================

@app.delete("/api/projects/{filename}/pages/{hidden_id}")
def delete_page(filename: str, hidden_id: str):
    fpath = _get_project_path(filename)
    manifest = cm.remove_page_from_bundle(fpath, hidden_id)
    return {"status": "ok", "page_count": len(manifest["pages"])}


# ============================================================
# API: Xóa dự án
# ============================================================

@app.delete("/api/projects/{filename}")
def delete_project(filename: str):
    fpath = _get_project_path(filename)
    os.remove(fpath)
    return {"status": "ok"}


# ============================================================
# API: Sắp xếp trang
# ============================================================

@app.post("/api/projects/{filename}/reorder")
async def reorder_pages(filename: str, request: Request):
    body = await request.json()
    order = body.get("order", [])
    if not order:
        raise HTTPException(400, "Cần danh sách hidden_id")

    fpath = _get_project_path(filename)
    manifest = cm.reorder_pages(fpath, order)
    return {"status": "ok"}


# ============================================================
# API: Đổi tên dự án
# ============================================================

@app.post("/api/projects/{filename}/rename")
async def rename_project(filename: str, request: Request):
    body = await request.json()
    new_name = body.get("name", "").strip()
    if not new_name:
        raise HTTPException(400, "Tên không được trống")

    fpath = _get_project_path(filename)
    manifest = cm.rename_project(fpath, new_name)
    return {"status": "ok", "project_name": new_name}


# ============================================================
# API: Reset trang
# ============================================================

@app.post("/api/projects/{filename}/pages/{hidden_id}/reset")
async def reset_page(filename: str, hidden_id: str, request: Request):
    body = await request.json()
    layers = body.get("layers", ["clean", "mask", "detections"])
    fpath = _get_project_path(filename)
    manifest = cm.reset_page(fpath, hidden_id, layers)
    return {"status": "ok"}


# ============================================================
# API: Cập nhật imgcraft_config
# ============================================================

@app.post("/api/projects/{filename}/imgcraft-config")
async def update_imgcraft_config(filename: str, request: Request):
    body = await request.json()
    flux_size = body.get("flux_size", 1280)
    if flux_size not in (1024, 1280, 1536):
        raise HTTPException(400, "flux_size phải là 1024, 1280, hoặc 1536")

    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)

    imgcraft_config = manifest.get("imgcraft_config", {})
    imgcraft_config["flux_size"] = flux_size
    manifest["imgcraft_config"] = imgcraft_config

    with zipfile.ZipFile(fpath, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            existing_files[name] = zf.read(name)

    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(fpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return {"status": "ok", "imgcraft_config": imgcraft_config}


# ============================================================
# API: Update từ file .aniga (merge)
# ============================================================

@app.post("/api/projects/{filename}/update")
async def update_project(filename: str, update_file: UploadFile = File(...)):
    fpath = _get_project_path(filename)

    tmp_path = os.path.join(PROJECTS_DIR, f"_update_{uuid.uuid4().hex[:8]}.aniga")
    try:
        with open(tmp_path, 'wb') as out:
            content = await update_file.read()
            out.write(content)

        result = cm.merge_bundles(fpath, tmp_path, delete_update=True)
        # Clear server cache (images + detections) cho file này
        keys_to_del = [k for k in _image_cache if k.startswith(fpath + ":")]
        for k in keys_to_del:
            _image_cache.pop(k, None)
        det_keys = [k for k in _detections_cache if k.startswith(fpath + ":")]
        for k in det_keys:
            _detections_cache.pop(k, None)
        # Re-preload file đã merge
        _preload_bundle(fpath)
        result["cache_bust"] = True
        return result
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(500, str(e))


# ============================================================
# API: Xuất sản phẩm (resolve) — Đa dạng
# ============================================================

@app.get("/api/projects/{filename}/resolve")
def resolve_project(filename: str, layers: str = "raw,clean,mask,detections"):
    """Xuất sản phẩm: .aniga → ZIP download.
    Tham số layers: raw,clean,mask,detections (phân tách dấu phẩy)
    """
    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)
    safe_name = manifest["project_name"].replace(" ", "_")

    include_layers = [l.strip() for l in layers.split(",") if l.strip()]

    tmp_dir = tempfile.mkdtemp()
    try:
        resolve_dir = os.path.join(tmp_dir, safe_name)
        cm.resolve_bundle(fpath, resolve_dir, include_layers=include_layers)

        zip_path = os.path.join(tmp_dir, f"{safe_name}.zip")
        shutil.make_archive(os.path.join(tmp_dir, safe_name), 'zip', resolve_dir)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{safe_name}.zip"
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


# ============================================================
# API: Download file .aniga gốc
# ============================================================

@app.get("/api/projects/{filename}/download")
def download_bundle(filename: str):
    fpath = _get_project_path(filename)
    return FileResponse(fpath, media_type="application/octet-stream", filename=filename)


# ============================================================
# HTML SPA
# ============================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aniga Project Manager</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0c0c12; --bg2: #14141f; --bg3: #1e1e2e; --bg4: #282840;
            --text: #e4e4f0; --text2: #7a7a9e; --text3: #52526e;
            --accent: #7c6ff7; --accent2: #5f52e0; --accent-glow: rgba(124,111,247,0.15);
            --green: #34d399; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;
            --border: #2a2a40; --radius: 12px;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter',system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

        /* ── TOPBAR ── */
        .topbar { background:linear-gradient(180deg,var(--bg2),var(--bg)); border-bottom:1px solid var(--border); padding:14px 28px; display:flex; align-items:center; gap:14px; position:sticky; top:0; z-index:50; backdrop-filter:blur(12px); }
        .topbar h1 { font-size:17px; font-weight:700; letter-spacing:-0.3px; }
        .back-btn { cursor:pointer; background:var(--bg3); border:1px solid var(--border); color:var(--text); padding:6px 14px; border-radius:8px; font-size:12px; display:none; transition:all .2s; }
        .back-btn:hover { border-color:var(--accent); background:var(--accent-glow); }

        .container { max-width:1200px; margin:0 auto; padding:24px; }

        /* ── HOME ── */
        .home-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }
        .home-header h2 { font-size:15px; color:var(--text2); font-weight:500; }
        .btn-create { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; border:none; padding:10px 22px; border-radius:10px; cursor:pointer; font-weight:600; font-size:13px; transition:all .2s; box-shadow:0 4px 20px var(--accent-glow); }
        .btn-create:hover { transform:translateY(-1px); box-shadow:0 6px 28px rgba(124,111,247,0.25); }

        .project-card { background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:16px 20px; margin-bottom:10px; cursor:pointer; transition:all .2s; display:flex; align-items:center; gap:16px; }
        .project-card:hover { border-color:var(--accent); background:var(--bg3); transform:translateX(4px); }
        .project-card .icon { font-size:28px; }
        .project-card .info { flex:1; }
        .project-card .name { font-size:15px; font-weight:600; }
        .project-card .meta { color:var(--text2); font-size:12px; margin-top:4px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
        .id-tag { background:var(--bg4); padding:2px 8px; border-radius:5px; font-family:'Courier New',monospace; font-size:11px; color:var(--text2); }
        .project-card .del-btn { background:none; border:none; color:var(--text3); font-size:18px; cursor:pointer; padding:4px 8px; border-radius:6px; transition:all .2s; }
        .project-card .del-btn:hover { color:var(--red); background:rgba(248,113,113,0.1); }

        /* ── DETAIL ── */
        .detail-layout { display:grid; grid-template-columns:280px 1fr; gap:24px; }
        @media (max-width:768px) { .detail-layout { grid-template-columns:1fr; } }

        .sidebar { background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:20px; height:fit-content; position:sticky; top:80px; }
        .sidebar h2 { font-size:16px; font-weight:700; margin-bottom:4px; word-break:break-word; }
        .sidebar .meta-line { font-size:12px; color:var(--text2); margin-bottom:16px; }

        .sidebar-section { margin-bottom:16px; }
        .sidebar-section h4 { font-size:11px; text-transform:uppercase; color:var(--text3); letter-spacing:1px; margin-bottom:8px; }
        .sidebar-btn { display:block; width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--text); padding:9px 14px; border-radius:8px; cursor:pointer; font-size:12px; text-align:left; margin-bottom:6px; transition:all .2s; }
        .sidebar-btn:hover { border-color:var(--accent); color:var(--accent); background:var(--accent-glow); }
        .sidebar-select { width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--text); padding:8px 10px; border-radius:8px; font-size:12px; cursor:pointer; margin-bottom:6px; }

        .content-area { min-height:400px; }
        .pages-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:10px; }
        .page-card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:8px; text-align:center; cursor:pointer; transition:all .2s; }
        .page-card:hover { border-color:var(--accent); transform:translateY(-2px); box-shadow:0 4px 16px rgba(0,0,0,0.3); }
        .page-card .thumb { width:100%; aspect-ratio:0.7; background:var(--bg); border-radius:6px; margin-bottom:6px; overflow:hidden; position:relative; }
        .page-card .thumb img { width:100%; height:100%; object-fit:cover; }
        .page-card .page-name { font-size:11px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .page-card .status { font-size:10px; display:flex; gap:4px; justify-content:center; margin-top:4px; }
        .page-card .status span { padding:1px 5px; border-radius:3px; font-weight:600; }
        .s-done { background:rgba(52,211,153,0.15); color:var(--green); }
        .s-pending { background:var(--bg4); color:var(--text3); }

        /* ── MODAL ── */
        .modal-overlay { position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.75); display:none; justify-content:center; align-items:center; z-index:100; backdrop-filter:blur(4px); }
        .modal-overlay.active { display:flex; }
        .modal { background:var(--bg2); border:1px solid var(--border); border-radius:16px; padding:28px; min-width:420px; max-width:90vw; max-height:90vh; overflow-y:auto; box-shadow:0 24px 64px rgba(0,0,0,0.5); }
        .modal h3 { margin-bottom:16px; font-size:16px; }
        .modal input[type="text"],.modal input[type="file"] { width:100%; padding:10px 14px; background:var(--bg3); border:1px solid var(--border); border-radius:8px; color:var(--text); margin-bottom:12px; font-size:13px; }
        .modal .btn-row { display:flex; gap:8px; justify-content:flex-end; margin-top:16px; }
        .modal .btn-row button { padding:8px 18px; border-radius:8px; border:none; cursor:pointer; font-size:13px; font-weight:600; transition:all .2s; }
        .btn-primary { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; }
        .btn-cancel { background:var(--bg3); color:var(--text); border:1px solid var(--border)!important; }
        .btn-danger { background:rgba(248,113,113,0.1); color:var(--red); border:1px solid rgba(248,113,113,0.3)!important; }

        /* ── PREVIEW MODAL (full screen) ── */
        .preview-modal .modal { min-width:90vw; max-width:96vw; min-height:85vh; max-height:95vh; display:flex; flex-direction:column; padding:16px 20px; overflow:hidden; }
        .preview-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
        .preview-header h3 { margin:0; font-size:15px; }
        .preview-nav { display:flex; gap:6px; }
        .preview-nav button { background:var(--bg3); border:1px solid var(--border); color:var(--text2); width:32px; height:32px; border-radius:8px; cursor:pointer; font-size:16px; transition:all .2s; }
        .preview-nav button:hover { border-color:var(--accent); color:var(--accent); }

        .layer-bar { display:flex; gap:5px; margin-bottom:8px; flex-wrap:wrap; align-items:center; }
        .layer-btn { background:var(--bg3); border:1px solid var(--border); color:var(--text2); padding:6px 14px; border-radius:8px; cursor:pointer; font-size:11px; font-weight:700; transition:all .15s; letter-spacing:0.3px; }
        .layer-btn.active { border-color:var(--accent); color:#fff; background:var(--accent); }
        .layer-btn:hover { border-color:var(--accent); }
        .layer-bar .sep { width:1px; height:20px; background:var(--border); margin:0 4px; }
        .layer-bar .zoom-info { margin-left:auto; font-size:11px; color:var(--text3); font-weight:600; }

        .preview-body { flex:1; display:flex; gap:12px; min-height:0; overflow:hidden; }
        .canvas-wrap { flex:1; background:var(--bg); border-radius:10px; overflow:hidden; position:relative; cursor:grab; }
        .canvas-wrap:active { cursor:grabbing; }
        .canvas-wrap canvas { position:absolute; top:0; left:0; image-rendering:auto; }
        .canvas-wrap .no-data { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:var(--text3); font-size:14px; }

        .json-panel { width:300px; background:var(--bg3); border-radius:10px; padding:12px; overflow-y:auto; font-size:11px; font-family:'Courier New',monospace; color:var(--text2); white-space:pre-wrap; word-break:break-all; flex-shrink:0; }

        .preview-footer { display:flex; gap:8px; margin-top:8px; justify-content:space-between; align-items:center; }
        .preview-footer .left { display:flex; gap:6px; }
        .bbox-tooltip { position:absolute; background:rgba(0,0,0,0.85); color:#fff; padding:6px 10px; border-radius:6px; font-size:12px; pointer-events:none; z-index:10; max-width:300px; white-space:pre-wrap; display:none; }

        /* ── TOAST ── */
        .toast { position:fixed; bottom:24px; right:24px; background:var(--bg2); border:1px solid var(--accent); color:var(--text); padding:12px 22px; border-radius:10px; font-size:13px; z-index:200; display:none; animation:slideIn .3s; box-shadow:0 8px 32px rgba(0,0,0,0.3); }
        @keyframes slideIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

        .loading { text-align:center; padding:40px; color:var(--text3); }

        /* ── Export Modal ── */
        .export-cb { display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; cursor:pointer; }
        .export-cb input { accent-color:var(--accent); }
    </style>
</head>
<body>
    <div class="topbar">
        <button class="back-btn" id="backBtn" onclick="goHome()">⬅ Quay lại</button>
        <h1>🏠 Aniga Project Manager</h1>
    </div>

    <div class="container">
        <div id="homePage">
            <div class="home-header">
                <h2>Danh sách dự án</h2>
                <button class="btn-create" onclick="showCreateModal()">+ Tạo dự án mới</button>
            </div>
            <div id="projectList"><div class="loading">Đang tải...</div></div>
        </div>

        <div id="detailPage" style="display:none;">
            <div class="detail-layout">
                <div class="sidebar">
                    <h2 id="sidebarName"></h2>
                    <div class="meta-line" id="sidebarMeta"></div>

                    <div class="sidebar-section">
                        <h4>Thao tác</h4>
                        <button class="sidebar-btn" onclick="showAddPagesModal()">➕ Thêm trang</button>
                        <button class="sidebar-btn" onclick="showRenameModal()">✏️ Đổi tên</button>
                        <button class="sidebar-btn" onclick="showUpdateModal()">🔄 Cập nhật từ .aniga</button>
                    </div>

                    <div class="sidebar-section">
                        <h4>Xuất / Tải</h4>
                        <button class="sidebar-btn" onclick="showExportModal()">📤 Xuất sản phẩm (ZIP)</button>
                        <button class="sidebar-btn" onclick="downloadBundle()">📥 Tải file .aniga</button>
                    </div>

                    <div class="sidebar-section">
                        <h4>Cấu hình</h4>
                        <label style="font-size:12px;color:var(--text2);margin-bottom:4px;display:block;">Flux Size</label>
                        <select class="sidebar-select" id="fluxSizeSelect" onchange="updateFluxSize(this.value)">
                            <option value="1024">1024</option>
                            <option value="1280" selected>1280</option>
                            <option value="1536">1536</option>
                        </select>
                    </div>
                </div>

                <div class="content-area">
                    <div class="pages-grid" id="pagesGrid"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Create Modal -->
    <div class="modal-overlay" id="createModal">
        <div class="modal">
            <h3>Tạo dự án mới</h3>
            <input type="text" id="createName" placeholder="Tên dự án (vd: Naruto Ch.001)">
            <input type="file" id="createFiles" multiple accept="image/*">
            <div id="createStatus" style="color:var(--text2);font-size:12px;margin-bottom:8px;"></div>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="createProject()">Tạo</button>
            </div>
        </div>
    </div>

    <!-- Update Modal -->
    <div class="modal-overlay" id="updateModal">
        <div class="modal">
            <h3>🔄 Cập nhật từ file .aniga</h3>
            <p style="color:var(--text2);margin-bottom:12px;font-size:12px;">Chọn file .aniga đã xử lý từ Colab để merge vào dự án gốc.</p>
            <input type="file" id="updateFile" accept=".aniga">
            <div id="updateStatus" style="color:var(--text2);font-size:12px;"></div>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="updateProject()">Cập nhật</button>
            </div>
        </div>
    </div>

    <!-- Rename Modal -->
    <div class="modal-overlay" id="renameModal">
        <div class="modal">
            <h3>✏️ Đổi tên dự án</h3>
            <input type="text" id="renameName" placeholder="Tên mới">
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="renameProject()">Lưu</button>
            </div>
        </div>
    </div>

    <!-- Add Pages Modal -->
    <div class="modal-overlay" id="addPagesModal">
        <div class="modal">
            <h3>➕ Thêm trang</h3>
            <input type="file" id="addPagesFiles" multiple accept="image/*">
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="addPages()">Thêm</button>
            </div>
        </div>
    </div>

    <!-- Export Modal -->
    <div class="modal-overlay" id="exportModal">
        <div class="modal">
            <h3>📤 Xuất sản phẩm (ZIP)</h3>
            <p style="color:var(--text2);margin-bottom:14px;font-size:12px;">Chọn các layers muốn xuất:</p>
            <label class="export-cb"><input type="checkbox" value="raw" checked> Ảnh Gốc (RAW)</label>
            <label class="export-cb"><input type="checkbox" value="clean" checked> Ảnh Clean (ImgCraft)</label>
            <label class="export-cb"><input type="checkbox" value="mask" checked> Mask (Aniga3)</label>
            <label class="export-cb"><input type="checkbox" value="detections" checked> JSON BBox & OCR</label>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="doExport()">Xuất</button>
            </div>
        </div>
    </div>

    <!-- Preview Modal -->
    <div class="modal-overlay preview-modal" id="previewModal">
        <div class="modal">
            <div class="preview-header">
                <h3 id="previewTitle">Preview</h3>
                <div class="preview-nav">
                    <button onclick="prevPage()" title="Trang trước">◀</button>
                    <button onclick="nextPage()" title="Trang sau">▶</button>
                    <button onclick="resetZoom()" title="Reset zoom">1:1</button>
                    <button onclick="closeModals()" title="Đóng" style="color:var(--red);">✕</button>
                </div>
            </div>
            <div class="layer-bar" id="layerBar"></div>
            <div class="preview-body">
                <div class="canvas-wrap" id="canvasWrap">
                    <canvas id="mainCanvas"></canvas>
                    <div class="bbox-tooltip" id="bboxTooltip"></div>
                    <span class="no-data" id="canvasMsg">Chọn trang để xem</span>
                </div>
                <div class="json-panel" id="jsonPanel" style="display:none;"></div>
            </div>
            <div class="preview-footer">
                <div class="left">
                    <button class="btn-cancel btn-danger" onclick="resetCurrentPage()">🔄 Reset</button>
                    <button class="btn-cancel btn-danger" onclick="deleteCurrentPage()">🗑️ Xóa trang</button>
                </div>
                <div id="zoomLabel" style="font-size:11px;color:var(--text3);"></div>
            </div>
        </div>
    </div>

    <!-- Toast -->
    <div class="toast" id="toast"></div>

    <script>
    let currentFile = null;
    let currentProject = null;
    let previewPageId = null;
    let previewPageIndex = -1;
    let currentDetections = null;
    // Cache ảnh theo hidden_id + layer → Image object (tránh tải lại từ ZIP)
    const imageCache = {};
    function getCacheKey(hid, layer) { return hid + '/' + layer; }

    // Màu cho từng class (giống Aniga3)
    const CLASS_COLORS = {
        text:'#00ff00', text2:'#88ff88', sfx:'#ff00ff',
        b1:'#ff0000', b2:'#0000ff', b3:'#ffff00', b4:'#00ffff', b5:'#ff8800'
    };

    // ── UTILS ──
    function toast(msg, dur=3000) {
        const t = document.getElementById('toast');
        t.textContent = msg; t.style.display = 'block';
        clearTimeout(t._tid);
        t._tid = setTimeout(() => t.style.display='none', dur);
    }
    function closeModals() {
        document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
    }

    // ── HOME ──
    async function loadProjects() {
        const res = await fetch('/api/projects');
        const projects = await res.json();
        const el = document.getElementById('projectList');
        if (!projects.length) { el.innerHTML = '<div class="loading">Chưa có dự án nào</div>'; return; }
        el.innerHTML = projects.map(p => p.error
            ? `<div class="project-card"><div class="icon">⚠️</div><div class="info"><span class="name">${p.filename}</span><div class="meta">${p.error}</div></div></div>`
            : `<div class="project-card" onclick="openProject('${p.filename}')">
                <div class="icon">📁</div>
                <div class="info">
                    <span class="name">${p.project_name}</span>
                    <div class="meta">
                        <span class="id-tag">${p.project_id}</span>
                        <span>${p.page_count} trang</span>
                        <span style="color:var(--green)">C:${p.clean_progress}</span>
                        <span style="color:var(--blue)">M:${p.mask_progress}</span>
                    </div>
                </div>
                <button class="del-btn" onclick="event.stopPropagation();deleteProject('${p.filename}')" title="Xóa">🗑️</button>
               </div>`
        ).join('');
    }

    async function deleteProject(fn) {
        if (!confirm('Xóa dự án này? Không thể hoàn tác!')) return;
        await fetch('/api/projects/'+fn, {method:'DELETE'});
        loadProjects();
        toast('✅ Đã xóa dự án');
    }

    function goHome() {
        document.getElementById('homePage').style.display = '';
        document.getElementById('detailPage').style.display = 'none';
        document.getElementById('backBtn').style.display = 'none';
        currentFile = null; currentProject = null;
        loadProjects();
    }

    // ── DETAIL ──
    async function openProject(filename) {
        currentFile = filename;
        // Preload tất cả ảnh vào RAM server-side trước
        fetch('/api/projects/'+filename+'/preload', {method:'POST'});
        const res = await fetch('/api/projects/'+filename);
        currentProject = await res.json();
        renderDetail();
        document.getElementById('homePage').style.display = 'none';
        document.getElementById('detailPage').style.display = '';
        document.getElementById('backBtn').style.display = '';
    }

    function renderDetail() {
        const p = currentProject;
        document.getElementById('sidebarName').textContent = p.project_name;
        document.getElementById('sidebarMeta').innerHTML = `<span class="id-tag">${p.project_id}</span> &nbsp; ${p.page_count} trang`;
        const fluxSize = (p.imgcraft_config && p.imgcraft_config.flux_size) || 1280;
        document.getElementById('fluxSizeSelect').value = fluxSize;

        const grid = document.getElementById('pagesGrid');
        grid.innerHTML = p.pages.map((pg, idx) => {
            const dn = pg.display_name.split('_').pop();
            return `<div class="page-card" onclick="showPreview(${idx})">
                <div class="thumb"><img src="/api/projects/${currentFile}/pages/${pg.hidden_id}/raw" loading="lazy" onerror="this.style.display='none'"></div>
                <div class="page-name">${dn}</div>
                <div class="status">
                    <span class="${pg.has_clean?'s-done':'s-pending'}">C</span>
                    <span class="${pg.has_mask?'s-done':'s-pending'}">M</span>
                    <span class="${pg.has_detections?'s-done':'s-pending'}">D</span>
                </div>
            </div>`;
        }).join('');
    }

    // ── PREVIEW (Zoom/Pan Canvas + Layer Toggle) ──
    let activeLayer = 'RAW';
    let layerImages = {};     // layer name -> Image or Canvas
    let pvZoom = 1, pvPanX = 0, pvPanY = 0;
    let pvDragging = false, pvDragStartX = 0, pvDragStartY = 0;
    let pvImgW = 0, pvImgH = 0;
    let pvBoxes = [];

    async function showPreview(idx) {
        previewPageIndex = idx;
        const pg = currentProject.pages[idx];
        previewPageId = pg.hidden_id;
        currentDetections = undefined;
        layerImages = {};
        pvBoxes = [];
        pvZoom = 1; pvPanX = 0; pvPanY = 0;

        document.getElementById('previewTitle').textContent = pg.display_name;
        document.getElementById('jsonPanel').style.display = 'none';
        document.getElementById('canvasMsg').style.display = '';
        document.getElementById('canvasMsg').textContent = 'Đang tải...';

        // Tạo layer bar
        const layers = ['RAW'];
        if (pg.has_clean) layers.push('CLEAN');
        if (pg.has_mask) layers.push('MASK');
        if (pg.has_clean && pg.has_mask) layers.push('BLEND');
        if (pg.has_detections) layers.push('BBOX');
        if (pg.has_detections) layers.push('JSON');

        document.getElementById('layerBar').innerHTML = layers.map(l =>
            `<button class="layer-btn${l==='RAW'?' active':''}" onclick="switchLayer('${l}',this)">${l}</button>`
        ).join('') + '<span class="sep"></span><span class="zoom-info" id="zoomInfo">100%</span>';

        activeLayer = 'RAW';
        document.getElementById('previewModal').classList.add('active');

        // Preload tất cả layers song song
        const promises = [cachedLoadImage(previewPageId, 'raw').then(img => { layerImages['RAW'] = img; pvImgW = img.width; pvImgH = img.height; })];
        if (pg.has_clean) promises.push(cachedLoadImage(previewPageId, 'clean').then(img => layerImages['CLEAN'] = img));
        if (pg.has_mask) promises.push(cachedLoadImage(previewPageId, 'mask').then(img => layerImages['MASK'] = img));
        if (pg.has_detections) promises.push(fetchDetections().then(d => { if(d && d.boxes) pvBoxes = d.boxes; }));
        await Promise.all(promises);

        // Tạo BLEND nếu có
        if (layerImages['RAW'] && layerImages['CLEAN'] && layerImages['MASK']) {
            layerImages['BLEND'] = buildBlend(layerImages['RAW'], layerImages['CLEAN'], layerImages['MASK']);
        }
        // Tạo BBOX canvas nếu có
        if (layerImages['RAW'] && pvBoxes.length) {
            layerImages['BBOX'] = buildBBoxCanvas(layerImages['RAW'], pvBoxes);
        }

        document.getElementById('canvasMsg').style.display = 'none';
        fitToView();
        renderCanvas();
    }

    function buildBlend(rawImg, cleanImg, maskImg) {
        const w = rawImg.width, h = rawImg.height;
        const c = document.createElement('canvas'); c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        ctx.drawImage(rawImg, 0, 0); const rd = ctx.getImageData(0,0,w,h);
        ctx.drawImage(cleanImg, 0, 0, w, h); const cd = ctx.getImageData(0,0,w,h);
        ctx.drawImage(maskImg, 0, 0, w, h); const md = ctx.getImageData(0,0,w,h);
        const out = ctx.createImageData(w, h);
        for (let i=0; i<rd.data.length; i+=4) {
            const m = md.data[i]/255;
            out.data[i]=rd.data[i]*(1-m)+cd.data[i]*m;
            out.data[i+1]=rd.data[i+1]*(1-m)+cd.data[i+1]*m;
            out.data[i+2]=rd.data[i+2]*(1-m)+cd.data[i+2]*m;
            out.data[i+3]=255;
        }
        ctx.putImageData(out, 0, 0);
        return c;
    }

    function buildBBoxCanvas(rawImg, boxes) {
        const w = rawImg.width, h = rawImg.height;
        const c = document.createElement('canvas'); c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        ctx.drawImage(rawImg, 0, 0);
        for (const item of boxes) {
            const [x1,y1,x2,y2] = item.bbox;
            const cls = item['class'];
            const color = CLASS_COLORS[cls] || '#ffffff';
            ctx.strokeStyle = color; ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2-x1, y2-y1);
            const label = `${cls} ${(item.confidence*100).toFixed(0)}%`;
            ctx.font = 'bold 14px Inter,sans-serif';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color; ctx.fillRect(x1, y1-20, tw+8, 20);
            ctx.fillStyle = '#000'; ctx.fillText(label, x1+4, y1-5);
        }
        return c;
    }

    function switchLayer(layer, btn) {
        document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');
        activeLayer = layer;
        const jp = document.getElementById('jsonPanel');
        if (layer === 'JSON') {
            jp.style.display = '';
            const dets = currentDetections;
            jp.textContent = dets ? JSON.stringify(dets, null, 2) : 'Chưa có dữ liệu';
        } else {
            jp.style.display = 'none';
        }
        renderCanvas();
    }

    function renderCanvas() {
        const canvas = document.getElementById('mainCanvas');
        const wrap = document.getElementById('canvasWrap');
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        canvas.width = ww; canvas.height = wh;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, ww, wh);

        const src = layerImages[activeLayer] || layerImages['RAW'];
        if (!src) return;

        ctx.save();
        ctx.translate(pvPanX, pvPanY);
        ctx.scale(pvZoom, pvZoom);
        ctx.drawImage(src, 0, 0);
        ctx.restore();

        document.getElementById('zoomInfo').textContent = `${Math.round(pvZoom*100)}%`;
        document.getElementById('zoomLabel').textContent = `${pvImgW}×${pvImgH}px  |  ${Math.round(pvZoom*100)}%`;
    }

    function fitToView() {
        const wrap = document.getElementById('canvasWrap');
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        if (!pvImgW || !pvImgH) return;
        pvZoom = Math.min(ww / pvImgW, wh / pvImgH, 1);
        pvPanX = (ww - pvImgW * pvZoom) / 2;
        pvPanY = (wh - pvImgH * pvZoom) / 2;
    }

    function resetZoom() {
        fitToView();
        renderCanvas();
    }

    // Zoom (scroll)
    document.addEventListener('wheel', (e) => {
        if (!document.getElementById('previewModal').classList.contains('active')) return;
        const wrap = document.getElementById('canvasWrap');
        if (!wrap.contains(e.target)) return;
        e.preventDefault();
        const rect = wrap.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const oldZoom = pvZoom;
        const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
        pvZoom = Math.max(0.1, Math.min(20, pvZoom * factor));
        // Zoom toàn bộ về phía con trỏ
        pvPanX = mx - (mx - pvPanX) * (pvZoom / oldZoom);
        pvPanY = my - (my - pvPanY) * (pvZoom / oldZoom);
        renderCanvas();
    }, {passive: false});

    // Pan (drag)
    (function(){
        const wrap = document.getElementById('canvasWrap');
        wrap.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            pvDragging = true;
            pvDragStartX = e.clientX - pvPanX;
            pvDragStartY = e.clientY - pvPanY;
        });
        window.addEventListener('mousemove', (e) => {
            if (!pvDragging) return;
            pvPanX = e.clientX - pvDragStartX;
            pvPanY = e.clientY - pvDragStartY;
            renderCanvas();
        });
        window.addEventListener('mouseup', () => { pvDragging = false; });
    })();

    // BBox hover tooltip
    document.getElementById('canvasWrap').addEventListener('mousemove', (e) => {
        if (activeLayer !== 'BBOX' || !pvBoxes.length) { document.getElementById('bboxTooltip').style.display = 'none'; return; }
        const wrap = document.getElementById('canvasWrap');
        const rect = wrap.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const imgX = (cx - pvPanX) / pvZoom;
        const imgY = (cy - pvPanY) / pvZoom;
        let found = null;
        for (const item of pvBoxes) {
            const [x1,y1,x2,y2] = item.bbox;
            if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) { found = item; break; }
        }
        const tip = document.getElementById('bboxTooltip');
        if (found && found.ocr_text) {
            tip.style.display = 'block';
            tip.style.left = (cx + 12) + 'px';
            tip.style.top = (cy - 8) + 'px';
            tip.textContent = `[${found['class']}] ${found.ocr_text}`;
        } else {
            tip.style.display = 'none';
        }
    });

    function prevPage() { if (previewPageIndex > 0) showPreview(previewPageIndex - 1); }
    function nextPage() { if (previewPageIndex < currentProject.pages.length - 1) showPreview(previewPageIndex + 1); }

    // Cache-aware image loader
    function cachedLoadImage(hid, layer) {
        const key = getCacheKey(hid, layer);
        if (imageCache[key]) return Promise.resolve(imageCache[key]);
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => { imageCache[key] = img; resolve(img); };
            img.onerror = reject;
            img.src = `/api/projects/${currentFile}/pages/${hid}/${layer}`;
        });
    }

    async function fetchDetections() {
        if (currentDetections !== undefined) return currentDetections;
        try {
            const res = await fetch(`/api/projects/${currentFile}/pages/${previewPageId}/detections`);
            if (!res.ok) { currentDetections = null; return null; }
            currentDetections = await res.json();
            return currentDetections;
        } catch(e) { currentDetections = null; return null; }
    }

    async function deleteCurrentPage() {
        if (!confirm('Xóa trang này?')) return;
        await fetch(`/api/projects/${currentFile}/pages/${previewPageId}`, {method:'DELETE'});
        closeModals(); openProject(currentFile);
        toast('✅ Đã xóa trang');
    }
    async function resetCurrentPage() {
        if (!confirm('Reset clean/mask/detections?')) return;
        await fetch(`/api/projects/${currentFile}/pages/${previewPageId}/reset`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({layers:['clean','mask','detections']})});
        closeModals(); openProject(currentFile);
        toast('✅ Đã reset trang');
    }

    // ── Keyboard nav ──
    document.addEventListener('keydown', (e) => {
        if (!document.getElementById('previewModal').classList.contains('active')) return;
        if (e.key === 'ArrowLeft') prevPage();
        else if (e.key === 'ArrowRight') nextPage();
        else if (e.key === 'Escape') closeModals();
    });

    // ── CREATE ──
    function showCreateModal() {
        document.getElementById('createName').value = '';
        document.getElementById('createFiles').value = '';
        document.getElementById('createStatus').textContent = '';
        document.getElementById('createModal').classList.add('active');
    }
    async function createProject() {
        const name = document.getElementById('createName').value.trim();
        const files = document.getElementById('createFiles').files;
        if (!name) { toast('⚠️ Nhập tên!'); return; }
        if (!files.length) { toast('⚠️ Chọn ảnh!'); return; }
        document.getElementById('createStatus').textContent = `Đang tạo... (${files.length} ảnh)`;
        const fd = new FormData();
        fd.append('project_name', name);
        for (const f of files) fd.append('files', f);
        const res = await fetch('/api/projects/create', {method:'POST', body:fd});
        const data = await res.json();
        closeModals(); loadProjects();
        toast(`✅ Đã tạo "${name}" (${data.page_count} trang)`);
    }

    // ── RENAME ──
    function showRenameModal() {
        document.getElementById('renameName').value = currentProject.project_name;
        document.getElementById('renameModal').classList.add('active');
    }
    async function renameProject() {
        const name = document.getElementById('renameName').value.trim();
        if (!name) return;
        await fetch(`/api/projects/${currentFile}/rename`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})});
        closeModals(); openProject(currentFile);
        toast('✅ Đã đổi tên');
    }

    // ── ADD PAGES ──
    function showAddPagesModal() { document.getElementById('addPagesModal').classList.add('active'); }
    async function addPages() {
        const files = document.getElementById('addPagesFiles').files;
        if (!files.length) return;
        const fd = new FormData();
        for (const f of files) fd.append('files', f);
        await fetch(`/api/projects/${currentFile}/add-pages`, {method:'POST', body:fd});
        closeModals(); openProject(currentFile);
        toast(`✅ Đã thêm ${files.length} trang`);
    }

    // ── UPDATE ──
    function showUpdateModal() {
        document.getElementById('updateFile').value = '';
        document.getElementById('updateStatus').textContent = '';
        document.getElementById('updateModal').classList.add('active');
    }
    async function updateProject() {
        const file = document.getElementById('updateFile').files[0];
        if (!file) { toast('⚠️ Chọn file .aniga!'); return; }
        document.getElementById('updateStatus').textContent = 'Đang merge...';
        const fd = new FormData();
        fd.append('update_file', file);
        const res = await fetch(`/api/projects/${currentFile}/update`, {method:'POST', body:fd});
        const data = await res.json();
        // Clear client cache để force reload ảnh mới
        if (data.cache_bust) {
            for (const key in imageCache) delete imageCache[key];
        }
        closeModals(); openProject(currentFile);
        if (data.errors?.length) toast(`⚠️ ${data.errors.join(', ')}`, 5000);
        else toast(`✅ Đã sync ${data.synced} trang`);
    }

    // ── EXPORT ──
    function showExportModal() { document.getElementById('exportModal').classList.add('active'); }
    function doExport() {
        const checks = document.querySelectorAll('#exportModal .export-cb input:checked');
        const layers = Array.from(checks).map(c => c.value).join(',');
        if (!layers) { toast('⚠️ Chọn ít nhất 1 layer!'); return; }
        window.location.href = `/api/projects/${currentFile}/resolve?layers=${layers}`;
        closeModals();
        toast('📤 Đang xuất...');
    }

    // ── FLUX SIZE ──
    async function updateFluxSize(val) {
        const res = await fetch(`/api/projects/${currentFile}/imgcraft-config`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({flux_size:parseInt(val)})
        });
        if (res.ok) toast(`✅ Flux Size → ${val}`);
        else toast('❌ Lỗi cập nhật');
    }

    // ── DOWNLOAD ──
    function downloadBundle() { window.location.href = `/api/projects/${currentFile}/download`; }

    // ── INIT ──
    loadProjects();
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_CONTENT


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🏠 Aniga Project Manager — http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
