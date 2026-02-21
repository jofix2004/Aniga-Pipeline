# local_app.py — Aniga Local Manager (FastAPI + Embedded SPA)
# Kiến trúc: Mở .aniga = extract ra disk → serve static file → nhanh như PTS

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
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import sys
import threading
import traceback
from PIL import Image

import chapter_manager as cm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
WORKING_DIR = os.path.join(BASE_DIR, "_working")
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ============================================================
# Cấu hình Đường dẫn Aniga3 Core
# ============================================================
ANIGA3_DIR = os.path.abspath(os.path.join(BASE_DIR, "core_aniga3"))
if ANIGA3_DIR not in sys.path:
    sys.path.append(ANIGA3_DIR)

# (Lazy Load sẽ được gọi bên trong khi bắt đầu xử lý Aniga3)

app = FastAPI(title="Aniga Project Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount thư mục _working để serve static files
app.mount("/static", StaticFiles(directory=WORKING_DIR), name="static")

# ============================================================
# HELPERS
# ============================================================

def _get_project_list():
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
                })
            except Exception as e:
                projects.append({"filename": f, "error": str(e)})
    return projects


def _get_project_path(filename):
    fpath = os.path.join(PROJECTS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy: {filename}")
    return fpath


def _extract_project(fpath):
    """Giải nén .aniga ra _working/{project_id}/ — serve trực tiếp từ disk."""
    manifest = cm.read_manifest(fpath)
    pid = manifest["project_id"]
    work_dir = os.path.join(WORKING_DIR, pid)

    # Xóa thư mục cũ nếu có
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    with zipfile.ZipFile(fpath, 'r') as zf:
        zf.extractall(work_dir)

    return pid, manifest


# ============================================================
# ANIGA3 WORKER THREAD
# ============================================================

class Aniga3State:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.active_pid = None
        self.current_page = 0
        self.total_pages = 0
        self.done_pages = 0
        self.status_msg = "Sẵn sàng"
        self.worker_thread = None

aniga3_state = Aniga3State()

def _aniga3_worker_loop(pid, manifest_path, mode, config_overrides):
    global aniga3_state
    import traceback
    import json as _json
    
    # --- LAZY LOAD CORE ANIGA3 ---
    aniga3_state.status_msg = "Đang nạp Model (VRAM)..."
    try:
        import core as aniga3_core
        import config as aniga3_config
    except Exception as e:
        err_msg = f"Lỗi nạp Core Aniga3: {e}"
        print(f"❌ {err_msg}")
        traceback.print_exc()
        aniga3_state.status_msg = err_msg
        aniga3_state.is_running = False
        aniga3_state.active_pid = None
        return
    # ----------------------------

    def _read_manifest_json(path):
        """Đọc manifest.json thuần trong thư mục _working (không phải zip)."""
        with open(path, 'r', encoding='utf-8') as f:
            return _json.load(f)

    def _write_manifest_json(path, manifest):
        """Ghi lại manifest.json vào thư mục _working."""
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(manifest, f, ensure_ascii=False, indent=2)

    try:
        working_dir = os.path.dirname(manifest_path)
        manifest = _read_manifest_json(manifest_path)
        
        # Thiết lập config runtime
        default_conf = {
            "mask_classes": ["text", "text2", "b1", "b2", "b3", "b4", "b5"],
            "expand": 31, "feather": 21, "blur": 5, "min_area": 100,
            "cleanup": 7, "overlap_threshold": 0.1, "confidence_threshold": 0.05,
            "ocr_mode": "Không bật"
        }
        config = {**default_conf, **(manifest.get("aniga3_config") or {}), **config_overrides}
        
        mask_params = {
            "blur": config["blur"], "min_area": config["min_area"],
            "cleanup": config["cleanup"], "overlap_threshold": config["overlap_threshold"],
            "expand": config["expand"], "feather": config["feather"],
        }
        aniga3_config.CONFIG['single_model_defaults']['conf_threshold'] = config["confidence_threshold"]
        
        # Phân loại trang cần chạy
        if mode == "errors":
            pending = [p for p in manifest["pages"] if p.get("error") and not p.get("has_mask", False)]
        elif mode == "all_force":
            pending = manifest["pages"]
        else: # "resume"
            pending = [p for p in manifest["pages"] if not p.get("has_mask", False) and not p.get("error")]
            if not pending:  # Tự động chạy lại trang lỗi nếu đã hết trang mới
                pending = [p for p in manifest["pages"] if p.get("error") and not p.get("has_mask", False)]
            
        pending = sorted(pending, key=lambda p: p["display_order"])
        aniga3_state.total_pages = len(pending)
        aniga3_state.done_pages = 0
        
        if not pending:
            aniga3_state.status_msg = "Không có trang nào cần chạy."
            return

        for idx, page in enumerate(pending):
            if aniga3_state.should_stop:
                aniga3_state.status_msg = "Đã dừng bởi người dùng."
                break
                
            hid = page["hidden_id"]
            aniga3_state.current_page = page["display_order"] + 1
            aniga3_state.status_msg = f"Đang xử lý trang {aniga3_state.current_page}..."
            
            clean_path = cm.get_page_from_dir(working_dir, hid, "clean")
            if not clean_path:
                aniga3_state.status_msg = f"Bỏ qua trang {aniga3_state.current_page}: Thiếu Clean"
                continue
                
            raw_path = cm.get_page_from_dir(working_dir, hid, "raw")
            try:
                clean_img = Image.open(clean_path).convert("RGB")
                raw_img = Image.open(raw_path).convert("RGB") if raw_path else None
                
                # --- AUTO-PADDING RAW IMAGE TO MATCH CLEAN IMAGE ---
                # ImgCraft thường trả về clean_img size 2048x2048 (đã scale+padding viền trắng)
                # nhưng raw_img lại giữ nguyên kích thước gốc lạc quẻ.
                if raw_img and raw_img.size != clean_img.size:
                    def _pad_image_to_match(img_pil, target_size):
                        # Bắt chước logic center_image của ImgCraft
                        canvas_size = target_size[0] # Giả định vuông 2048
                        margin = 60
                        original_width, original_height = img_pil.size
                        max_size = canvas_size - (margin * 2)
                        scale = min(max_size / original_width, max_size / original_height)
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                        
                        img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        canvas = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
                        
                        x = (canvas_size - new_width) // 2
                        y = (canvas_size - new_height) // 2
                        
                        if img_resized.mode == 'RGBA':
                            canvas.paste(img_resized, (x, y), img_resized)
                        else:
                            canvas.paste(img_resized, (x, y))
                        return canvas, [x, y, x + new_width, y + new_height]

                    # Thực hiện auto-pad và lưu đè trực tiếp vào file raw_path để đồng bộ
                    raw_img, content_bbox = _pad_image_to_match(raw_img, clean_img.size)
                    raw_img.save(raw_path) 
                    
                    # Update content_bbox cho manifest nếu thiếu
                    manifest = _read_manifest_json(manifest_path)
                    for p in manifest["pages"]:
                        if p["hidden_id"] == hid:
                            p["content_bbox"] = content_bbox
                    _write_manifest_json(manifest_path, manifest)
                            
                    print(f"🔄 Auto-padded raw_img for page {hid} to match clean_img size {clean_img.size}")
                # ---------------------------------------------------
                
                # Gọi Core
                result = aniga3_core.run_full_pipeline(
                    raw_image_pil=raw_img, clean_image_pil=clean_img,
                    device_mode="Auto", ocr_mode=config["ocr_mode"],
                    mask_classes=config["mask_classes"], mask_params=mask_params
                )
                
                mask_pil = Image.fromarray(result["final_mask"])
                detections = {"boxes": result["bbox_json"], "logs": result["logs"]}
                
                # Lưu đè vào _working
                cm.save_page_to_dir(working_dir, hid, "mask", mask_pil)
                cm.save_detections_to_dir(working_dir, hid, detections)
                
                # Cập nhật manifest.json trong _working
                manifest = _read_manifest_json(manifest_path)
                cm.mark_page_done(manifest, hid, "mask")
                cm.mark_page_done(manifest, hid, "detections")
                for p in manifest["pages"]:
                    if p["hidden_id"] == hid:
                        p["error"] = None
                        break
                _write_manifest_json(manifest_path, manifest)
                
                aniga3_state.done_pages += 1
            except Exception as e:
                manifest = _read_manifest_json(manifest_path)
                cm.mark_page_done(manifest, hid, "mask", error=str(e))
                _write_manifest_json(manifest_path, manifest)
                aniga3_state.status_msg = f"Lỗi trang {aniga3_state.current_page}: {str(e)}"
                print(f"❌ Aniga3 Lỗi trang {hid}:\n{traceback.format_exc()}")
        
        if not aniga3_state.should_stop:
            aniga3_state.status_msg = "Hoàn thành."
            
        # Tự động pack lại .aniga dự án khi hoàn thành
        target_fpath = None
        for f in os.listdir(PROJECTS_DIR):
            if f.endswith(".aniga"):
                try:
                    m = cm.read_manifest(os.path.join(PROJECTS_DIR, f))
                    if m["project_id"] == pid:
                        target_fpath = os.path.join(PROJECTS_DIR, f)
                        break
                except: pass
                
        if target_fpath:
            aniga3_state.status_msg = "Đang Pack lưu dự án..."
            cm.pack_to_bundle(working_dir, target_fpath)
            aniga3_state.status_msg = "Hoàn thành & Đã lưu file .aniga gốc."
            
    except Exception as e:
        aniga3_state.status_msg = f"Lỗi nghiêm trọng: {str(e)}"
        print(f"❌ Aniga3 Fatal Error:\n{traceback.format_exc()}")
    finally:
        aniga3_state.is_running = False
        aniga3_state.active_pid = None


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
# API: Mở dự án (extract ra disk)
# ============================================================

@app.post("/api/projects/{filename}/open")
def open_project(filename: str):
    """Mở dự án = extract .aniga → disk. Trả về metadata + static URL paths."""
    fpath = _get_project_path(filename)
    pid, manifest = _extract_project(fpath)

    pages_info = []
    for page in sorted(manifest["pages"], key=lambda p: p["display_order"]):
        display_name = cm.get_display_name(manifest["project_name"], page["display_order"])
        hid = page["hidden_id"]
        p = {
            **page,
            "display_name": display_name,
            "urls": {
                "raw": f"/static/{pid}/pages/{hid}/raw.png",
            }
        }
        if page.get("has_clean"):
            p["urls"]["clean"] = f"/static/{pid}/pages/{hid}/clean.png"
        if page.get("has_mask"):
            p["urls"]["mask"] = f"/static/{pid}/pages/{hid}/mask.png"
        if page.get("has_detections"):
            p["urls"]["detections"] = f"/static/{pid}/pages/{hid}/detections.json"
        pages_info.append(p)

    return {
        "filename": filename,
        "project_id": pid,
        "project_name": manifest["project_name"],
        "page_count": len(manifest["pages"]),
        "pages": pages_info,
        "imgcraft_config": manifest.get("imgcraft_config", {"flux_size": 1280}),
        "aniga3_config": manifest.get("aniga3_config"),
    }


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
# API: Aniga3 Tích hợp (Auto-Mask)
# ============================================================

@app.post("/api/aniga3/{filename}/start")
async def start_aniga3(filename: str, request: Request):
    global aniga3_state
    if aniga3_state.is_running:
        raise HTTPException(400, "Aniga3 đang chạy một tiến trình khác.")
        
    import json
    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode('utf-8'))
    except Exception as e:
        body = {}
    mode = body.get("mode", "resume")
    config_overrides = body.get("config", {})
    
    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)
    pid = manifest["project_id"]
    working_dir = os.path.join(WORKING_DIR, pid)
    
    if not os.path.exists(working_dir):
        # Fallback extract nếu chưa mở
        _extract_project(fpath)
        
    manifest_path = os.path.join(working_dir, "manifest.json")
    
    aniga3_state.is_running = True
    aniga3_state.should_stop = False
    aniga3_state.active_pid = pid
    aniga3_state.status_msg = "Đang nạp Model (VRAM)..."
    
    aniga3_state.worker_thread = threading.Thread(
        target=_aniga3_worker_loop, 
        args=(pid, manifest_path, mode, config_overrides), 
        daemon=True
    )
    aniga3_state.worker_thread.start()
    
    return {"status": "started", "pid": pid}

@app.post("/api/aniga3/{filename}/stop")
def stop_aniga3(filename: str):
    global aniga3_state
    if not aniga3_state.is_running:
        return {"status": "not_running"}
    aniga3_state.should_stop = True
    aniga3_state.status_msg = "Đang chờ dừng..."
    return {"status": "stopping"}

@app.get("/api/aniga3/{filename}/status")
def status_aniga3(filename: str):
    global aniga3_state
    return {
        "is_running": (aniga3_state.is_running),
        "active_pid": aniga3_state.active_pid,
        "is_stopping": aniga3_state.should_stop,
        "current_page": aniga3_state.current_page,
        "done_pages": aniga3_state.done_pages,
        "total_pages": aniga3_state.total_pages,
        "message": aniga3_state.status_msg,
        "is_available": True
    }


# ============================================================
# API: Xóa trang / Xóa dự án / Đổi tên / Reset
# ============================================================

@app.delete("/api/projects/{filename}/pages/{hidden_id}")
def delete_page(filename: str, hidden_id: str):
    fpath = _get_project_path(filename)
    manifest = cm.remove_page_from_bundle(fpath, hidden_id)
    return {"status": "ok", "page_count": len(manifest["pages"])}


@app.delete("/api/projects/{filename}")
def delete_project(filename: str):
    fpath = _get_project_path(filename)
    os.remove(fpath)
    return {"status": "ok"}


@app.post("/api/projects/{filename}/rename")
async def rename_project(filename: str, request: Request):
    body = await request.json()
    new_name = body.get("name", "").strip()
    if not new_name:
        raise HTTPException(400, "Tên không được trống")
    fpath = _get_project_path(filename)
    cm.rename_project(fpath, new_name)
    return {"status": "ok", "project_name": new_name}


@app.post("/api/projects/{filename}/pages/{hidden_id}/reset")
async def reset_page(filename: str, hidden_id: str, request: Request):
    body = await request.json()
    layers = body.get("layers", ["clean", "mask", "detections"])
    fpath = _get_project_path(filename)
    cm.reset_page(fpath, hidden_id, layers)
    return {"status": "ok"}


# ============================================================
# API: Lưu mask đã sửa (từ brush/eraser)
# ============================================================

@app.post("/api/projects/{filename}/pages/{hidden_id}/save-mask")
async def save_mask(filename: str, hidden_id: str, request: Request):
    """Nhận mask PNG base64, lưu ra _working dir + cập nhật .aniga."""
    body = await request.json()
    b64 = body.get("mask_base64", "")
    if not b64:
        raise HTTPException(400, "Thiếu mask_base64")

    import base64
    # Strip header nếu có
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    mask_bytes = base64.b64decode(b64)

    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)
    pid = manifest["project_id"]

    # 1. Lưu ra _working dir
    working_mask = os.path.join(WORKING_DIR, pid, "pages", hidden_id, "mask.png")
    os.makedirs(os.path.dirname(working_mask), exist_ok=True)
    with open(working_mask, 'wb') as f:
        f.write(mask_bytes)

    # 2. Cập nhật vào .aniga ZIP
    with zipfile.ZipFile(fpath, 'r') as zf:
        existing = {name: zf.read(name) for name in zf.namelist()}
    existing[f"pages/{hidden_id}/mask.png"] = mask_bytes

    # Cập nhật has_mask trong manifest
    for page in manifest["pages"]:
        if page["hidden_id"] == hidden_id:
            page["has_mask"] = True
            break
    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(fpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing.items():
            zf.writestr(name, data)

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
        existing_files = {name: zf.read(name) for name in zf.namelist()}
    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')
    with zipfile.ZipFile(fpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)
    return {"status": "ok", "imgcraft_config": imgcraft_config}


# ============================================================
# API: Update từ file .aniga (merge) → re-extract
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
        return result
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(500, str(e))


# ============================================================
# API: Resolve (export ZIP)
# ============================================================

@app.get("/api/projects/{filename}/resolve")
def resolve_project(filename: str, layers: str = "raw,clean,mask,detections", crop_resize: bool = True):
    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)
    safe_name = manifest["project_name"].replace(" ", "_")
    include_layers = [l.strip() for l in layers.split(",") if l.strip()]
    tmp_dir = tempfile.mkdtemp()
    try:
        resolve_dir = os.path.join(tmp_dir, safe_name)
        cm.resolve_bundle(fpath, resolve_dir, include_layers=include_layers, auto_crop_resize=crop_resize)
        zip_path = os.path.join(tmp_dir, f"{safe_name}.zip")
        shutil.make_archive(os.path.join(tmp_dir, safe_name), 'zip', resolve_dir)
        return FileResponse(zip_path, media_type="application/zip", filename=f"{safe_name}.zip")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.get("/api/projects/{filename}/download")
def download_bundle(filename: str):
    fpath = _get_project_path(filename)
    return FileResponse(fpath, media_type="application/octet-stream", filename=filename)




# ============================================================

# ============================================================
# HTML SPA
# ============================================================

app.mount("/web", StaticFiles(directory=os.path.join(BASE_DIR, "web")), name="web")

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(os.path.join(BASE_DIR, "web", "index.html"))

if __name__ == "__main__":
    print("🏠 Aniga Local — http://localhost:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
