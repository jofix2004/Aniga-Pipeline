# local_app.py — Aniga Local Manager (FastAPI + Embedded SPA)
# Quản lý dự án .aniga: tạo/mở/update/resolve/preview

import os
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

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
    """Trả về đường dẫn tuyệt đối tới file .aniga trong projects/."""
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
    """Upload ảnh + tạo file .aniga mới."""
    if not project_name.strip():
        raise HTTPException(400, "Tên dự án không được trống")
    if not files:
        raise HTTPException(400, "Cần ít nhất 1 ảnh")

    # Lưu ảnh tạm
    tmp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        for f in files:
            tmp_path = os.path.join(tmp_dir, f.filename)
            with open(tmp_path, 'wb') as out:
                content = await f.read()
                out.write(content)
            image_paths.append(tmp_path)

        # Tạo bundle
        safe_name = "".join(c for c in project_name if c.isalnum() or c in " _-").strip()
        output_filename = f"{safe_name}.aniga"
        output_path = os.path.join(PROJECTS_DIR, output_filename)

        # Tránh trùng tên
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
# API: Preview ảnh
# ============================================================

@app.get("/api/projects/{filename}/pages/{hidden_id}/{layer}")
def get_page_image(filename: str, hidden_id: str, layer: str):
    """Trả về ảnh PNG của 1 page (raw/clean/mask)."""
    if layer not in ("raw", "clean", "mask"):
        raise HTTPException(400, "Layer phải là raw, clean, hoặc mask")

    fpath = _get_project_path(filename)
    data = cm.get_page_from_bundle(fpath, hidden_id, layer)
    if data is None:
        raise HTTPException(404, f"Không tìm thấy {layer} cho page {hidden_id}")

    import io
    return StreamingResponse(io.BytesIO(data), media_type="image/png")


@app.get("/api/projects/{filename}/pages/{hidden_id}/detections")
def get_page_detections(filename: str, hidden_id: str):
    fpath = _get_project_path(filename)
    data = cm.get_detections_from_bundle(fpath, hidden_id)
    if data is None:
        raise HTTPException(404, "Không có detections cho page này")
    return data


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

    # Ghi lại manifest vào bundle
    import zipfile
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

    # Lưu file update tạm
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
# API: Xuất sản phẩm (resolve)
# ============================================================

@app.get("/api/projects/{filename}/resolve")
def resolve_project(filename: str):
    """Xuất sản phẩm: .aniga → ZIP download."""
    fpath = _get_project_path(filename)
    manifest = cm.read_manifest(fpath)
    safe_name = manifest["project_name"].replace(" ", "_")

    # Tạo thư mục tạm → resolve → zip → stream
    tmp_dir = tempfile.mkdtemp()
    try:
        resolve_dir = os.path.join(tmp_dir, safe_name)
        cm.resolve_bundle(fpath, resolve_dir)

        # Tạo ZIP
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
    <style>
        :root {
            --bg: #0f0f14; --bg2: #1a1a24; --bg3: #252535;
            --text: #e0e0ef; --text2: #8888aa;
            --accent: #7c6ff7; --accent2: #5a4fd4;
            --green: #4caf50; --red: #e74c3c; --yellow: #f39c12;
            --border: #2a2a3a; --radius: 10px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

        .topbar { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
        .topbar h1 { font-size: 20px; font-weight: 700; }
        .topbar .back-btn { cursor: pointer; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 6px; font-size: 13px; display: none; }
        .topbar .back-btn:hover { border-color: var(--accent); }

        .container { max-width: 1100px; margin: 0 auto; padding: 24px; }

        /* HOME */
        .home-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .btn-create { background: var(--accent); color: #fff; border: none; padding: 10px 20px; border-radius: var(--radius); cursor: pointer; font-weight: 600; font-size: 14px; }
        .btn-create:hover { background: var(--accent2); }

        .project-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 12px; cursor: pointer; transition: border-color 0.2s; }
        .project-card:hover { border-color: var(--accent); }
        .project-card .name { font-size: 16px; font-weight: 600; }
        .project-card .meta { color: var(--text2); font-size: 13px; margin-top: 6px; display: flex; gap: 16px; flex-wrap: wrap; }
        .project-card .id-tag { background: var(--bg3); padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 12px; }

        /* DETAIL */
        .detail-header { margin-bottom: 20px; }
        .detail-header h2 { font-size: 22px; margin-bottom: 6px; }
        .detail-header .meta { color: var(--text2); font-size: 13px; }
        .detail-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
        .detail-actions button { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
        .detail-actions button:hover { border-color: var(--accent); color: var(--accent); }

        .pages-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; }
        .page-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
        .page-card:hover { border-color: var(--accent); }
        .page-card .page-name { font-size: 13px; font-weight: 600; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .page-card .thumb { width: 100%; aspect-ratio: 1; background: var(--bg3); border-radius: 4px; margin-bottom: 6px; overflow: hidden; }
        .page-card .thumb img { width: 100%; height: 100%; object-fit: contain; }
        .page-card .status { font-size: 11px; display: flex; gap: 6px; justify-content: center; }
        .page-card .status span { padding: 1px 5px; border-radius: 3px; }
        .s-done { background: #1b5e20; color: #a5d6a7; }
        .s-pending { background: var(--bg3); color: var(--text2); }
        .s-error { background: #b71c1c; color: #ef9a9a; }

        /* MODAL */
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: none; justify-content: center; align-items: center; z-index: 100; }
        .modal-overlay.active { display: flex; }
        .modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; min-width: 400px; max-width: 90vw; max-height: 90vh; overflow-y: auto; }
        .modal h3 { margin-bottom: 16px; }
        .modal input[type="text"], .modal input[type="file"] { width: 100%; padding: 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; color: var(--text); margin-bottom: 12px; font-size: 14px; }
        .modal .btn-row { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
        .modal .btn-row button { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; font-size: 14px; }
        .modal .btn-primary { background: var(--accent); color: #fff; }
        .modal .btn-cancel { background: var(--bg3); color: var(--text); border: 1px solid var(--border) !important; }

        /* PREVIEW MODAL */
        .preview-tabs { display: flex; gap: 8px; margin-bottom: 12px; }
        .preview-tabs button { background: var(--bg3); border: 1px solid var(--border); color: var(--text2); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
        .preview-tabs button.active { border-color: var(--accent); color: var(--accent); }
        .preview-img { width: 100%; max-width: 600px; aspect-ratio: 1; background: var(--bg); border-radius: 8px; display: flex; justify-content: center; align-items: center; overflow: hidden; }
        .preview-img img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .preview-img .no-data { color: var(--text2); font-size: 14px; }

        /* TOAST */
        .toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 1px solid var(--accent); color: var(--text); padding: 12px 20px; border-radius: 8px; font-size: 14px; z-index: 200; display: none; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .loading { text-align: center; padding: 40px; color: var(--text2); }
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
                <h2 style="font-size:18px; color:var(--text2);">Danh sách dự án</h2>
                <button class="btn-create" onclick="showCreateModal()">+ Tạo dự án mới</button>
            </div>
            <div id="projectList"><div class="loading">Đang tải...</div></div>
        </div>

        <div id="detailPage" style="display:none;">
            <div class="detail-header">
                <h2 id="detailName"></h2>
                <div class="meta" id="detailMeta"></div>
            </div>
            <div class="detail-actions">
                <button onclick="showAddPagesModal()">+ Thêm trang</button>
                <button onclick="showRenameModal()">✏️ Đổi tên</button>
                <button onclick="showUpdateModal()">🔄 Update từ .aniga</button>
                <button onclick="resolveProject()">📤 Xuất sản phẩm</button>
                <button onclick="downloadBundle()">📥 Tải .aniga</button>
                <span style="margin-left:auto; display:flex; align-items:center; gap:8px; font-size:13px; color:var(--text2);">
                    Flux Size:
                    <select id="fluxSizeSelect" onchange="updateFluxSize(this.value)" style="background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px;font-size:13px;cursor:pointer;">
                        <option value="1024">1024</option>
                        <option value="1280" selected>1280</option>
                        <option value="1536">1536</option>
                    </select>
                </span>
            </div>
            <div class="pages-grid" id="pagesGrid"></div>
        </div>
    </div>

    <!-- Create Modal -->
    <div class="modal-overlay" id="createModal">
        <div class="modal">
            <h3>Tạo dự án mới</h3>
            <input type="text" id="createName" placeholder="Tên dự án (vd: Naruto Ch.001)">
            <input type="file" id="createFiles" multiple accept="image/*">
            <div id="createStatus" style="color:var(--text2); font-size:13px; margin-bottom:8px;"></div>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="createProject()">Tạo</button>
            </div>
        </div>
    </div>

    <!-- Update Modal -->
    <div class="modal-overlay" id="updateModal">
        <div class="modal">
            <h3>🔄 Update từ file .aniga</h3>
            <p style="color:var(--text2); margin-bottom:12px; font-size:13px;">Chọn file .aniga đã xử lý từ Colab để merge vào dự án gốc.</p>
            <input type="file" id="updateFile" accept=".aniga">
            <div id="updateStatus" style="color:var(--text2); font-size:13px;"></div>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="updateProject()">Update</button>
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
            <h3>+ Thêm trang</h3>
            <input type="file" id="addPagesFiles" multiple accept="image/*">
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Hủy</button>
                <button class="btn-primary" onclick="addPages()">Thêm</button>
            </div>
        </div>
    </div>

    <!-- Preview Modal -->
    <div class="modal-overlay" id="previewModal">
        <div class="modal" style="min-width:640px;">
            <h3 id="previewTitle">Preview</h3>
            <div class="preview-tabs" id="previewTabs"></div>
            <div class="preview-img" id="previewImg"><span class="no-data">Chọn layer để xem</span></div>
            <div class="btn-row">
                <button class="btn-cancel" onclick="closeModals()">Đóng</button>
                <button class="btn-cancel" onclick="resetCurrentPage()" style="color:var(--red);">🔄 Reset trang</button>
                <button class="btn-cancel" onclick="deleteCurrentPage()" style="color:var(--red);">🗑️ Xóa trang</button>
            </div>
        </div>
    </div>

    <!-- Toast -->
    <div class="toast" id="toast"></div>

    <script>
    let currentFile = null;
    let currentProject = null;
    let previewPageId = null;

    // ── UTILS ──
    function toast(msg, duration=3000) {
        const t = document.getElementById('toast');
        t.textContent = msg; t.style.display = 'block';
        setTimeout(() => t.style.display = 'none', duration);
    }
    function closeModals() {
        document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
    }

    // ── HOME ──
    async function loadProjects() {
        const res = await fetch('/api/projects');
        const projects = await res.json();
        const el = document.getElementById('projectList');
        if (!projects.length) { el.innerHTML = '<div class="loading">Chưa có dự án nào. Bấm "Tạo dự án mới" để bắt đầu.</div>'; return; }
        el.innerHTML = projects.map(p => p.error
            ? `<div class="project-card"><span class="name">⚠️ ${p.filename}</span><div class="meta">${p.error}</div></div>`
            : `<div class="project-card" onclick="openProject('${p.filename}')">
                <span class="name">📁 ${p.project_name}</span>
                <div class="meta">
                    <span class="id-tag">${p.project_id}</span>
                    <span>${p.page_count} trang</span>
                    <span>Clean: ${p.clean_progress}</span>
                    <span>Mask: ${p.mask_progress}</span>
                </div>
               </div>`
        ).join('');
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
        const res = await fetch(`/api/projects/${filename}`);
        currentProject = await res.json();
        renderDetail();
        document.getElementById('homePage').style.display = 'none';
        document.getElementById('detailPage').style.display = '';
        document.getElementById('backBtn').style.display = '';
    }
    function renderDetail() {
        const p = currentProject;
        document.getElementById('detailName').textContent = `📁 ${p.project_name}`;
        document.getElementById('detailMeta').innerHTML = `<span class="id-tag" style="background:var(--bg3);padding:2px 8px;border-radius:4px;font-family:monospace;">${p.project_id}</span> &nbsp; ${p.page_count} trang`;
        // Set flux_size dropdown
        const fluxSize = (p.imgcraft_config && p.imgcraft_config.flux_size) || 1280;
        const sel = document.getElementById('fluxSizeSelect');
        if (sel) sel.value = fluxSize;
        const grid = document.getElementById('pagesGrid');
        grid.innerHTML = p.pages.map(pg => {
            const dn = pg.display_name.split('_').pop();
            const cs = pg.has_clean ? 's-done' : 's-pending';
            const ms = pg.has_mask ? 's-done' : 's-pending';
            const ds = pg.has_detections ? 's-done' : 's-pending';
            const es = pg.error ? 's-error' : '';
            return `<div class="page-card ${es}" onclick="showPreview('${pg.hidden_id}','${pg.display_name}', ${pg.has_clean}, ${pg.has_mask})">
                <div class="thumb"><img src="/api/projects/${currentFile}/pages/${pg.hidden_id}/raw" loading="lazy" onerror="this.style.display='none'"></div>
                <div class="page-name">${dn}</div>
                <div class="status">
                    <span class="${cs}">C</span>
                    <span class="${ms}">M</span>
                    <span class="${ds}">D</span>
                </div>
                ${pg.error ? '<div style="font-size:10px;color:var(--red);margin-top:4px;">⚠️</div>' : ''}
            </div>`;
        }).join('');
    }

    // ── PREVIEW ──
    function showPreview(hiddenId, displayName, hasClean, hasMask) {
        previewPageId = hiddenId;
        document.getElementById('previewTitle').textContent = displayName;
        const tabs = document.getElementById('previewTabs');
        const layers = ['raw'];
        if (hasClean) layers.push('clean');
        if (hasMask) layers.push('mask');
        tabs.innerHTML = layers.map((l,i) =>
            `<button class="${i===0?'active':''}" onclick="switchPreview('${hiddenId}','${l}',this)">${l.toUpperCase()}</button>`
        ).join('');
        switchPreview(hiddenId, 'raw', tabs.firstChild);
        document.getElementById('previewModal').classList.add('active');
    }
    function switchPreview(hiddenId, layer, btn) {
        document.querySelectorAll('#previewTabs button').forEach(b => b.classList.remove('active'));
        if(btn) btn.classList.add('active');
        const imgDiv = document.getElementById('previewImg');
        imgDiv.innerHTML = `<img src="/api/projects/${currentFile}/pages/${hiddenId}/${layer}" onerror="this.outerHTML='<span class=\\'no-data\\'>Không có dữ liệu</span>'">`;
    }
    async function deleteCurrentPage() {
        if (!confirm('Xóa trang này? Không thể hoàn tác!')) return;
        await fetch(`/api/projects/${currentFile}/pages/${previewPageId}`, {method:'DELETE'});
        closeModals(); openProject(currentFile);
        toast('✅ Đã xóa trang');
    }
    async function resetCurrentPage() {
        if (!confirm('Reset trang này? Sẽ xóa clean/mask/detections.')) return;
        await fetch(`/api/projects/${currentFile}/pages/${previewPageId}/reset`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({layers:['clean','mask','detections']})});
        closeModals(); openProject(currentFile);
        toast('✅ Đã reset trang');
    }

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
        if (!name) { toast('⚠️ Nhập tên dự án!'); return; }
        if (!files.length) { toast('⚠️ Chọn ít nhất 1 ảnh!'); return; }
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
        closeModals(); openProject(currentFile);
        if (data.errors?.length) toast(`⚠️ ${data.errors.join(', ')}`, 5000);
        else toast(`✅ Đã sync ${data.synced} trang`);
    }

    // ── FLUX SIZE CONFIG ──
    async function updateFluxSize(val) {
        const res = await fetch(`/api/projects/${currentFile}/imgcraft-config`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({flux_size: parseInt(val)})
        });
        if (res.ok) toast(`✅ Flux Size → ${val}`);
        else toast('❌ Lỗi cập nhật Flux Size');
    }

    // ── RESOLVE / DOWNLOAD ──
    function resolveProject() { window.location.href = `/api/projects/${currentFile}/resolve`; }
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
