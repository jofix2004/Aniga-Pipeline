# chapter_manager.py — Shared module: Đọc/ghi file .aniga (ZIP-based project format)
# Dùng chung cho: aniga_local, aniga_imgcraft, aniga_aniga3

import os
import json
import time
import random
import string
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import io

# ============================================================
# CONSTANTS
# ============================================================
ANIGA_VERSION = "1.0"


# ============================================================
# HELPER: Tạo hidden_id
# ============================================================
def _generate_hidden_id():
    """Tạo hidden_id duy nhất: YYYYMMDD_HHMMSS_xxxx (4 ký tự random)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rand}"


def _generate_project_id():
    """Tạo project_id ngắn gọn: 8 ký tự hex."""
    return ''.join(random.choices(string.hexdigits[:16], k=8))


# ============================================================
# HELPER: Display name
# ============================================================
def get_display_name(project_name, display_order):
    """Tính display name từ project name và thứ tự."""
    return f"{project_name}_P{display_order + 1:04d}"


# ============================================================
# HELPER: Natural sort (cho tên file ảnh)
# ============================================================
def _natural_sort_key(s):
    """Natural sort key: 'page2' < 'page10'."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]





# ============================================================
# CORE: Tạo bundle mới từ danh sách ảnh
# ============================================================
def create_bundle(image_paths, chapter_name, output_path):
    """
    Tạo file .aniga mới từ danh sách ảnh raw.
    Lưu ảnh gốc nguyên vẹn, không resize.
    """
    image_paths = sorted(image_paths, key=_natural_sort_key)

    project_id = _generate_project_id()
    now = datetime.now().isoformat(timespec='seconds')

    pages = []
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for i, img_path in enumerate(image_paths):
            hidden_id = _generate_hidden_id()
            time.sleep(0.01)

            # Copy file gốc nguyên vẹn vào ZIP
            zf.write(img_path, f"pages/{hidden_id}/raw.png")

            pages.append({
                "hidden_id": hidden_id,
                "display_order": i,
                "original_filename": os.path.basename(img_path),
                "has_clean": False,
                "has_mask": False,
                "has_detections": False,
                "error": None
            })

        manifest = {
            "version": ANIGA_VERSION,
            "project_id": project_id,
            "project_name": chapter_name,
            "created_at": now,
            "updated_at": now,
            "pages": pages,
            "aniga3_config": None
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest


# ============================================================
# READ: Đọc manifest
# ============================================================
def read_manifest(bundle_path):
    """Đọc manifest.json từ file .aniga. Trả về dict."""
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        data = zf.read("manifest.json")
        return json.loads(data)


def read_manifest_from_dir(working_dir):
    """Đọc manifest.json từ thư mục làm việc (đã giải nén)."""
    manifest_path = os.path.join(working_dir, "manifest.json")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# READ: Lấy ảnh từ bundle
# ============================================================
def get_page_from_bundle(bundle_path, hidden_id, layer="raw"):
    """
    Lấy 1 ảnh từ file .aniga dưới dạng bytes.
    layer: "raw" | "clean" | "mask"
    Trả về bytes hoặc None.
    """
    filename = f"pages/{hidden_id}/{layer}.png"
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        if filename in zf.namelist():
            return zf.read(filename)
    return None


def get_page_from_dir(working_dir, hidden_id, layer="raw"):
    """Lấy đường dẫn ảnh từ thư mục làm việc. Trả về path hoặc None."""
    filepath = os.path.join(working_dir, "pages", hidden_id, f"{layer}.png")
    if os.path.exists(filepath):
        return filepath
    return None


def get_detections_from_bundle(bundle_path, hidden_id):
    """Lấy detections.json từ file .aniga. Trả về dict hoặc None."""
    filename = f"pages/{hidden_id}/detections.json"
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        if filename in zf.namelist():
            data = zf.read(filename)
            return json.loads(data)
    return None


def get_detections_from_dir(working_dir, hidden_id):
    """Lấy detections.json từ thư mục làm việc."""
    filepath = os.path.join(working_dir, "pages", hidden_id, "detections.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ============================================================
# QUERY: Tìm trang chưa xử lý
# ============================================================
def get_pending_pages(manifest, layer="clean"):
    """
    Trả về danh sách page entries chưa có layer đó, sắp xếp theo display_order.
    layer: "clean" | "mask" | "detections"
    """
    key = f"has_{layer}"
    pending = [p for p in manifest["pages"] if not p.get(key, False) and p.get("error") is None]
    return sorted(pending, key=lambda p: p["display_order"])


def get_next_pending(manifest, layer="clean"):
    """Trả về page entry tiếp theo chưa xử lý, hoặc None nếu xong hết."""
    pending = get_pending_pages(manifest, layer)
    return pending[0] if pending else None


def get_progress(manifest, layer="clean"):
    """Trả về (done, total) cho layer."""
    key = f"has_{layer}"
    total = len(manifest["pages"])
    done = sum(1 for p in manifest["pages"] if p.get(key, False))
    return done, total


# ============================================================
# WRITE: Giải nén bundle ra thư mục làm việc
# ============================================================
def extract_to_working_dir(bundle_path, working_dir):
    """
    Giải nén .aniga vào thư mục làm việc.
    Nếu thư mục đã tồn tại (resume), không giải nén lại.
    """
    manifest_path = os.path.join(working_dir, "manifest.json")

    if os.path.exists(manifest_path):
        # Đã giải nén trước đó → resume
        print(f"📂 Tìm thấy working dir cũ: {working_dir} — Resume mode")
        return read_manifest_from_dir(working_dir)

    # Giải nén mới
    os.makedirs(working_dir, exist_ok=True)
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        zf.extractall(working_dir)
    print(f"📦 Đã giải nén {bundle_path} → {working_dir}")
    return read_manifest_from_dir(working_dir)


# ============================================================
# WRITE: Lưu ảnh vào thư mục làm việc (Transactional)
# ============================================================
def save_page_to_dir(working_dir, hidden_id, layer, data):
    """
    Lưu ảnh vào thư mục làm việc theo cơ chế transactional.
    data: PIL Image hoặc bytes.
    """
    page_dir = os.path.join(working_dir, "pages", hidden_id)
    os.makedirs(page_dir, exist_ok=True)

    final_path = os.path.join(page_dir, f"{layer}.png")
    temp_path = os.path.join(page_dir, f"_temp_{layer}.png")

    if isinstance(data, bytes):
        with open(temp_path, 'wb') as f:
            f.write(data)
    else:
        # PIL Image
        data.save(temp_path, format='PNG')

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(temp_path, final_path)


def save_detections_to_dir(working_dir, hidden_id, detections_dict):
    """Lưu detections.json vào thư mục làm việc (transactional)."""
    page_dir = os.path.join(working_dir, "pages", hidden_id)
    os.makedirs(page_dir, exist_ok=True)

    final_path = os.path.join(page_dir, "detections.json")
    temp_path = os.path.join(page_dir, "_temp_detections.json")

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(detections_dict, f, indent=2, ensure_ascii=False)

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(temp_path, final_path)


# ============================================================
# WRITE: Cập nhật manifest trong thư mục làm việc
# ============================================================
def update_manifest_in_dir(working_dir, manifest):
    """Ghi manifest.json vào thư mục làm việc (transactional)."""
    final_path = os.path.join(working_dir, "manifest.json")
    temp_path = os.path.join(working_dir, "_temp_manifest.json")

    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(temp_path, final_path)


def mark_page_done(manifest, hidden_id, layer="clean", error=None):
    """
    Đánh dấu 1 trang đã xong (hoặc lỗi) trong manifest dict.
    PHẢI gọi update_manifest_in_dir() sau đó để persist.
    """
    key = f"has_{layer}"
    for page in manifest["pages"]:
        if page["hidden_id"] == hidden_id:
            if error:
                page["error"] = error
            else:
                page[key] = True
                page["error"] = None
            break
    return manifest


# ============================================================
# WRITE: Đóng gói thư mục làm việc thành .aniga
# ============================================================
def pack_to_bundle(working_dir, output_path):
    """
    Đóng gói thư mục làm việc → file .aniga (ZIP_STORED).
    Dọn dẹp file tạm (_temp_*) trước khi đóng gói.
    """
    # Dọn file tạm
    for root, dirs, files in os.walk(working_dir):
        for f in files:
            if f.startswith("_temp_"):
                os.remove(os.path.join(root, f))

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(working_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, working_dir)
                zf.write(file_path, arcname)

    print(f"📦 Đã đóng gói → {output_path}")


# ============================================================
# UPDATE: Merge dữ liệu từ bundle update vào bundle gốc
# ============================================================
def merge_bundles(original_path, update_path, delete_update=False):
    """
    Merge dữ liệu từ update bundle vào original bundle.
    Xác minh bằng project_id và hidden_id.

    Returns:
        dict — {"synced": int, "skipped": int, "errors": list[str]}
    """
    orig_manifest = read_manifest(original_path)
    update_manifest = read_manifest(update_path)

    # Xác minh project_id
    if orig_manifest["project_id"] != update_manifest["project_id"]:
        return {"synced": 0, "skipped": 0, "errors": ["❌ Sai dự án! Project ID không khớp."]}

    # Tạo lookup bằng hidden_id
    orig_pages = {p["hidden_id"]: p for p in orig_manifest["pages"]}
    update_pages = {p["hidden_id"]: p for p in update_manifest["pages"]}

    result = {"synced": 0, "skipped": 0, "errors": []}

    # Các layer cần sync
    layers = [
        ("clean.png", "has_clean"),
        ("mask.png", "has_mask"),
        ("detections.json", "has_detections")
    ]

    with zipfile.ZipFile(update_path, 'r') as zf_update:
        # Đọc original vào memory để sửa
        with zipfile.ZipFile(original_path, 'r') as zf_orig:
            orig_files = {}
            for name in zf_orig.namelist():
                orig_files[name] = zf_orig.read(name)

        synced_any = False
        for hidden_id, update_page in update_pages.items():
            if hidden_id not in orig_pages:
                result["errors"].append(f"⚠️ Page {hidden_id} không có trong gốc — bỏ qua")
                result["skipped"] += 1
                continue

            page_synced = False
            for filename, flag_key in layers:
                arcname = f"pages/{hidden_id}/{filename}"
                if arcname in zf_update.namelist():
                    if update_page.get(flag_key, False) or filename == "detections.json":
                        # Copy/overwrite
                        orig_files[arcname] = zf_update.read(arcname)
                        orig_pages[hidden_id][flag_key] = True
                        page_synced = True

            if page_synced:
                result["synced"] += 1
                synced_any = True

        if synced_any:
            # Cập nhật manifest
            orig_manifest["pages"] = list(orig_pages.values())
            orig_manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
            orig_files["manifest.json"] = json.dumps(orig_manifest, indent=2, ensure_ascii=False).encode('utf-8')

            # Ghi lại file gốc
            with zipfile.ZipFile(original_path, 'w', compression=zipfile.ZIP_STORED) as zf_out:
                for name, data in orig_files.items():
                    zf_out.writestr(name, data)

    if delete_update and os.path.exists(update_path):
        os.remove(update_path)
        print(f"🗑️ Đã xóa file update: {update_path}")

    return result


# ============================================================
# RESOLVE: Xuất sản phẩm cuối cùng
# ============================================================
def resolve_bundle(bundle_path, output_dir, include_layers=None):
    """
    Xuất .aniga → thư mục/ZIP với tên hiển thị.

    Args:
        bundle_path: đường dẫn file .aniga
        output_dir: thư mục output
        include_layers: list — ["raw", "clean", "mask", "detections"] (default: all)
    """
    if include_layers is None:
        include_layers = ["raw", "clean", "mask", "detections"]

    manifest = read_manifest(bundle_path)
    project_name = manifest["project_name"]
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        for page in sorted(manifest["pages"], key=lambda p: p["display_order"]):
            display_name = get_display_name(project_name, page["display_order"])
            hidden_id = page["hidden_id"]

            for layer in include_layers:
                if layer == "detections":
                    src = f"pages/{hidden_id}/detections.json"
                    dst = os.path.join(output_dir, f"{display_name}.json")
                else:
                    src = f"pages/{hidden_id}/{layer}.png"
                    dst = os.path.join(output_dir, f"{display_name}_{layer}.png")

                if src in zf.namelist():
                    data = zf.read(src)
                    with open(dst, 'wb') as f:
                        f.write(data)

    print(f"📤 Đã xuất {len(manifest['pages'])} trang → {output_dir}")


# ============================================================
# MANAGE: Thêm trang
# ============================================================
def add_pages_to_bundle(bundle_path, image_paths):
    """
    Thêm trang mới vào file .aniga. Lưu ảnh gốc nguyên vẹn.
    """
    manifest = read_manifest(bundle_path)
    current_max_order = max((p["display_order"] for p in manifest["pages"]), default=-1)

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            existing_files[name] = zf.read(name)

    for i, img_path in enumerate(sorted(image_paths, key=_natural_sort_key)):
        hidden_id = _generate_hidden_id()
        time.sleep(0.01)

        # Đọc bytes gốc
        with open(img_path, 'rb') as f:
            existing_files[f"pages/{hidden_id}/raw.png"] = f.read()

        page_entry = {
            "hidden_id": hidden_id,
            "display_order": current_max_order + 1 + i,
            "original_filename": os.path.basename(img_path),
            "has_clean": False,
            "has_mask": False,
            "has_detections": False,
            "error": None
        }
        manifest["pages"].append(page_entry)

    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest


# ============================================================
# MANAGE: Xóa trang
# ============================================================
def remove_page_from_bundle(bundle_path, hidden_id):
    """Xóa 1 trang khỏi file .aniga. Cập nhật display_order."""
    manifest = read_manifest(bundle_path)

    # Lọc bỏ page
    manifest["pages"] = [p for p in manifest["pages"] if p["hidden_id"] != hidden_id]

    # Renumber display_order
    for i, page in enumerate(sorted(manifest["pages"], key=lambda p: p["display_order"])):
        page["display_order"] = i

    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')

    # Đọc file, loại bỏ folder của page bị xóa
    prefix = f"pages/{hidden_id}/"
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            if not name.startswith(prefix):
                existing_files[name] = zf.read(name)

    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest


# ============================================================
# MANAGE: Đổi tên dự án
# ============================================================
def rename_project(bundle_path, new_name):
    """Đổi tên dự án trong manifest."""
    manifest = read_manifest(bundle_path)
    manifest["project_name"] = new_name
    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            existing_files[name] = zf.read(name)

    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest


# ============================================================
# MANAGE: Sắp xếp lại thứ tự trang
# ============================================================
def reorder_pages(bundle_path, hidden_id_order):
    """
    Sắp xếp lại thứ tự trang.
    hidden_id_order: list[str] — danh sách hidden_id theo thứ tự mới.
    """
    manifest = read_manifest(bundle_path)
    page_map = {p["hidden_id"]: p for p in manifest["pages"]}

    new_pages = []
    for i, hid in enumerate(hidden_id_order):
        if hid in page_map:
            page_map[hid]["display_order"] = i
            new_pages.append(page_map[hid])

    manifest["pages"] = new_pages
    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            existing_files[name] = zf.read(name)

    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest


# ============================================================
# MANAGE: Clean data (Aniga3 — xóa mask + detections)
# ============================================================
def clean_detection_data(bundle_path):
    """Xóa toàn bộ mask.png + detections.json, reset flags."""
    manifest = read_manifest(bundle_path)

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            # Bỏ qua mask + detections
            basename = os.path.basename(name)
            if basename in ("mask.png", "detections.json"):
                continue
            existing_files[name] = zf.read(name)

    for page in manifest["pages"]:
        page["has_mask"] = False
        page["has_detections"] = False
        page["error"] = None

    manifest["aniga3_config"] = None
    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest


def clean_detection_data_in_dir(working_dir):
    """Xóa toàn bộ mask.png + detections.json trong thư mục làm việc."""
    manifest = read_manifest_from_dir(working_dir)
    pages_dir = os.path.join(working_dir, "pages")

    for page in manifest["pages"]:
        page_dir = os.path.join(pages_dir, page["hidden_id"])
        for fname in ("mask.png", "detections.json"):
            fpath = os.path.join(page_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        page["has_mask"] = False
        page["has_detections"] = False
        page["error"] = None

    manifest["aniga3_config"] = None
    update_manifest_in_dir(working_dir, manifest)
    return manifest


# ============================================================
# MANAGE: Reset trang (xóa clean/mask/detections cho 1 trang)
# ============================================================
def reset_page(bundle_path, hidden_id, layers=None):
    """
    Reset 1 trang: xóa layers chỉ định, reset flags.
    layers: list — ["clean", "mask", "detections"] (default: tất cả)
    """
    if layers is None:
        layers = ["clean", "mask", "detections"]

    manifest = read_manifest(bundle_path)

    files_to_remove = set()
    for layer in layers:
        if layer == "detections":
            files_to_remove.add(f"pages/{hidden_id}/detections.json")
        else:
            files_to_remove.add(f"pages/{hidden_id}/{layer}.png")

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        existing_files = {}
        for name in zf.namelist():
            if name not in files_to_remove:
                existing_files[name] = zf.read(name)

    for page in manifest["pages"]:
        if page["hidden_id"] == hidden_id:
            for layer in layers:
                page[f"has_{layer}"] = False
            page["error"] = None
            break

    manifest["updated_at"] = datetime.now().isoformat(timespec='seconds')
    existing_files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for name, data in existing_files.items():
            zf.writestr(name, data)

    return manifest
