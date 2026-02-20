
# ImgCraft Pipeline V2 - All-In-One (PARITY VERSION)
# Contains EXACT copy of valid logic from standalone_pipeline.py
# Orchestrates Crop (CPU) -> Flux Clean (GPU) -> Stitch (CPU)

import os
import time
import socket
import atexit
import signal
import shutil
import threading
import sys
import cv2
import numpy as np
import json # <--- Added json import
import gc
import subprocess # Fix for is_pid_running
from PIL import Image, ImageDraw
import random
import torch # Explicit import for FluxProcessor type hints
# ===================================================================
# COMPONENT 1: FLUX PROCESSOR (CORE LOGIC FROM IMGCRAFT)
# ===================================================================


# ===================================================================
# COMPONENT 1: FLUX PROCESSOR (CORE LOGIC FROM IMGCRAFT)
# ===================================================================
class FluxProcessor:
    def __init__(self):
        print("⏳ [Flux] Initializing ComfyUI Core...")
        
        # Setup ComfyUI path
        self.comfyui_path = '/content/ComfyUI'
        if self.comfyui_path not in sys.path:
            sys.path.insert(0, self.comfyui_path)
            
        # Add custom_nodes to path to allow direct imports if needed
        custom_nodes_path = os.path.join(self.comfyui_path, 'custom_nodes')
        if custom_nodes_path not in sys.path:
            sys.path.append(custom_nodes_path)
            
        try:
            import gc # <--- Added gc
            import torch
            import comfy.model_management as model_management
            import folder_paths 

            # Memory Management Mode
            # HIGH_VRAM = Force Keep in VRAM (Fast but OOM risk on <24GB)
            # NORMAL_VRAM = Allow Offload (Slower but safer)
            # On L4 (24GB), Flux FP8 + T5 FP8 is tight. Let's use NORMAL or LOW based on detection.
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # AGGRESSIVE PRE-CLEANUP
            model_management.soft_empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # --- 3-TIER STRATEGY ---
            if total_vram > 30:
                print(f"🚀 [Flux] Powerhouse GPU detected ({total_vram:.1f}GB). MODE: H100/A100 (Balanced)")
                model_management.vram_state = model_management.VRAMState.HIGH_VRAM
                self.batch_size = 4
                self.tier = "H100/A100"
            elif total_vram > 20:
                print(f"⚖️ [Flux] Balanced GPU detected ({total_vram:.1f}GB). MODE: L4 (Standard)")
                model_management.vram_state = model_management.VRAMState.NORMAL_VRAM
                self.batch_size = 2
                self.tier = "L4"
            else:
                print(f"🐢 [Flux] Budget GPU detected ({total_vram:.1f}GB). MODE: T4 (Conservative)")
                model_management.vram_state = model_management.VRAMState.NORMAL_VRAM
                self.batch_size = 1
                self.tier = "T4"

            # Import Standard Nodes
            from nodes import (
                DualCLIPLoader, CLIPTextEncode, VAEEncode, VAEDecode, VAELoader,
                KSamplerAdvanced, ConditioningZeroOut, LoraLoaderModelOnly,
                UNETLoader, ImageBatch # <--- Added UNETLoader (FP8) & ImageBatch (Batching)
            )
            
            # Remove Legacy GGUF Imports logic
            # (FP8 uses standard UNETLoader)

            from comfy_extras.nodes_edit_model import ReferenceLatent
            from comfy_extras.nodes_flux import FluxGuidance
            from comfy_extras.nodes_sd3 import EmptySD3LatentImage
            
            self.nodes = {
                "clip_loader": DualCLIPLoader(),
                "unet_loader": UNETLoader(), # Standard Loader for FP8
                "vae_loader": VAELoader(),
                "vae_encode": VAEEncode(),
                "vae_decode": VAEDecode(),
                "ksampler": KSamplerAdvanced(),
                "lora_loader": LoraLoaderModelOnly(),
                "clip_encode": CLIPTextEncode(),
                "zero_out": ConditioningZeroOut(),
                "empty_latent": EmptySD3LatentImage(),
                "flux_guidance": FluxGuidance(),
                "ref_latent": ReferenceLatent(),
                "image_batch": ImageBatch() # Node for Batching
            }
            
            # Load Models
            print("⏳ [Flux] Loading Models into VRAM (FP8 Mode)...")
            # 1. Load CLIP
            self.clip = self.nodes["clip_loader"].load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
            gc.collect()
            torch.cuda.empty_cache()
            
            # 2. Load VAE
            self.vae = self.nodes["vae_loader"].load_vae("ae.sft")[0]
            gc.collect()
            torch.cuda.empty_cache()
            
            # 3. Load FP8 UNET
            # File must be in models/unet/
            model_name = "flux1-kontext-dev-fp8-e4m3fn.safetensors"
            print(f"   Loading UNET: {model_name}")
            try:
                # UNETLoader.load_unet(unet_name, weight_dtype)
                # Force FP8 to save VRAM on L4
                print(f"   ► Loading UNET with explicit FP8 Dtype...")
                model = self.nodes["unet_loader"].load_unet(model_name, "fp8_e4m3fn")[0]
                gc.collect() # <--- Important Cleanup
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ Error loading UNET {model_name}: {e}")
                print("⚠️ Falling back to safeguard or check path.")
                raise e

            # 4. Load LoRAs
            model = self.nodes["lora_loader"].load_lora_model_only(model, "flux_1_turbo_alpha.safetensors", 1.0)[0]
            model = self.nodes["lora_loader"].load_lora_model_only(model, "AniGaKontext2080v3_Full_000000650.safetensors", 1.0)[0]
            self.model = model
            
            # FINAL CLEANUP
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ [Flux] Models Loaded & Ready.")
            self.ready = True
            
        except ImportError as e:
            print(f"⚠️ [Flux] Import Error: {e}. GPU Cleaner will NOT work.")
            self.ready = False
        except Exception as e:
            print(f"❌ [Flux] Initialization Error: {e}")
            self.ready = False

    def _pil_to_tensor(self, pil_img):
        img = np.array(pil_img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img

    def _tensor_to_pil(self, tensor):
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

    def process(self, input_images: list, prompt: str = "clean manga, remove text, remove sfx, AniGa"):
        """
        Process images (Single or Batch).
        input_images: List[PIL.Image] or Single PIL.Image
        """
        if not self.ready:
            print("⚠️ [Flux] Not Ready. Skipping.")
            return input_images
        
        # 1. Normalize Input to List
        if not isinstance(input_images, list):
            input_images = [input_images]
            
        import torch
        import comfy.model_management as model_management
        
        # --- USE TIER STRATEGY ---
        MAX_BATCH_SIZE = getattr(self, 'batch_size', 1) 
        print(f"\n🚀 [Flux] Processing {len(input_images)} images (Tier: {getattr(self, 'tier', 'Unknown')} | Batch: {MAX_BATCH_SIZE})...")

        processed_results = []
        total_images = len(input_images)

        with torch.inference_mode():
            try:
                # Pre-encode Prompt (Once for all batches if same prompt)
                prompt_encode = self.nodes["clip_encode"].encode(self.clip, prompt)[0]
                negative = self.nodes["zero_out"].zero_out(prompt_encode)[0]
                
                # Process in Chunks (Safeguard Logic + OOM Fallback)
                processed_count = 0
                
                # Dynamic Setup
                current_chunk_index = 0
                while current_chunk_index < total_images:
                    # Try current batch size
                    end_idx = min(current_chunk_index + MAX_BATCH_SIZE, total_images)
                    chunk = input_images[current_chunk_index : end_idx]
                    current_batch_len = len(chunk)
                    
                    try:
                        print(f"   ► Encoding Batch Chunk (Size: {current_batch_len}) [Idx: {current_chunk_index}-{end_idx}]...")
                        
                        # 1. Prepare Batch Image Tensor
                        batch_tensor = self._pil_to_tensor(chunk[0])
                        if current_batch_len > 1:
                            for idx in range(1, current_batch_len):
                                next_tensor = self._pil_to_tensor(chunk[idx])
                                batch_tensor = self.nodes["image_batch"].batch(batch_tensor, next_tensor)[0]
                        
                        # 2. VAE Encode
                        latent = self.nodes["vae_encode"].encode(self.vae, batch_tensor)[0]
                        
                        # 3. Conditioning
                        conditioning = self.nodes["ref_latent"].append(prompt_encode, latent)[0]
                        positive = self.nodes["flux_guidance"].append(conditioning, 3.5)[0]

                        # 4. Empty Latent
                        w, h = chunk[0].size 
                        output_latent = self.nodes["empty_latent"].generate(w, h, current_batch_len)[0]
                        
                        seed = random.randint(0, 2**32 - 1)
                        print(f"   ► Sampling (Seed: {seed})...")
                        
                        # 5. Sample
                        image_out_latent = self.nodes["ksampler"].sample(
                            model=self.model, add_noise="enable", noise_seed=seed, steps=8, cfg=1.0,
                            sampler_name="dpmpp_2m", scheduler="beta", positive=positive, negative=negative,
                            latent_image=output_latent, start_at_step=0, end_at_step=1000, return_with_leftover_noise="disable"
                        )[0]
                        
                        print("   ► Decoding Batch...")
                        decoded_tensor = self.nodes["vae_decode"].decode(self.vae, image_out_latent)[0]
                        
                        # 6. Extract
                        for b in range(current_batch_len):
                            single_tensor = decoded_tensor[b] 
                            img_pil = Image.fromarray((single_tensor.cpu().numpy() * 255).astype(np.uint8))
                            processed_results.append(img_pil)
                            
                        # Success -> Move to next chunk
                        current_chunk_index += MAX_BATCH_SIZE
                        processed_count += current_batch_len
                        
                    except Exception as e:
                        if "CUDA out of memory" in str(e):
                            print(f"❌ OOM Detected with Batch {MAX_BATCH_SIZE}. Clearing Cache & Retrying with Batch 1...")
                            
                            # Aggressive Cleanup
                            model_management.soft_empty_cache()
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            if MAX_BATCH_SIZE > 1:
                                MAX_BATCH_SIZE = 1 
                                continue # Retry chunk with size 1
                            else:
                                # Even Batch 1 failed?
                                print("❌ Critical OOM even at Batch 1. Attempting final recovery...")
                                model_management.unload_all_models()
                                model_management.soft_empty_cache()
                                gc.collect()
                                torch.cuda.empty_cache()
                                raise e # Give up for this tile if it still fails
                        else:
                            torch.cuda.empty_cache()
                            raise e

                print(f"✅ [Flux] Done. Processed {len(processed_results)}/{total_images} images.")
                return processed_results
            
            except Exception as e:
                print(f"❌ Render Error: {e}")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                model_management.soft_empty_cache()
                raise

# ===================================================================
# STAGE WORKERS
# ===================================================================


print("\n[INFO] LOADED PIPELINE V2 - MY VERSION 2026-02-09\n")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# V3 STRUCTURE PATHS
INPUT_DIR = os.path.join(BASE_DIR, "Input")
PROJECTS_DIR = os.path.join(BASE_DIR, "Projects")
SYS_Q_DIR = os.path.join(BASE_DIR, "System_Queues")

# Hidden Queue Paths for Job Tickets
Q_CROP_JOBS = os.path.join(SYS_Q_DIR, "Q1_Crop_Jobs") # (Optional, mostly for tracking)
Q_GPU_JOBS = os.path.join(SYS_Q_DIR, "Q2_GPU_Jobs")
Q_STITCH_JOBS = os.path.join(SYS_Q_DIR, "Q3_Stitch_Jobs")

ERROR_DIR = os.path.join(BASE_DIR, "Errors")

def ensure_dirs():
    for d in [INPUT_DIR, PROJECTS_DIR, SYS_Q_DIR, Q_CROP_JOBS, Q_GPU_JOBS, Q_STITCH_JOBS, ERROR_DIR]:
        os.makedirs(d, exist_ok=True)

def ensure_dir_path(path):
    if not os.path.exists(path): os.makedirs(path)

# --- CRITICAL UTILS (V7: ROBUSTNESS) ---

# 1. Global Start Lock (File + PID Check)
# Replaced Socket Lock due to Windows Subprocess issues.
# Logic: Check if Lock File exists -> Check if PID in Lock is alive -> If yes, Exit. If no, Takeover.

LOCK_FILE = os.path.join(BASE_DIR, ".pipeline.lock")

def is_pid_running(pid):
    """Check if PID is running using Windows tasklist"""
    try:
        # Filter by PID and exact image name for safety
        cmd = f'tasklist /FI "PID eq {pid}" /NH'
        output = subprocess.check_output(cmd, shell=True).decode()
        # If running, output contains the PID. If not, output contains "No tasks"
        if str(pid) in output: return True
    except: pass
    return False

def acquire_instance_lock():
    print("[DEBUG] Checking Instance Lock...")
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    old_pid = int(content)
                    if is_pid_running(old_pid):
                        print(f"[ALERT] [System] Pipeline already running (PID {old_pid}). Exiting.")
                        sys.exit(1)
                    else:
                        print(f"[WARN] [System] Stale Lock found (PID {old_pid} dead). Taking over...")
        except Exception:
            pass # File corrupt, take over
            
    # Write current PID
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # Register cleanup
    def release_lock():
        try:
            if os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, 'r') as f:
                    if int(f.read().strip()) == os.getpid():
                        os.remove(LOCK_FILE)
        except: pass
        
    atexit.register(release_lock)
    print(f"[LOCK] [System] Lock Acquired (PID {os.getpid()})")

# 2. Wait for File Sync (Drive Lag Fix)
def wait_for_file_ready(file_path, timeout=30, min_size=1000):
    """
    Waits for a file to be fully written/uploaded.
    timeout: max seconds to wait in one go.
    min_size: ignore files smaller than this (default 1KB).
    """
    fname = os.path.basename(file_path)
    if fname.startswith("._") or fname.startswith("."):
        return False # Ignore hidden/temp files
        
    start = time.time()
    last_size = -1
    stable_count = 0
    
    while time.time() - start < timeout:
        if not os.path.exists(file_path):
            time.sleep(1)
            continue
            
        try:
            current_size = os.path.getsize(file_path)
        except: 
            time.sleep(0.5)
            continue
        
        # Check minimum size (Phase 2: Input Validation)
        if current_size < min_size:
            time.sleep(1)
            continue
            
        if current_size == last_size and current_size > 0:
            stable_count += 1
            # Require 2 consecutive matches (approx 1s stability)
            if stable_count >= 2: 
                return True
        else:
            stable_count = 0
            
        last_size = current_size
        time.sleep(0.5)
        
    return False


# 3. Input Validation (Use Pillow for auto-format detection) [Phase 2]
SUPPORTED_FORMATS = {'PNG', 'JPEG', 'JPG', 'WEBP', 'BMP', 'GIF', 'TIFF', 'ICO'}

def validate_image_file(path):
    """Validate image using Pillow - supports PNG, JPG, WebP, BMP, GIF, TIFF, etc."""
    try:
        if not os.path.exists(path): return False
        if os.path.getsize(path) < 100: return False  # Too small
        
        # Use Pillow to detect and verify image
        try:
            with Image.open(path) as img:
                img.verify()  # Check integrity
            
            # Re-open to get format (verify() invalidates the image object)
            with Image.open(path) as img:
                fmt = img.format
                if fmt and fmt.upper() in SUPPORTED_FORMATS:
                    return True
                else:
                    print(f"⚠️ [Sanitizer] Unsupported format '{fmt}': {os.path.basename(path)}")
                    return False
                    
        except Exception as e:
            print(f"⚠️ [Sanitizer] Cannot open image: {os.path.basename(path)} ({e})")
            return False
            
    except Exception as e:
        print(f"⚠️ [Sanitizer] Read Error {path}: {e}")
        return False

# 4. Resource Guard [Phase 2]
def check_system_health():
    try:
        # Disk Space Check (Stop if < 1GB)
        total, used, free = shutil.disk_usage(BASE_DIR)
        if free < 1 * 1024**3: # 1GB
            print("🚨 [System] DISK FULL (<1GB Free)! Pausing Pipeline...")
            return False
    except: pass
    return True

# 5. Filename Sanitization [Phase 3]
import re
def sanitize_filename(name):
    """Replace invalid chars with underscore"""
    return re.sub(r'[<>:"/\\|?*#&%\s]', '_', name)

# 6. Minimum Image Size [Phase 3]
MIN_IMAGE_SIZE = 512 # pixels

# 7. Safe Metadata Read [Phase 3]
def read_meta_safe(proj_dir):
    """Read metadata.json safely, return empty dict on error"""
    meta_path = os.path.join(proj_dir, "meta.json")
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return {}

# 3. Thread-Safe Atomic Metadata
META_LOCKS = {}
META_LOCK_GLOBAL = threading.Lock()

def get_meta_lock(proj_dir):
    with META_LOCK_GLOBAL:
        if proj_dir not in META_LOCKS:
            META_LOCKS[proj_dir] = threading.Lock()
        return META_LOCKS[proj_dir]

def update_meta_safe(proj_dir, key, value):
    lock = get_meta_lock(proj_dir)
    with lock:
        meta_path = os.path.join(proj_dir, "meta.json")
        backup_path = meta_path + ".bak"
        tmp_path = meta_path + ".tmp"
        
        data = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                try: shutil.copy2(meta_path, backup_path)
                except: pass
            except:
                if os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, meta_path)
                        with open(meta_path, 'r') as f: data = json.load(f)
                    except: pass
        
        data[key] = value
        data["last_updated"] = time.time()
        
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, meta_path)
        except Exception as e:
            print(f"⚠️ Meta Write Fail: {e}")

# Override old update_meta to use safe version
def update_meta(proj_dir, key, value):
    update_meta_safe(proj_dir, key, value)

# [Phase 4] Robust Job Ticket Creation #11
def create_job_ticket(queue_dir, job_name, max_retries=3):
    """Create job ticket with retry logic"""
    job_path = os.path.join(queue_dir, job_name)
    for attempt in range(max_retries):
        try:
            # Ensure queue dir exists
            if not os.path.exists(queue_dir):
                os.makedirs(queue_dir, exist_ok=True)
            # Create empty job file
            with open(job_path, 'w') as f:
                f.write("{}")
            return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print(f"⚠️ [Phase 4] Failed to create job ticket after {max_retries} attempts: {job_name}")
                return False
    return False

# [Phase 4] Safe Image Write with Retry #36
def safe_write_image(img_path, img_data, max_retries=3):
    """Write image with atomic save and retry on PermissionError"""
    # Use .png.tmp extension so CV2 recognizes format
    base, ext = os.path.splitext(img_path)
    tmp_path = base + ".tmp" + ext  # e.g., image.tmp.png
    for attempt in range(max_retries):
        try:
            if isinstance(img_data, Image.Image):
                img_data.save(tmp_path, format="PNG", quality=95)
            else:  # numpy array (cv2)
                cv2.imwrite(tmp_path, img_data)
            os.replace(tmp_path, img_path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"⚠️ [Phase 4] File locked, retry {attempt+1}/{max_retries}: {os.path.basename(img_path)}")
                time.sleep(1)
            else:
                print(f"❌ [Phase 4] Cannot write (file locked): {os.path.basename(img_path)}")
                return False
        except Exception as e:
            print(f"❌ [Phase 4] Write error: {e}")
            return False
    return False

# [Phase 4] Shorten Path for Windows #26
def shorten_path(base_name, max_len=50):
    """Shorten project name if too long for Windows path limit"""
    if len(base_name) > max_len:
        # Keep first 20 + last 20 + hash
        import hashlib
        hash_suffix = hashlib.md5(base_name.encode()).hexdigest()[:8]
        return f"{base_name[:20]}_{hash_suffix}_{base_name[-20:]}"
    return base_name

# ===================================================================
# CORE FUNCTIONS FROM STANDALONE_PIPELINE.PY
# ===================================================================

def center_image(img_pil, canvas_size=2048, margin=20):
    """Resize and center image on white canvas"""
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
        
    return canvas

def crop_4_tiles(img_pil, output_base, size=1024):
    """Crop 4 predefined tiles (No Preview)"""
    coords = [
        (250, 20),         # Tile 0: Top-Left
        (774, 20),        # Tile 1: Top-Right
        (250, 1004),      # Tile 2: Bottom-Left
        (774, 1004)     # Tile 3: Bottom-Right
    ]
    
    tile_paths = []
    base_name, ext = os.path.splitext(output_base)
    # base_name examples: ".../Projects/Ani-001/raw_tiles/Ani-001"
    
    for i, (x, y) in enumerate(coords):
        box = (x, y, x + size, y + size)
        tile = img_pil.crop(box)
        
        # Save tile with explicit system name: Ani-001_tile_0.png
        # output_base is passed as joined path, so splitext works on the full path
        tile_filename = f"{os.path.basename(base_name)}_tile_{i}{ext}" 
        tile_dir = os.path.dirname(base_name)
        tile_path = os.path.join(tile_dir, tile_filename)
        
        # Atomic Write
        tmp_path = tile_path + ".tmp"
        tile.save(tmp_path, format="PNG", quality=95)
        os.replace(tmp_path, tile_path)
        
        tile_paths.append(tile_path)
        
    return tile_paths, None

# ... (Helper functions restored) ...

def preprocess_grayscale_invert(img):
    """Convert to grayscale and invert colors"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.bitwise_not(gray)

def preprocess_ink_mask(img):
    """Create ink mask (binary)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def prepare_master_data(master_img):
    """Pre-calculate master image features"""
    gray_inv = preprocess_grayscale_invert(master_img)
    detector = cv2.AKAZE_create()
    kpts, descs = detector.detectAndCompute(gray_inv, None)
    ink_mask = preprocess_ink_mask(master_img)
    return {
        "gray_inv": gray_inv,
        "kpts": kpts,
        "descs": descs,
        "ink_mask": ink_mask
    }

def find_homography(gray_img, master_data):
    """Find Homography matrix using AKAZE matching"""
    detector = cv2.AKAZE_create()
    kpts1, descs1 = detector.detectAndCompute(gray_img, None)
    
    kpts2 = master_data["kpts"]
    descs2 = master_data["descs"]
    
    if descs1 is None or descs2 is None or len(descs1) < 5 or len(descs2) < 5:
        return None, 0
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descs1, descs2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) < 4:
        return None, len(good)
    
    src = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H, len(good)

def calculate_score(master_img, tile_img, tile_mask=None, master_ink_mask=None):
    """Calculate overlap score and ink score"""
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    tile_gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(master_gray, tile_gray)
    
    if tile_mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=tile_mask)
        total_pixels = cv2.countNonZero(tile_mask)
    else:
        total_pixels = diff.size
    
    _, diff_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    
    if tile_mask is not None:
        diff_mask = cv2.bitwise_and(diff_mask, tile_mask)
    
    if total_pixels == 0:
        total_pixels = 1
    
    diff_pixels = cv2.countNonZero(diff_mask)
    overlap_score = ((total_pixels - diff_pixels) / total_pixels) * 100
    
    if master_ink_mask is not None:
        ink_mask = master_ink_mask
    else:
        ink_mask = preprocess_ink_mask(master_img)
    
    if tile_mask is not None:
        ink_mask = cv2.bitwise_and(ink_mask, tile_mask)
    
    ink_pixels = cv2.countNonZero(ink_mask)
    
    if ink_pixels > 0:
        diff_on_ink = cv2.bitwise_and(diff_mask, ink_mask)
        ink_diff_pixels = cv2.countNonZero(diff_on_ink)
        ink_score = 100.0 - (ink_diff_pixels / ink_pixels * 100.0)
        ink_score = max(0, ink_score)
    else:
        ink_score = 0.0
    
    return overlap_score, ink_score, diff_mask

def compute_idw_cpu(src_img, master_img, master_data, src_mask=None):
    """Inverse Distance Weighting Warp (CPU)"""
    h, w = master_img.shape[:2]
    rows = 8
    cols = 8
    step_y = h // rows
    step_x = w // cols
    
    src_gray_inv = preprocess_grayscale_invert(src_img)
    
    detector = cv2.AKAZE_create()
    kpts1, descs1 = detector.detectAndCompute(src_gray_inv, None)
    
    kpts2 = master_data["kpts"]
    descs2 = master_data["descs"]
    
    if descs1 is None or descs2 is None or len(descs1) < 10 or len(descs2) < 10:
        return src_img, src_mask, {"matches": 0, "anchors": 0}, None, None
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descs1, descs2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    def filter_by_grid(matches_in, offset):
        off_x = int(step_x * offset[0])
        off_y = int(step_y * offset[1])
        grid_cells = {}
        for m in matches_in:
            pt = kpts2[m.trainIdx].pt
            c = int((pt[0] - off_x) / step_x)
            r = int((pt[1] - off_y) / step_y)
            if (r, c) not in grid_cells:
                grid_cells[(r, c)] = []
            grid_cells[(r, c)].append(m)
        src_pts = []
        dst_pts = []
        for key in grid_cells:
            ms = grid_cells[key]
            if len(ms) < 4: continue
            pts_s = np.float32([kpts1[m.queryIdx].pt for m in ms]).reshape(-1, 2)
            pts_d = np.float32([kpts2[m.trainIdx].pt for m in ms]).reshape(-1, 2)
            H_local, mask_local = cv2.findHomography(pts_s, pts_d, cv2.RANSAC, 5.0)
            if H_local is not None:
                for i, valid in enumerate(mask_local.ravel()):
                    if valid:
                        src_pts.append(pts_s[i])
                        dst_pts.append(pts_d[i])
        return src_pts, dst_pts
    
    src_A, dst_A = filter_by_grid(good, (0.0, 0.0))
    src_B, dst_B = filter_by_grid(good, (0.5, 0.5))
    src_all = src_A + src_B
    dst_all = dst_A + dst_B
    
    if len(src_all) < 10:
        return src_img, src_mask, {"matches": len(good), "anchors": len(src_all)}, None, None
    
    src_all, dst_all = np.array(src_all), np.array(dst_all)
    displacement = src_all - dst_all
    
    step_avg = (step_x + step_y) / 2.0
    threshold = 0.5 * step_avg
    norms = np.linalg.norm(displacement, axis=1)
    outliers = np.where(norms > threshold)[0]
    
    if len(outliers) > 0:
        src_all = np.delete(src_all, outliers, axis=0)
        dst_all = np.delete(dst_all, outliers, axis=0)
        displacement = np.delete(displacement, outliers, axis=0)
    
    grid_size = 32
    y_coords = np.linspace(0, h, grid_size)
    x_coords = np.linspace(0, w, grid_size)
    gv_x, gv_y = np.meshgrid(x_coords, y_coords)
    grid_pts = np.column_stack((gv_x.flatten(), gv_y.flatten()))
    
    dx = np.zeros(grid_pts.shape[0], dtype=np.float32)
    dy = np.zeros(grid_pts.shape[0], dtype=np.float32)
    epsilon = 1e-6
    batch_size = 512
    
    for i in range(0, grid_pts.shape[0], batch_size):
        end = min(i + batch_size, grid_pts.shape[0])
        batch_grid = grid_pts[i:end]
        dist_sq = np.sum((batch_grid[:, np.newaxis, :] - dst_all[np.newaxis, :, :]) ** 2, axis=2)
        weights = 1.0 / (dist_sq + epsilon)
        sum_weights = np.sum(weights, axis=1)
        dx[i:end] = np.sum(weights * displacement[:, 0], axis=1) / (sum_weights + epsilon)
        dy[i:end] = np.sum(weights * displacement[:, 1], axis=1) / (sum_weights + epsilon)
        
    map_x = dx.reshape(grid_size, grid_size).astype(np.float32)
    map_y = dy.reshape(grid_size, grid_size).astype(np.float32)
    full_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_CUBIC)
    full_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    np.clip(full_x, -100, 100, out=full_x)
    np.clip(full_y, -100, 100, out=full_y)
    
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x_final = grid_x + full_x
    map_y_final = grid_y + full_y
    
    warped_img = cv2.remap(src_img, map_x_final, map_y_final, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    valid_mask = src_mask if src_mask is not None else np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.remap(valid_mask, map_x_final, map_y_final, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    stats = {"matches": len(good), "anchors": len(src_all), "src_pts": src_all, "dst_pts": dst_all}
    return warped_img, warped_mask, stats, map_x, map_y

def process_tile_core(tile_img, master_img, master_data, refine=True):
    h_master, w_master = master_img.shape[:2]
    gray_tile = preprocess_grayscale_invert(tile_img)
    H, num_matches = find_homography(gray_tile, master_data)
    
    result = {
        "success": False, "warped_img": None, "warped_mask": None, 
        "metrics": {"num_matches": num_matches}
    }
    
    if H is None: return result
    
    if H is not None:
        angle_rad = np.arctan2(H[1, 0], H[0, 0])
        angle_deg = np.degrees(angle_rad)
        result["metrics"]["angle_deg"] = angle_deg
    else:
        result["metrics"]["angle_deg"] = 0.0

    warped_coarse = cv2.warpPerspective(tile_img, H, (w_master, h_master))
    h_tile, w_tile = tile_img.shape[:2]
    tile_mask = np.ones((h_tile, w_tile), dtype=np.uint8) * 255
    mask_coarse = cv2.warpPerspective(tile_mask, H, (w_master, h_master), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    score_coarse, ink_score_coarse, _ = calculate_score(master_img, warped_coarse, mask_coarse, master_data["ink_mask"])
    
    result["warped_img"] = warped_coarse
    result["warped_mask"] = mask_coarse
    result["metrics"]["final_score"] = score_coarse
    
    if not refine:
        result["success"] = True
        return result
        
    print(f"     [IDW] Refining alignment...")
    t_idw_start = time.perf_counter()
    warped_fine, mask_fine, stats, _, _ = compute_idw_cpu(warped_coarse, master_img, master_data, mask_coarse)
    t_idw_end = time.perf_counter()
    score_fine, ink_score_fine, _ = calculate_score(master_img, warped_fine, mask_fine, master_data["ink_mask"])
    
    print(f"     [IDW] Anchors: {stats.get('anchors',0)} | Matches: {stats.get('matches',0)} | Time: {t_idw_end - t_idw_start:.2f}s")
    
    if score_fine >= score_coarse:
        print(f"     [IDW] Improved Score: {score_coarse:.2f} -> {score_fine:.2f}")
        result["warped_img"] = warped_fine
        result["warped_mask"] = mask_fine
        result["metrics"]["final_score"] = score_fine
    else:
        print(f"     [IDW] No Improvement (Kept Coarse)")
        
    result["success"] = True
    return result

# --- SMART STITCH HELPERS ---

def create_directional_mask(full_h, full_w, tile_idx, content_mask, feat_x=50, feat_y=20):
    # tile_idx: 0=TL, 1=TR, 2=BL, 3=BR
    # Mask starts opaque (1.0)
    mask = np.ones((full_h, full_w), dtype=np.float32)
    
    # Find Bounding Box of Content
    if content_mask is None: return mask
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask
        
    # Get overall bounding box
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    # Tile 0 (TL): Feather Right & Bottom
    if tile_idx == 0:
        start_x = x + w
        for i in range(feat_x):
            curr_x = start_x - 1 - i
            if curr_x < 0: break
            mask[:, curr_x] *= (i / feat_x)
        start_y = y + h
        for i in range(feat_y):
            curr_y = start_y - 1 - i
            if curr_y < 0: break
            mask[curr_y, :] *= (i / feat_y)

    # Tile 1 (TR): Feather Left & Bottom
    elif tile_idx == 1:
        start_x = x
        for i in range(feat_x):
            curr_x = start_x + i
            if curr_x >= full_w: break
            mask[:, curr_x] *= (i / feat_x)
        start_y = y + h
        for i in range(feat_y):
            curr_y = start_y - 1 - i
            if curr_y < 0: break
            mask[curr_y, :] *= (i / feat_y)

    # Tile 2 (BL): Feather Right & Top
    elif tile_idx == 2:
        start_x = x + w
        for i in range(feat_x):
            curr_x = start_x - 1 - i
            if curr_x < 0: break
            mask[:, curr_x] *= (i / feat_x)
        start_y = y
        for i in range(feat_y):
            curr_y = start_y + i
            if curr_y >= full_h: break
            mask[curr_y, :] *= (i / feat_y)

    # Tile 3 (BR): Feather Left & Top
    elif tile_idx == 3:
        start_x = x
        for i in range(feat_x):
            curr_x = start_x + i
            if curr_x >= full_w: break
            mask[:, curr_x] *= (i / feat_x)
        start_y = y
        for i in range(feat_y):
            curr_y = start_y + i
            if curr_y >= full_h: break
            mask[curr_y, :] *= (i / feat_y)
            
    return mask

def match_histogram_l_channel(source_bgr, target_bgr, tile_geo_mask, master_ink_mask):
    """ Matches L-channel Histogram on Background Only """
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)
    l_src, a_src, b_src = cv2.split(src_lab)
    l_tgt, _, _ = cv2.split(tgt_lab)
    
    # 1. Geometry Mask
    if tile_geo_mask is None:
        _, geo_mask = cv2.threshold(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    else:
        _, geo_mask = cv2.threshold(tile_geo_mask, 10, 255, cv2.THRESH_BINARY)
        
    # 2. Ink Mask (Exclusion)
    if master_ink_mask is None:
        ink_mask = np.zeros_like(geo_mask)
    else:
        _, ink_mask = cv2.threshold(master_ink_mask, 10, 255, cv2.THRESH_BINARY)
        
    # 3. Background Reference Mask
    bg_mask = cv2.bitwise_and(geo_mask, cv2.bitwise_not(ink_mask))
    
    if cv2.countNonZero(bg_mask) < 100: bg_mask = geo_mask
        
    valid_src = l_src[bg_mask > 0]
    valid_tgt = l_tgt[bg_mask > 0]
    
    if len(valid_src) == 0 or len(valid_tgt) == 0: return source_bgr

    hist_src, _ = np.histogram(valid_src.flatten(), 256, [0,256])
    hist_tgt, _ = np.histogram(valid_tgt.flatten(), 256, [0,256])
    
    cdf_src = hist_src.cumsum().astype(np.float32)
    cdf_tgt = hist_tgt.cumsum().astype(np.float32)
    
    if cdf_src[-1] > 0: cdf_src /= cdf_src[-1]
    if cdf_tgt[-1] > 0: cdf_tgt /= cdf_tgt[-1]
    
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(cdf_tgt - cdf_src[i])
        lut[i] = np.argmin(diff)
        
    l_new = cv2.LUT(l_src, lut)
    merged_lab = cv2.merge([l_new, a_src, b_src])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def stack_images_directional(master_cv, tiles_data):
    # Alpha Compositing (Painter's Algorithm)
    h_canvas, w_canvas = master_cv.shape[:2]
    canvas = cv2.cvtColor(master_cv, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    for item in tiles_data:
        t_img = item["warped_img"]
        t_idx = item.get("tile_index", -1)
        if t_idx == -1: continue # Skip unknown tiles
        
        geo_mask_u8 = item["warped_mask"]
        _, geo_bin = cv2.threshold(geo_mask_u8, 10, 1.0, cv2.THRESH_BINARY)
        geo_mask = geo_bin.astype(np.float32)
        
        dir_mask = create_directional_mask(h_canvas, w_canvas, t_idx, geo_mask_u8, feat_x=50, feat_y=20)
        
        alpha = geo_mask * dir_mask
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        
        t_img_f = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        canvas = canvas * (1.0 - alpha_3ch) + t_img_f * alpha_3ch
        
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

def stack_images(master_img, tile_results):
    # Legacy wrapper for back-compat if needed, but we should use directional
    # If tile_index is missing, this might fail.
    # But for now, let's keep it acting as a proxy or just leave the old one?
    # No, we replace it with the NEW one if tiles have index.
    return stack_images_directional(master_img, tile_results)
    
    for res in tile_results:
        if not res.get("success") or res.get("warped_img") is None: continue
        img_f = cv2.cvtColor(res["warped_img"], cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = res["warped_mask"] if res["warped_mask"] is not None else np.ones((h, w), dtype=np.uint8) * 255
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        acc_color += img_f * dist[:, :, np.newaxis]
        acc_weight += dist
        
    blended = np.zeros_like(master_rgb)
    mask_exists = acc_weight > 0
    blended[mask_exists] = acc_color[mask_exists] / acc_weight[mask_exists][:, np.newaxis]
    blended[~mask_exists] = master_rgb[~mask_exists]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

# ===================================================================
# STAGE WORKERS (V3 ARCHITECTURE)
# ===================================================================

# ... (Previous Helper Functions) ...

# ===================================================================
# STAGE WORKERS (V3 ARCHITECTURE)
# ===================================================================

import json
import time

def update_meta(project_dir, key, value):
    """Update metadata json for time tracking"""
    meta_path = os.path.join(project_dir, "meta.json")
    data = {}
    
    # Read existing
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
        except:
            pass
            
    # Update value
    data[key] = value
    data["last_updated"] = time.time()
    
    # Write back (Atomic)
    tmp_meta = meta_path + ".tmp"
    try:
        with open(tmp_meta, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_meta, meta_path)
    except:
        pass

def create_job_ticket(queue_dir, job_name):
    """Create a 0KB job ticket file"""
    path = os.path.join(queue_dir, job_name)
    with open(path, 'w') as f:
        pass # Empty file

# Sync Event
STARTUP_EVENT = threading.Event()
SHUTDOWN_FLAG = threading.Event()

def signal_handler(signum, frame):
    print("\n🛑 [System] Shutdown Signal Received. Stopping Workers safely...")
    SHUTDOWN_FLAG.set()
    STARTUP_EVENT.set() # Unblock waiters so they can exit

# Register Signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def worker_crop():
    print("👀 [Stage 1] Waiting for System Init...")
    STARTUP_EVENT.wait()
    if SHUTDOWN_FLAG.is_set(): return
    
    print("👀 [Stage 1] Crop Monitor Started (Scanning Input/).")
    while not SHUTDOWN_FLAG.is_set():
        try:
            ensure_dirs()
            files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                time.sleep(1)
                continue
            
            for f in files:
                src = os.path.join(INPUT_DIR, f)
                
                # [V7] Wait for Upload Finish
                if not wait_for_file_ready(src):
                    print(f"⚠️ [Stage 1] Skipping {f} (File not ready/uploading...)")
                    continue

                print(f"🚀 [Stage 1] Processing NEW Project: {f}")
                
                try:
                    # 1. Setup Project Structure (Unique ID)
                    # [Phase 3] Filename Sanitization #4
                    base_name = sanitize_filename(os.path.splitext(f)[0])
                    # [Phase 4] Shorten Long Paths #26
                    base_name = shorten_path(base_name)
                    # Format: TIMESTAMP_BASENAME to avoid conflicts
                    timestamp_id = time.strftime("%Y%m%d_%H%M%S")
                    unique_id = f"{timestamp_id}_{base_name}"
                    
                    proj_dir = os.path.join(PROJECTS_DIR, unique_id)
                    raw_dir = os.path.join(proj_dir, "raw_tiles")
                    clean_dir = os.path.join(proj_dir, "clean_tiles")
                    
                    ensure_dir_path(proj_dir)
                    ensure_dir_path(raw_dir)
                    ensure_dir_path(clean_dir)
                    
                    # [TIME TRACKING] Created At
                    update_meta(proj_dir, "created_at", time.time())
                    update_meta(proj_dir, "original_name", f)
                    update_meta(proj_dir, "status", "cropping")
                    
                    # 2. Consume Input (Move to Project Folder)
                    dest_src = os.path.join(proj_dir, f)
                    # [Phase 3] Defensive #9: Try-Catch FileNotFound
                    # [Phase 4] Defensive #10: Try-Catch PermissionError
                    try:
                        shutil.move(src, dest_src)
                    except FileNotFoundError:
                        print(f"⚠️ [Stage 1] File deleted before processing: {f}")
                        continue
                    except PermissionError:
                        print(f"⚠️ [Stage 1] File locked by another app: {f}. Skipping...")
                        continue
                    
                    # [Phase 2] Input Sanitization #1, #2
                    if not validate_image_file(dest_src):
                        print(f"🗑️ [Sanitizer] Removing invalid file: {f}")
                        try:
                            err_dest = os.path.join(ERROR_DIR, f"INVALID_{f}")
                            shutil.move(dest_src, err_dest)
                        except: 
                            os.remove(dest_src)
                        continue
                    
                    # 3. Create Canvas
                    img = Image.open(dest_src)
                    
                    # [Phase 3] Auto RGB Convert #7
                    if img.mode != "RGB":
                        print(f"🔄 [Phase 3] Converting {img.mode} -> RGB: {f}")
                        img = img.convert("RGB")
                    
                    # [Phase 3] Minimum Size Check #6
                    if img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE:
                        print(f"⚠️ [Phase 3] Image too small ({img.width}x{img.height}): {f}")
                        try:
                            err_dest = os.path.join(ERROR_DIR, f"TOO_SMALL_{f}")
                            shutil.move(dest_src, err_dest)
                        except: pass
                        continue
                    
                    canvas = center_image(img)
                    
                    # Save Master to Project Root
                    canvas.save(os.path.join(proj_dir, f"{unique_id}_original_full.png"))
                    
                    # 4. Crop & Save Tiles to raw_tiles
                    # NOTE: Tiles must prefix with unique_id for Stage 3 to group correctly
                    out_base = os.path.join(raw_dir, unique_id + ".png") 
                    tiles, _ = crop_4_tiles(canvas, out_base)
                    
                    # 5. Issue Job Tickets to Q2
                    for t_path in tiles:
                        t_name = os.path.basename(t_path) # e.g. Ani-001_tile_0.png
                        job_name = t_name + ".job"        # Ani-001_tile_0.png.job
                        create_job_ticket(Q_GPU_JOBS, job_name)
                        
                    update_meta(proj_dir, "status", "queued_gpu")
                    print(f"✅ [Stage 1] Created Project {base_name} & 4 Job Tickets.")
                    
                except Exception as e:
                    print(f"❌ [Stage 1] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(ERROR_DIR, f))
        except Exception as e:
            print(f"❌ [Stage 1] Loop Error: {e}")
            time.sleep(1)

def worker_clean():
    print("⏳ [Stage 2] Initializing GPU Worker (High Priority)...")
    if SHUTDOWN_FLAG.is_set(): return
    
    # print("🧹 [Stage 2] GPU Worker Started.") # Removed redundant print
    processed_count = 0
    
    # [V7] Robust Init with Retry & GC
    MAX_RETRIES = 3
    flux = None
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🔌 [Stage 2] GPU Init Attempt {attempt+1}/{MAX_RETRIES}...")
            
            # Aggressive Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            flux = FluxProcessor()
            STARTUP_EVENT.set() # Release other workers
            print("✅ [Stage 2] GPU Initialized Successfully.")
            break
        except Exception as e:
            print(f"❌ [Stage 2] Init Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print("🚨 [CRITICAL] GPU Init Failed after retries. Pipeline halted.")
                return

    if not flux: return

    print("👀 [Stage 2] GPU Monitor Started (Scanning System_Queues/Q2).")
    while not SHUTDOWN_FLAG.is_set():
        # [Phase 2] Resource Guard #8
        if not check_system_health():
            time.sleep(30)
            continue
            
        try:
            # Find jobs
            job_files = sorted([f for f in os.listdir(Q_GPU_JOBS) if f.endswith(".job")])
            if not job_files:
                time.sleep(1)
                continue
            
            # --- SMART BATCH ACCUMULATION (V6: PAGE-CONSISTENT) ---
            # We always try to grab at least 4 tiles (1 full page) so pages don't get split.
            # Internal FluxProcessor logic will still respect flux.batch_size for VRAM.
            TARGET_BATCH_POOL = max(4, flux.batch_size) 
            WAIT_FOR_BATCH_SECONDS = 1.0 
            MAX_WAIT_ATTEMPTS = 15 
            
            valid_batch_jobs = []
            
            for attempt in range(MAX_WAIT_ATTEMPTS):
                # 1. Get all jobs and Group by Project
                all_raw_jobs = [f for f in os.listdir(Q_GPU_JOBS) if f.endswith(".job")]
                
                project_groups = {}
                for job in all_raw_jobs:
                    tile_filename = job.replace(".job", "")
                    if "_tile_" not in tile_filename: 
                        try: os.remove(os.path.join(Q_GPU_JOBS, job))
                        except: pass
                        continue
                    base = tile_filename.split("_tile_")[0]
                    
                    # Verify file exists
                    proj_dir = os.path.join(PROJECTS_DIR, base)
                    src_path = os.path.join(proj_dir, "raw_tiles", tile_filename)
                    if not os.path.exists(src_path):
                        try: os.remove(os.path.join(Q_GPU_JOBS, job))
                        except: pass
                        continue
                        
                    if base not in project_groups: project_groups[base] = []
                    project_groups[base].append(job)
                
                # Natural Sort Projects
                def natural_sort_key(s):
                    import re
                    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
                
                sorted_project_names = sorted(project_groups.keys(), key=natural_sort_key)
                
                # 2. Try to fill Batch with complete pages (4 tiles each)
                temp_batch = []
                for p_name in sorted_project_names:
                    p_jobs = sorted(project_groups[p_name], key=natural_sort_key)
                    # We pick if it's a complete set of 4
                    if len(p_jobs) == 4:
                        if len(temp_batch) + 4 <= TARGET_BATCH_POOL:
                            temp_batch.extend(p_jobs)
                    
                    if len(temp_batch) >= TARGET_BATCH_POOL: break
                
                # 3. Decision Logic: Go if Batch is Full
                if len(temp_batch) >= TARGET_BATCH_POOL:
                    valid_batch_jobs = temp_batch
                    break
                
                # If Batch not full, check if we should wait for more (Page Integrity)
                input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not input_files:
                    # No more work coming, gather whatever is available (even incomplete sets)
                    final_batch = []
                    for p_name in sorted_project_names:
                        p_jobs = sorted(project_groups[p_name], key=natural_sort_key)
                        final_batch.extend(p_jobs)
                        if len(final_batch) >= TARGET_BATCH_POOL: break
                    valid_batch_jobs = final_batch
                    break
                
                # Still have work coming, only go if we have a full "Chunk" (e.g. 4 for Page Consistency)
                if attempt < MAX_WAIT_ATTEMPTS - 1:
                    if attempt % 3 == 0:
                        print(f"⏳ [Stage 2] Page-Aware Gathering... ({len(temp_batch)}/{TARGET_BATCH_POOL} tiles). Waiting for more...")
                    time.sleep(WAIT_FOR_BATCH_SECONDS)
                else:
                    # Timeout, process whatever we have in temp_batch or force pick
                    if not temp_batch:
                         for p_name in sorted_project_names:
                            p_jobs = sorted(project_groups[p_name], key=natural_sort_key)
                            temp_batch.extend(p_jobs)
                            if len(temp_batch) >= TARGET_BATCH_POOL: break
                    valid_batch_jobs = temp_batch[:TARGET_BATCH_POOL]

            if not valid_batch_jobs:
                time.sleep(1)
                continue

            # CRITICAL SLICE: Never send more than TARGET_BATCH_POOL
            valid_batch_jobs = valid_batch_jobs[:TARGET_BATCH_POOL]

            if not valid_batch_jobs:
                time.sleep(1)
                continue

            print(f"📦 [Stage 2] Batch Strategy: Picking {len(valid_batch_jobs)} tiles (Pages: {set([j.split('_tile_')[0] for j in valid_batch_jobs])} | Target: {TARGET_BATCH_POOL}).")
            
            # Group valid processed data
            batch_input_images = []
            batch_job_tuples = [] 
            
            for job in valid_batch_jobs:
                tile_filename = job.replace(".job", "")
                base_name = tile_filename.split("_tile_")[0]
                proj_dir = os.path.join(PROJECTS_DIR, base_name)
                src_path = os.path.join(proj_dir, "raw_tiles", tile_filename)
                
                # Load Image
                # [Phase 2] Data Integrity #14
                if not os.path.exists(src_path):
                    print(f"👻 [Sanitizer] Zombie Job detected (File missing): {tile_filename}")
                    try: os.remove(os.path.join(Q_GPU_JOBS, job))
                    except: pass
                    continue

                try:
                    img = None
                    for attempt in range(3):
                        try:
                            img = Image.open(src_path).convert("RGB")
                            img.load()
                            break
                        except:
                            time.sleep(0.2)
                    if img is None: continue

                    # Resize STANDARD for Batching
                    img_resized = img.resize((1280, 1280), Image.Resampling.LANCZOS)
                    
                    batch_input_images.append(img_resized)
                    batch_job_tuples.append((job, base_name, proj_dir))
                    
                    # Update Start Ticket
                    # Only set gpu_start if it doesn't exist (start of project)
                    meta_p = os.path.join(proj_dir, "meta.json")
                    has_start = False
                    try:
                        if os.path.exists(meta_p):
                            with open(meta_p, 'r') as f: 
                                if "gpu_start" in json.load(f): has_start = True
                    except: pass
                    
                    if not has_start:
                        update_meta(proj_dir, "gpu_start", time.time())
                        
                    update_meta(proj_dir, "status", "processing_gpu")
                    
                except Exception as e:
                    print(f"⚠️ Load Error {job}: {e}")

            if not batch_input_images:
                time.sleep(1)
                continue

            # --- EXECUTE BATCH ---
            try:
                print(f"🎨 [Stage 2] Sending Batch of {len(batch_input_images)} to Flux...")
                results = flux.process(batch_input_images)
                
                if not isinstance(results, list): results = [results]
                
                # [Phase 3] Count Assertion #16
                if len(results) != len(batch_job_tuples):
                    print(f"🚨 [Phase 3] Output count mismatch! Expected {len(batch_job_tuples)}, Got {len(results)}")
                    # Cleanup jobs and skip this batch
                    for job_file, _, _ in batch_job_tuples:
                        job_path = os.path.join(Q_GPU_JOBS, job_file)
                        try: os.remove(job_path)
                        except: pass
                    continue
                
                # --- SAVE RESULTS ---
                for i, cleaned_img in enumerate(results):
                    job_file, base, p_dir = batch_job_tuples[i]
                    tile_file = job_file.replace(".job", "")
                    
                    dest_path = os.path.join(p_dir, "clean_tiles", tile_file)
                    
                    # Atomic Save
                    dest_tmp = dest_path + ".tmp"
                    cleaned_img.save(dest_tmp, format="PNG", quality=95)
                    os.replace(dest_tmp, dest_path)
                    
                    # Update Meta
                    update_meta(p_dir, "gpu_end", time.time())
                    
                    # Move Ticket -> Q3
                    src_job = os.path.join(Q_GPU_JOBS, job_file)
                    dst_job = os.path.join(Q_STITCH_JOBS, job_file)
                    if os.path.exists(src_job):
                        shutil.move(src_job, dst_job)
                        
                print(f"✅ [Stage 2] Batch Completed ({len(results)} tiles).")
                
            except Exception as e:
                print(f"❌ [Stage 2] Batch Process Error: {e}")
                # On error, maybe just leave jobs there to retry?
                # Cleanup Job
                for job_file, _, _ in batch_job_tuples:
                    job_path = os.path.join(Q_GPU_JOBS, job_file)
                    if os.path.exists(job_path):
                        os.remove(job_path)
                
            # [Phase 2] Periodic VRAM GC #15
            processed_count += len(batch_input_images) # Increment by batch size
            if processed_count % 10 == 0: # Check every 10 tiles processed
                print(f"♻️ [Resource] Performing Periodic GC (Processed {processed_count})...")
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ [Stage 2] Loop Error: {e}")
            time.sleep(1)

def worker_stitch():
    print("👀 [Stage 3] Waiting for System Init...")
    STARTUP_EVENT.wait()
    print("👀 [Stage 3] Stitch Monitor Started (Scanning System_Queues/Q3).")
    
    # [V7] Watchdog State
    stale_projects = {} # {base_name: first_seen_timestamp}
    PROJECT_TIMEOUT = 600 # 10 minutes
    
    PROJECT_TIMEOUT = 600 # 10 minutes
    
    while not SHUTDOWN_FLAG.is_set():
        try:
            ensure_dirs()
            jobs = [f for f in os.listdir(Q_STITCH_JOBS) if f.endswith(".job")]
            
            # Group jobs by Project/BaseName
            groups = {}
            for job in jobs:
                # 20240209_143000_Ani-001_tile_0.png.job
                # base = 20240209_143000_Ani-001
                tile_filename = job.replace(".job", "")
                base = tile_filename.split("_tile_")[0]
                if base not in groups: groups[base] = []
                groups[base].append(job)
            
            for base, job_list in groups.items():
                # 1. Check Full Page (4 tiles)
                if len(job_list) == 4:
                    if base in stale_projects: del stale_projects[base]
                    
                    print(f"🧩 [Stage 3] Stitching Project: {base}")
                    
                    try:
                        proj_dir = os.path.join(PROJECTS_DIR, base)
                        # ... process ...
                        
                        # [TIME TRACKING] Stitch Start
                        update_meta(proj_dir, "stitch_start", time.time())
                        update_meta(proj_dir, "status", "stitching")
                        
                        # Find Master
                        master_path = os.path.join(proj_dir, f"{base}_original_full.png")
                        
                        # [Phase 2] Data Integrity #22
                        if not os.path.exists(master_path):
                            print(f"🚨 [Stage 3] CRITICAL: Master File Missing for {base}. Moving to Error.")
                            # Cleanup Queue
                            for j in job_list:
                                try: os.remove(os.path.join(Q_STITCH_JOBS, j))
                                except: pass
                            # Move Project to Error
                            try:
                                err_dest = os.path.join(ERROR_DIR, f"MISSING_MASTER_{base}")
                                shutil.move(proj_dir, err_dest)
                            except: pass
                            continue

                        master_pil = Image.open(master_path).convert("RGB")
                        master_cv = cv2.cvtColor(np.array(master_pil), cv2.COLOR_RGB2BGR)
                        master_data = prepare_master_data(master_cv)
                        
                        # Process Tiles
                        results = []
                        # Sort jobs to ensure tile 0..3 order
                        job_list.sort() 
                        
                        for job in job_list:
                            tile_filename = job.replace(".job", "")
                            t_path = os.path.join(proj_dir, "clean_tiles", tile_filename)
                            
                            t_pil = Image.open(t_path).convert("RGB")
                            t_cv = cv2.cvtColor(np.array(t_pil), cv2.COLOR_RGB2BGR)
                            res = process_tile_core(t_cv, master_cv, master_data, refine=True)
                            
                            # --- 1. Parse Tile Index ---
                            # Expect format: "..._tile_0...", "..._tile_1..."
                            try:
                                if "_tile_0" in tile_filename: t_idx = 0
                                elif "_tile_1" in tile_filename: t_idx = 1
                                elif "_tile_2" in tile_filename: t_idx = 2
                                elif "_tile_3" in tile_filename: t_idx = 3
                                else: t_idx = -1
                                res["tile_index"] = t_idx
                            except:
                                res["tile_index"] = -1
                                
                            results.append(res)
                        
                        # --- SMART STITCH SELECTION ---
                        # Strategies: Focus on Left/Right Overlap (0vs1 and 2vs3)
                        strategies = [
                            (0, 1, 2, 3), # Top: 0->1, Bot: 2->3
                            (1, 0, 2, 3), # Top: 1->0, Bot: 2->3
                            (0, 1, 3, 2), # Top: 0->1, Bot: 3->2
                            (1, 0, 3, 2)  # Top: 1->0, Bot: 3->2
                        ]
                        
                        best_score = float('inf')
                        best_stacked_cv = None
                        best_strategy = None
                        
                        # Prepare Grayscale Master for MSE
                        gray_master = cv2.cvtColor(master_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)

                        # results is guaranteed sorted by filename (tile_0..3)
                        
                        for strategy in strategies:
                            # Reorder results based on strategy tuple
                            current_ordered_results = [results[i] for i in strategy]
                            
                            # Stack
                            # stack_images returns PIL
                            pil_variant = stack_images(master_cv, current_ordered_results) 
                            cv_variant = cv2.cvtColor(np.array(pil_variant), cv2.COLOR_RGB2BGR)
                            
                            # Calculate MSE (Structural Difference vs Original)
                            gray_variant = cv2.cvtColor(cv_variant, cv2.COLOR_BGR2GRAY).astype(np.float32)
                            mse = np.mean((gray_master - gray_variant) ** 2)
                            
                            if mse < best_score:
                                best_score = mse
                                best_stacked_cv = cv_variant
                                best_strategy = strategy
                        
                        print(f"     🏆 [Stage 3] Smart Stitch Selected: {best_strategy} (MSE: {best_score:.2f})")
                        stacked_cv = best_stacked_cv
                        # ------------------------------------------
                        
                        # --- Noise Filter & Feather ---
                        master_gray = cv2.cvtColor(master_cv, cv2.COLOR_BGR2GRAY)
                        stacked_gray = cv2.cvtColor(stacked_cv, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(master_gray, stacked_gray)
                        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        
                        h_img, w_img = master_gray.shape
                        total_area = h_img * w_img
                        aggressive_threshold = max(10, int(total_area * 0.00002))
                        dilate_size = max(5, int(w_img * 0.025))
                        if dilate_size % 2 == 0: dilate_size += 1
                        
                        # ... (Noise Filter Logic) ...
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
                        label_map = np.zeros(num_labels, dtype=np.uint8)
                        areas = stats[1:, cv2.CC_STAT_AREA] 
                        large_indices = np.where(areas > aggressive_threshold)[0] + 1
                        label_map[large_indices] = 255
                        large_blobs_mask = label_map[labels]
                        
                        label_map[:] = 0
                        small_indices = np.where(areas <= aggressive_threshold)[0] + 1
                        label_map[small_indices] = 255
                        small_blobs_mask = label_map[labels]
                        
                        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                        safe_zone = cv2.dilate(large_blobs_mask, dilate_kernel, iterations=1)
                        seeds = cv2.bitwise_and(small_blobs_mask, safe_zone)
                        selem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                        restored = seeds.copy()
                        for _ in range(100): 
                            dilated = cv2.dilate(restored, selem, iterations=1)
                            dilated = cv2.bitwise_and(dilated, small_blobs_mask)
                            if np.array_equal(dilated, restored): break
                            restored = dilated
                        
                        diff_mask = cv2.bitwise_or(large_blobs_mask, restored)
                        
                        # Feathering
                        diff_mask_sharp = diff_mask.copy()
                        expand_size = max(5, int(w_img * 0.08))
                        if expand_size % 2 == 0: expand_size += 1
                        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
                        diff_mask_expanded = cv2.dilate(diff_mask, expand_kernel, iterations=1)
                        
                        inverted_original = cv2.bitwise_not(diff_mask_sharp)
                        dist_from_org = cv2.distanceTransform(inverted_original, cv2.DIST_L2, 5)
                        feather_depth = max(3, int(w_img * 0.04))
                        t = np.clip(dist_from_org / feather_depth, 0, 1)
                        smoothstep = 1.0 - (3 * t**2 - 2 * t**3)
                        
                        expansion_region = cv2.subtract(diff_mask_expanded, diff_mask_sharp)
                        expansion_01 = expansion_region.astype(np.float32) / 255.0
                        feathered_expansion = expansion_01 * smoothstep
                        
                        diff_mask_final = diff_mask_sharp.astype(np.float32) / 255.0
                        diff_mask_final = np.maximum(diff_mask_final, feathered_expansion)
                        diff_mask = (diff_mask_final * 255).astype(np.uint8)
                        
                        # Composite
                        alpha = diff_mask.astype(np.float32) / 255.0
                        alpha_3ch = alpha[:, :, np.newaxis]
                        composite_f = master_cv.astype(np.float32) * (1 - alpha_3ch) + stacked_cv.astype(np.float32) * alpha_3ch
                        composite_cv = composite_f.astype(np.uint8)
                        
                        # [V2 Compat] Save _clean_full.png (stacked result before composite)
                        clean_full_path = os.path.join(proj_dir, f"{base}_clean_full.png")
                        if not safe_write_image(clean_full_path, stacked_cv):
                            print(f"⚠️ [Stage 3] Warning: Failed to save clean_full")
                        
                        # [V2 Compat] Save _mask_full.png (diff mask)
                        mask_full_path = os.path.join(proj_dir, f"{base}_mask_full.png")
                        if not safe_write_image(mask_full_path, diff_mask):
                            print(f"⚠️ [Stage 3] Warning: Failed to save mask_full")
                        
                        # [PRIMARY OUTPUT] Save _masked_composite.png
                        output_filename = f"{base}_masked_composite.png"
                        output_path = os.path.join(proj_dir, output_filename)
                        if not safe_write_image(output_path, composite_cv):
                            raise Exception(f"Failed to write output: {output_filename}")
                        
                        # [TIME TRACKING] Done
                        update_meta(proj_dir, "done_at", time.time())
                        update_meta(proj_dir, "status", "done")
                        
                        # Clean Job Tickets
                        for job in job_list:
                            job_path = os.path.join(Q_STITCH_JOBS, job)
                            if os.path.exists(job_path): os.remove(job_path)
                            
                        print(f"✅ [Stage 3] Finished: {output_filename}")

                        
                        # [Phase 3] Memory Cleanup #25
                        del master_cv, stacked_cv, composite_cv, composite_f, alpha, alpha_3ch
                        del diff_mask, diff_mask_sharp, diff_mask_expanded, diff_mask_final
                        gc.collect()
                        
                    except Exception as e:
                        print(f"❌ [Stage 3] Stitch Error {base}: {e}")
                        # Move to Error
                        if os.path.exists(proj_dir):
                            shutil.move(proj_dir, os.path.join(ERROR_DIR, base))
                        for job in job_list:
                            job_path = os.path.join(Q_STITCH_JOBS, job)
                            if os.path.exists(job_path): os.remove(job_path)

                else:
                    # [V7] Watchdog Logic for Incomplete Projects
                    if base not in stale_projects:
                        stale_projects[base] = time.time()
                    else:
                        elapsed = time.time() - stale_projects[base]
                        if elapsed > PROJECT_TIMEOUT:
                            print(f"🚨 [Stage 3] WATCHDOG TIMEOUT: Project {base} stuck for {elapsed:.0f}s. Moving to Errors.")
                            
                            # Move Project to Error
                            try:
                                p_dir = os.path.join(PROJECTS_DIR, base)
                                if os.path.exists(p_dir):
                                    err_dest = os.path.join(ERROR_DIR, f"{base}_TIMEOUT")
                                    shutil.move(p_dir, err_dest)
                            except: pass
                            
                            # Clean Jobs
                            for job in job_list:
                                try:
                                    os.remove(os.path.join(Q_STITCH_JOBS, job))
                                except: pass
                            
                            del stale_projects[base]
                # The original code had an extra print and comment here, which is now part of the else block.
                # This part was:
                # print(f"✅ [Stage 3] Project {base} COMPLETED.")
                # except Exception as e:
                #     print(f"❌ [Stage 3] Error: {e}")
                #     # Don't delete jobs on error so we can retry? Or move to error?
                #     # For now, keep them so it retries or manual intervention
        except Exception as e:
            print(f"❌ [Stage 3] Loop Error: {e}")
            time.sleep(1)





# ===================================================================
# MAIN CONTROLLER
# ===================================================================
if __name__ == "__main__":
    print("\n🏭 STARTING IMGCRAFT V2 PIPELINE (PARITY EDITION) 🏭")
    
    # [V7] Acquire Global Lock FIRST
    acquire_instance_lock()
    
    ensure_dirs()
    
    t1 = threading.Thread(target=worker_crop, daemon=True)
    t2 = threading.Thread(target=worker_clean, daemon=True)
    t3 = threading.Thread(target=worker_stitch, daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    print("✅ All Workers Started. Press Ctrl+C to stop.")
    try:
        while not SHUTDOWN_FLAG.is_set(): time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopping...")
        SHUTDOWN_FLAG.set()
