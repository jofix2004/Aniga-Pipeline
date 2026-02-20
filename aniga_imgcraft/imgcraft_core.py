# ImgCraft Core — FluxProcessor + Image Processing Functions
# Module thuần túy, được gọi bởi imgcraft_server.py

import os
import time
import sys
import cv2
import numpy as np
import gc
from PIL import Image
import random
import torch

# ===================================================================
# FLUX PROCESSOR (GPU)
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
            model = self.nodes["lora_loader"].load_lora_model_only(model, "AniGaKontext_1024x64x1605_v4_000001400.safetensors", 1.0)[0]
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

