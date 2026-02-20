
import os
import json
import time
import base64
import subprocess
import zipfile
import io
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ImgCraft V3 Ultimate")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Input")
PROJECTS_DIR = os.path.join(BASE_DIR, "Projects")
SYS_Q_DIR = os.path.join(BASE_DIR, "System_Queues")
Q_GPU_JOBS = os.path.join(SYS_Q_DIR, "Q2_GPU_Jobs")
Q_STITCH_JOBS = os.path.join(SYS_Q_DIR, "Q3_Stitch_Jobs")

# --- EMBEDDED SPA (SINGLE PAGE APP) ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImgCraft V3 Ultimate</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #09090b;
            --bg-panel: #18181b;
            --border: #27272a;
            --primary: #10b981; /* Emerald 500 */
            --primary-dim: rgba(16, 185, 129, 0.1);
            --text-main: #e4e4e7;
            --text-muted: #a1a1aa;
            --danger: #ef4444;
            --warning: #f59e0b;
            --info: #3b82f6;
        }
        * { box-sizing: border-box; }
        body {
            background-color: var(--bg-dark); color: var(--text-main);
            font-family: 'Inter', sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden;
        }
        
        /* SIDEBAR */
        aside {
            width: 260px; background: var(--bg-panel); border-right: 1px solid var(--border);
            display: flex; flex-direction: column; padding: 20px;
        }
        .brand {
            font-size: 20px; font-weight: 700; color: var(--primary); margin-bottom: 40px;
            display: flex; align-items: center; gap: 10px; letter-spacing: -0.5px;
        }
        .nav-item {
            padding: 12px 16px; margin-bottom: 8px; border-radius: 8px; cursor: pointer;
            color: var(--text-muted); font-weight: 500; transition: all 0.2s;
            display: flex; align-items: center; gap: 12px;
        }
        .nav-item:hover { background: rgba(255,255,255,0.05); color: var(--text-main); }
        .nav-item.active { background: var(--primary-dim); color: var(--primary); }
        
        /* MAIN CONTENT */
        main { flex: 1; padding: 30px; overflow-y: auto; position: relative; }
        
        .page { display: none; animation: fadeIn 0.3s ease; }
        .page.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        h1 { font-size: 24px; font-weight: 600; margin: 0 0 20px 0; }
        
        /* DASHBOARD GRID */
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .stat-card {
            background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px;
            display: flex; flex-direction: column;
        }
        .stat-label { font-size: 13px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-value { font-size: 32px; font-weight: 700; color: var(--text-main); margin-top: 8px; }
        
        /* TABLE */
        .table-container { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th { text-align: left; padding: 16px 24px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; }
        td { padding: 16px 24px; border-bottom: 1px solid var(--border); color: var(--text-main); }
        tr:last-child td { border-bottom: none; }
        tr:hover { background: rgba(255,255,255,0.02); }
        
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
        .s-done { background: var(--primary); box-shadow: 0 0 8px var(--primary); }
        .s-proc { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
        .s-init { background: var(--text-muted); }
        
        /* LIBRARY GRID */
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .card {
            background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
            cursor: pointer; transition: transform 0.2s, border-color 0.2s; position: relative;
        }
        .card:hover { transform: translateY(-4px); border-color: var(--primary); }
        .card-img { width: 100%; height: 180px; object-fit: contain; background: #000; border-bottom: 1px solid var(--border); }
        .card-body { padding: 12px; }
        .card-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .card-sub { font-size: 12px; color: var(--text-muted); display: flex; justify-content: space-between; }
        
        /* INSPECTOR MODAL */
        #inspector {
            position: fixed; inset: 0; background: var(--bg-dark); z-index: 9999;
            display: none; flex-direction: column;
        }
        #insp-header {
            height: 60px; border-bottom: 1px solid var(--border); display: flex; align-items: center;
            padding: 0 30px; justify-content: space-between; background: var(--bg-panel);
        }
        #insp-canvas { flex: 1; position: relative; background: #050505; overflow: hidden; cursor: grab; }
        
        .controls {
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            background: rgba(24, 24, 27, 0.9); backdrop-filter: blur(12px);
            padding: 12px 24px; border-radius: 100px; border: 1px solid var(--border);
            display: flex; gap: 24px; align-items: center; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }
        .btn-toggle {
            background: none; border: none; color: var(--text-main); font-weight: 600; cursor: pointer;
            display: flex; align-items: center; gap: 8px; font-size: 14px;
        }
        .btn-toggle.active { color: var(--primary); }
        
        /* UTILS */
        .btn-refresh {
            position: absolute; top: 30px; right: 30px;
            background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-main);
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px;
        }
        .btn-refresh:hover { border-color: var(--text-muted); }

        .status-badge {
            position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8);
            padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: #fff;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .status-Done { color: var(--primary); border-color: var(--primary); }
        .status-Processing, .status-processing_gpu { color: #ffaa00; border-color: #ffaa00; }
        .status-stitching { color: #8b5cf6; border-color: #8b5cf6; } /* Violet */
        .status-queued_gpu { color: #3b82f6; border-color: #3b82f6; } /* Blue */
        .status-cropping { color: #64748b; border-color: #64748b; } /* Slate */
        .status-Init { color: #aaa; border-color: #aaa; }

    </style>
</head>
<body>

    <aside>
        <div class="brand">🏭 ImgCraft V3</div>
        <div class="nav-item active" onclick="nav('dashboard', this)">
            📊 System Dashboard
        </div>
        <div class="nav-item" onclick="nav('progress', this)">
            ⏳ Detailed Progress
        </div>
        <div class="nav-item" onclick="nav('library', this)">
            📂 Visual Library
        </div>
    </aside>

    <main>
        <button class="btn-refresh" onclick="fetchData()">🔄 REFRESH</button>

        <!-- PAGE 1: DASHBOARD -->
        <div id="page-dashboard" class="page active">
            <h1>System Status</h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Input Waiting</span>
                    <span class="stat-value" id="d-input">0</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label" style="color:var(--warning)">GPU Queue</span>
                    <span class="stat-value" id="d-gpu">0</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label" style="color:var(--info)">Stitch Queue</span>
                    <span class="stat-value" id="d-stitch">0</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label" style="color:var(--primary)">Active Projects</span>
                    <span class="stat-value" id="d-proj">0</span>
                </div>
            </div>
            
            <div style="background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 24px; height: 400px;">
                <h3 style="margin-top:0">System Logs</h3>
                <div id="sys-logs" style="font-family: monospace; color: var(--text-muted); font-size: 13px; height: 320px; overflow-y: auto; white-space: pre-wrap;">
                    Logs are not currently streamed to frontend. (Feature coming soon)
                    Check Colab output for detailed logs.
                </div>
            </div>
        </div>

        <!-- PAGE 2: PROGRESS TABLE -->
        <div id="page-progress" class="page">
            <h1>Detailed Progress Tracking</h1>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Project</th>
                            <th>Status</th>
                            <th>Progress (Clean/Raw)</th>
                            <th>Age</th>
                            <th>Active Process Time</th>
                            <th>Last Update</th>
                        </tr>
                    </thead>
                    <tbody id="progress-tbody">
                        <!-- JS Render -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- PAGE 3: VISUAL LIBRARY -->
        <div id="page-library" class="page">
            <h1>Visual Library</h1>
            
            <!-- UPLOAD ZONE -->
            <div id="upload-zone" style="border: 2px dashed var(--border); border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 20px; cursor: pointer; transition: all 0.2s;" 
                 ondragover="event.preventDefault(); this.style.borderColor='var(--primary)'; this.style.background='var(--primary-dim)';"
                 ondragleave="this.style.borderColor='var(--border)'; this.style.background='transparent';"
                 ondrop="handleDrop(event)"
                 onclick="document.getElementById('file-input').click()">
                <div style="font-size: 32px; margin-bottom: 8px;">📤</div>
                <div style="color: var(--text-main); font-weight: 600;">Drop files here or click to upload</div>
                <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">PNG, JPG, WebP, BMP</div>
                <input type="file" id="file-input" multiple accept=".png,.jpg,.jpeg,.webp,.bmp" style="display:none;" onchange="handleFileSelect(event)">
            </div>
            
            <!-- TOOLBAR -->
            <div style="margin-bottom: 20px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                <label style="color: var(--text-muted); font-size: 14px;">Sort by:</label>
                <select id="sort-select" onchange="renderLibrary()" style="background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-main); padding: 8px 16px; border-radius: 6px; cursor: pointer;">
                    <option value="name-asc">Name (A-Z)</option>
                    <option value="name-desc">Name (Z-A)</option>
                    <option value="date-desc" selected>Date (Newest First)</option>
                    <option value="date-asc">Date (Oldest First)</option>
                    <option value="status">Status (Done First)</option>
                </select>
                
                <div style="flex: 1;"></div>
                
                <!-- SELECTION CONTROLS -->
                <span id="selection-count" style="color: var(--text-muted); font-size: 14px;">0 selected</span>
                <button onclick="selectAll()" style="background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-main); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500;">☑ Select All</button>
                <button onclick="clearSelection()" style="background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-muted); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500;">✖ Clear</button>
                
                <div style="width: 1px; height: 24px; background: var(--border);"></div>
                
                <button onclick="downloadSelected('full')" id="btn-dl-full" style="background: var(--primary-dim); border: 1px solid var(--primary); color: var(--primary); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; opacity: 0.5;" disabled>📦 Download Full</button>
                <button onclick="downloadSelected('results')" id="btn-dl-results" style="background: var(--primary); border: none; color: #000; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; opacity: 0.5;" disabled>🖼️ Download Results</button>
            </div>
            
            <div class="grid" id="library-grid" onclick="handleGridClick(event)">
                <!-- JS Render -->
            </div>
        </div>

    </main>

    <!-- INSPECTOR OVERLAY -->
    <div id="inspector">
        <div id="insp-header">
            <div style="font-weight: 700; font-size: 18px; display:flex; gap:10px; align-items:center;">
                <button onclick="closeInsp()" style="background:transparent; border:1px solid var(--border); color:var(--text-main); padding:6px 12px; border-radius:6px; cursor:pointer;">⬅ BACK</button>
                <span id="insp-title" style="color:var(--primary)">Project Name</span>
            </div>
            <div style="font-size: 13px; color: var(--text-muted);">SCROLL to Zoom • DRAG to Pan • SPACE to Toggle</div>
        </div>
        <div id="insp-canvas">
            <div id="img-wrapper" style="position:absolute; top:0; left:0; transform-origin:0 0;">
                <img id="img-bg" style="position:absolute; top:0; left:0; pointer-events:none;">
                <img id="img-fg" style="position:absolute; top:0; left:0; pointer-events:none; transition: opacity 0.05s;">
            </div>
            
            <div class="controls">
                <button class="btn-toggle active" id="btn-toggle" onclick="toggleLayer()">
                    <span>👁️</span> Showing Result
                </button>
                <div style="width:1px; height:20px; background:var(--border)"></div>
                <span id="zoom-val" style="color:var(--text-muted); font-variant-numeric: tabular-nums;">100%</span>
            </div>
        </div>
    </div>

    <script>
        // --- NAVIGATION ---
        function nav(pageId, el) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById('page-' + pageId).classList.add('active');
            
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            if(el) el.classList.add('active');
        }

        // --- DATA FETCHING ---
        let projects = [];
        let selectedProjects = new Set();
        let lastSelectedIndex = -1;
        
        // --- UPLOAD HANDLERS ---
        async function uploadFiles(files) {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            
            const zone = document.getElementById('upload-zone');
            zone.innerHTML = '<div style="color:var(--primary);">⏳ Uploading...</div>';
            
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                zone.innerHTML = `
                    <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
                    <div style="color: var(--primary); font-weight: 600;">Uploaded ${data.uploaded} file(s)</div>
                    <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Refreshing...</div>
                `;
                setTimeout(() => {
                    fetchData();
                    zone.innerHTML = `
                        <div style="font-size: 32px; margin-bottom: 8px;">📤</div>
                        <div style="color: var(--text-main); font-weight: 600;">Drop files here or click to upload</div>
                        <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">PNG, JPG, WebP, BMP</div>
                        <input type="file" id="file-input" multiple accept=".png,.jpg,.jpeg,.webp,.bmp" style="display:none;" onchange="handleFileSelect(event)">
                    `;
                }, 1500);
            } catch(e) {
                zone.innerHTML = '<div style="color:var(--danger);">❌ Upload Error</div>';
            }
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.target.style.borderColor = 'var(--border)';
            e.target.style.background = 'transparent';
            if (e.dataTransfer.files.length > 0) {
                uploadFiles(e.dataTransfer.files);
            }
        }
        
        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                uploadFiles(e.target.files);
            }
        }
        
        // --- SELECTION HANDLERS ---
        function handleGridClick(e) {
            const card = e.target.closest('.card');
            if (!card) {
                // Clicked empty space - clear selection
                clearSelection();
                return;
            }
            
            const name = card.dataset.name;
            const idx = parseInt(card.dataset.index);
            
            if (e.detail === 2) {
                // Double click - open inspector
                const proj = projects.find(p => p.name === name);
                if (proj) openInspector(proj);
                return;
            }
            
            // Single click - selection logic
            if (e.ctrlKey || e.metaKey) {
                // Ctrl+Click: Toggle individual
                if (selectedProjects.has(name)) {
                    selectedProjects.delete(name);
                } else {
                    selectedProjects.add(name);
                }
                lastSelectedIndex = idx;
            } else if (e.shiftKey && lastSelectedIndex >= 0) {
                // Shift+Click: Range select
                const sortedCards = Array.from(document.querySelectorAll('#library-grid .card'));
                const minIdx = Math.min(lastSelectedIndex, idx);
                const maxIdx = Math.max(lastSelectedIndex, idx);
                for (let i = minIdx; i <= maxIdx; i++) {
                    const cardName = sortedCards[i]?.dataset?.name;
                    if (cardName) selectedProjects.add(cardName);
                }
            } else {
                // Normal click: Toggle this one only
                if (selectedProjects.has(name) && selectedProjects.size === 1) {
                    selectedProjects.clear();
                } else {
                    selectedProjects.clear();
                    selectedProjects.add(name);
                }
                lastSelectedIndex = idx;
            }
            
            updateSelectionUI();
        }
        
        function selectAll() {
            projects.forEach(p => selectedProjects.add(p.name));
            updateSelectionUI();
        }
        
        function clearSelection() {
            selectedProjects.clear();
            lastSelectedIndex = -1;
            updateSelectionUI();
        }
        
        function updateSelectionUI() {
            const count = selectedProjects.size;
            document.getElementById('selection-count').innerText = `${count} selected`;
            
            const btnFull = document.getElementById('btn-dl-full');
            const btnResults = document.getElementById('btn-dl-results');
            
            if (count > 0) {
                btnFull.disabled = false;
                btnFull.style.opacity = '1';
                btnResults.disabled = false;
                btnResults.style.opacity = '1';
            } else {
                btnFull.disabled = true;
                btnFull.style.opacity = '0.5';
                btnResults.disabled = true;
                btnResults.style.opacity = '0.5';
            }
            
            renderLibrary();
        }
        
        // --- DOWNLOAD ---
        async function downloadSelected(mode) {
            const arr = Array.from(selectedProjects);
            if (arr.length === 0) return;
            
            try {
                const res = await fetch('/api/download', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ projects: arr, mode: mode })
                });
                
                if (!res.ok) throw new Error('Download failed');
                
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = res.headers.get('Content-Disposition')?.split('filename=')[1] || 'download.zip';
                a.click();
                URL.revokeObjectURL(url);
            } catch(e) {
                alert('Download error: ' + e.message);
            }
        }
        

        async function fetchData() {
            try {
                const [statsRes, projRes] = await Promise.all([ fetch('/api/stats'), fetch('/api/projects') ]);
                const stats = await statsRes.json();
                projects = await projRes.json();
                
                // Update Dashboard Elements
                document.getElementById('d-input').innerText = stats.input_count;
                document.getElementById('d-gpu').innerText = stats.gpu_queue;
                document.getElementById('d-stitch').innerText = stats.stitch_queue;
                document.getElementById('d-proj').innerText = stats.active_projects;
                
                renderProgressTable();
                renderLibrary();
                
            } catch(e) { console.error(e); }
        }

        function renderProgressTable() {
            const tbody = document.getElementById('progress-tbody');
            tbody.innerHTML = '';
            projects.forEach(p => {
                const tr = document.createElement('tr');
                let dotClass = 's-init';
                if(p.status === 'Done') dotClass = 's-done';
                if(p.status === 'Processing') dotClass = 's-proc';
                
                tr.innerHTML = `
                    <td style="font-weight:600">${p.name}</td>
                    <td><span class="status-dot ${dotClass}"></span>${p.status}</td>
                    <td>${p.progress}</td>
                    <td style="color:var(--text-muted)">${p.age}</td>
                    <td style="color:var(--primary)">${p.active_time}</td>
                    <td style="color:var(--text-muted)">${p.last_update}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderLibrary() {
            const grid = document.getElementById('library-grid');
            grid.innerHTML = '';
            
            // Get sort option
            const sortBy = document.getElementById('sort-select').value;
            
            // Clone and sort projects
            let sorted = [...projects];
            
            // Helper: extract original name from "YYYYMMDD_HHMMSS_originalname"
            function getOriginalName(name) {
                const parts = name.split('_');
                if (parts.length >= 3) {
                    return parts.slice(2).join('_'); // Everything after timestamp
                }
                return name;
            }
            
            switch(sortBy) {
                case 'name-asc':
                    sorted.sort((a, b) => getOriginalName(a.name).localeCompare(getOriginalName(b.name), undefined, {numeric: true}));
                    break;
                case 'name-desc':
                    sorted.sort((a, b) => getOriginalName(b.name).localeCompare(getOriginalName(a.name), undefined, {numeric: true}));
                    break;
                case 'date-desc':
                    sorted.sort((a, b) => b.name.localeCompare(a.name));
                    break;
                case 'date-asc':
                    sorted.sort((a, b) => a.name.localeCompare(b.name));
                    break;
                case 'status':
                    const order = {'Done': 0, 'stitching': 1, 'processing_gpu': 2, 'queued_gpu': 3, 'cropping': 4};
                    sorted.sort((a, b) => (order[a.status] || 99) - (order[b.status] || 99));
                    break;
            }
            
            sorted.forEach((p, idx) => {
                const card = document.createElement('div');
                card.className = 'card';
                card.dataset.name = p.name;
                card.dataset.index = idx;
                
                // Selection state
                if (selectedProjects.has(p.name)) {
                    card.style.borderColor = 'var(--primary)';
                    card.style.boxShadow = '0 0 0 2px var(--primary)';
                }
                
                const thumb = p.thumbnail ? p.thumbnail : `https://via.placeholder.com/300x200/18181b/333?text=${p.name}`;
                
                card.innerHTML = `
                    ${selectedProjects.has(p.name) ? '<div style="position:absolute;top:8px;left:8px;background:var(--primary);color:#000;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;z-index:2;">✓</div>' : ''}
                    <span class="status-badge status-${p.status}">${p.status}</span>
                    <img src="${thumb}" class="card-img" loading="lazy">
                    <div class="card-body">
                        <div class="card-title">${p.name}</div>
                        <div class="card-sub">
                            <span>${p.status}</span>
                            <span>${p.progress}</span>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        
        // --- INSPECTOR LOGIC ---
        let currentP = null;
        let showRes = true;
        let scale=1, px=0, py=0, isDrag=false, sx=0, sy=0;
        
        function openInspector(p) {
            currentP = p;
            document.getElementById('inspector').style.display = 'flex';
            document.getElementById('insp-title').innerText = p.name;
            
            // Reset
            scale=1; px=0; py=0; updateTx();
            
            const bg = document.getElementById('img-bg');
            const fg = document.getElementById('img-fg');
            
            bg.src = `/files/Projects/${p.name}/${p.name}_original_full.png`;
            if(p.has_result) {
                fg.src = `/files/Projects/${p.name}/${p.name}_masked_composite.png`;
                fg.style.display = 'block'; showRes = true;
            } else {
                fg.style.display = 'none'; showRes = false;
            }
            updateUI();
        }
        
        function closeInsp() { document.getElementById('inspector').style.display = 'none'; }
        
        function toggleLayer() {
            if(!currentP || !currentP.has_result) return;
            showRes = !showRes;
            document.getElementById('img-fg').style.opacity = showRes ? 1 : 0;
            updateUI();
        }
        
        function updateUI() {
            const btn = document.getElementById('btn-toggle');
            if(showRes) {
                btn.classList.add('active');
                btn.innerHTML = `<span>👁️</span> Showing Result`;
            } else {
                btn.classList.remove('active');
                btn.innerHTML = `<span>🚫</span> Showing Original`;
            }
        }
        
        // CANVAS INTERACTION
        const canvas = document.getElementById('insp-canvas');
        const wrap = document.getElementById('img-wrapper');
        
        function updateTx() {
            wrap.style.transform = `translate(${px}px, ${py}px) scale(${scale})`;
            document.getElementById('zoom-val').innerText = Math.round(scale*100) + "%";
        }
        
        canvas.onmousedown = e => { e.preventDefault(); isDrag=true; sx=e.clientX-px; sy=e.clientY-py; canvas.style.cursor='grabbing'; };
        window.onmouseup = () => { isDrag=false; canvas.style.cursor='grab'; };
        window.onmousemove = e => { if(!isDrag)return; e.preventDefault(); px=e.clientX-sx; py=e.clientY-sy; updateTx(); };
        
        canvas.onwheel = e => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const xs = (e.clientX - rect.left - px) / scale;
            const ys = (e.clientY - rect.top - py) / scale;
            
            const d = -e.deltaY;
            (d>0) ? (scale*=1.15) : (scale/=1.15);
            if(scale<0.05) scale=0.05; if(scale>50) scale=50;
            
            px = e.clientX - rect.left - xs*scale;
            py = e.clientY - rect.top - ys*scale;
            updateTx();
        };
        
        document.onkeydown = e => {
            if(document.getElementById('inspector').style.display === 'flex') {
                if(e.code==='Space') { e.preventDefault(); toggleLayer(); }
                if(e.code==='Escape') closeInsp();
            }
        };

        // INIT
        fetchData();
        setInterval(fetchData, 5000);

    </script>
</body>
</html>
"""

# --- HELPERS ---
def get_file_count(path):
    if not os.path.exists(path): return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.job'))])

def get_queue_details(path):
    if not os.path.exists(path): return []
    files = [f for f in os.listdir(path) if f.endswith('.job')]
    if not files: return []
    
    groups = {}
    for f in files:
        clean_name = f.replace(".job", "")
        if "_tile_" in clean_name:
            base = clean_name.split("_tile_")[0]
            groups[base] = groups.get(base, 0) + 1
        else:
            groups[f] = 1
    return [{"name": k, "count": v} for k, v in groups.items()]

# --- MODELS ---
class PipelineStats(BaseModel):
    input_count: int
    gpu_queue: int
    stitch_queue: int
    active_projects: int
    gpu_details: List[dict]
    stitch_details: List[dict]

class ProjectInfo(BaseModel):
    name: str
    status: str
    progress: str
    thumbnail: Optional[str]
    has_original: bool
    has_result: bool
    age: str
    active_time: str
    last_update: str

# --- API ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.get("/api/stats", response_model=PipelineStats)
async def get_stats():
    proj_c = 0
    if os.path.exists(PROJECTS_DIR):
        proj_c = len([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))])

    return PipelineStats(
        input_count=get_file_count(INPUT_DIR),
        gpu_queue=get_file_count(Q_GPU_JOBS),
        stitch_queue=get_file_count(Q_STITCH_JOBS),
        active_projects=proj_c,
        gpu_details=get_queue_details(Q_GPU_JOBS),
        stitch_details=get_queue_details(Q_STITCH_JOBS)
    )

@app.get("/api/projects", response_model=List[ProjectInfo])
async def get_projects():
    if not os.path.exists(PROJECTS_DIR): return []
    data = []
    try:
        projects = [p for p in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, p))]
        for p in projects:
            p_dir = os.path.join(PROJECTS_DIR, p)
            
            # Paths
            orig = os.path.join(p_dir, f"{p}_original_full.png")
            comp = os.path.join(p_dir, f"{p}_masked_composite.png")
            orig_exists = os.path.exists(orig)
            comp_exists = os.path.exists(comp)
            
            # Metadata
            meta_path = os.path.join(p_dir, "meta.json")
            meta = {}
            if os.path.exists(meta_path):
                try: 
                    with open(meta_path, 'r') as f: meta = json.load(f)
                except: pass

            # Time Logic
            now = time.time()
            created_at = meta.get("created_at", os.path.getctime(p_dir))
            
            def fmt(s): 
                if s < 60: return f"{int(s)}s"
                val = int(s)
                h = val // 3600
                m = (val % 3600) // 60
                return f"{h}h {m}m" if h > 0 else f"{m}m {val%60}s"

            age = fmt(now - created_at)
            
            gpu_dur = 0
            stitch_dur = 0
            
            # Check if project is done
            is_done = comp_exists or meta.get("status") == "done"
            done_time = meta.get("done_at", meta.get("completed_at", now))
            
            if "gpu_start" in meta:
                if "gpu_end" in meta:
                    gpu_dur = meta["gpu_end"] - meta["gpu_start"]
                elif is_done:
                    gpu_dur = done_time - meta["gpu_start"]
                else:
                    gpu_dur = now - meta["gpu_start"]
            
            if "stitch_start" in meta:
                if "stitch_end" in meta:
                    stitch_dur = meta["stitch_end"] - meta["stitch_start"]
                elif is_done:
                    stitch_dur = done_time - meta["stitch_start"]
                else:
                    stitch_dur = now - meta["stitch_start"]

            active_time = fmt(gpu_dur + stitch_dur)
            
            # Status Logic (Refined)
            raw_status = meta.get("status", "init")
            
            if comp_exists:
                status = "Done"
            elif raw_status == "done": # In case file not synced but meta says done
                status = "Done"
            elif raw_status in ["processing_gpu", "stitching", "cropping", "queued_gpu"]:
                status = raw_status 
            elif "processing" in raw_status:
                status = "Processing"
            elif not orig_exists:
                status = "Init"
            else:
                status = raw_status
            
            # Thumbnail & Placeholders
            thumb_url = None
            if comp_exists:
                thumb_url = f"/files/Projects/{p}/{p}_masked_composite.png"
            elif orig_exists:
                thumb_url = f"/files/Projects/{p}/{p}_original_full.png"
                
            # Progress
            raw_c = get_file_count(os.path.join(p_dir, "raw_tiles")) if os.path.exists(os.path.join(p_dir, "raw_tiles")) else 0
            clean_c = get_file_count(os.path.join(p_dir, "clean_tiles")) if os.path.exists(os.path.join(p_dir, "clean_tiles")) else 0
            progress = f"{clean_c}/{raw_c}"

            data.append(ProjectInfo(
                name=p,
                status=status,
                progress=progress,
                thumbnail=thumb_url,
                has_original=orig_exists,
                has_result=comp_exists,
                age=age,
                active_time=active_time,
                last_update=time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(p_dir)))
            ))
    except Exception as e:
        print(f"Error listing projects: {e}")
        
    # Sort by recent
    data.sort(key=lambda x: x.last_update, reverse=True)
    return data

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    # Securely serve files from BASE_DIR
    full_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="File not found")

# --- UPLOAD ENDPOINT ---
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple files to Input folder"""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
    
    results = []
    for file in files:
        try:
            # Validate extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                results.append({"file": file.filename, "status": "error", "message": "Invalid format"})
                continue
            
            # Save file
            dest_path = os.path.join(INPUT_DIR, file.filename)
            
            # Handle duplicates
            if os.path.exists(dest_path):
                base, extension = os.path.splitext(file.filename)
                dest_path = os.path.join(INPUT_DIR, f"{base}_{int(time.time())}{extension}")
            
            with open(dest_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            results.append({"file": file.filename, "status": "ok"})
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})
    
    return {"uploaded": len([r for r in results if r["status"] == "ok"]), "results": results}

# --- DOWNLOAD ENDPOINT ---
class DownloadRequest(BaseModel):
    projects: List[str]
    mode: str  # "full" or "results"

@app.post("/api/download")
async def download_projects(req: DownloadRequest):
    """Download selected projects as ZIP"""
    if not req.projects:
        raise HTTPException(status_code=400, detail="No projects selected")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for proj_name in req.projects:
            proj_dir = os.path.join(PROJECTS_DIR, proj_name)
            if not os.path.exists(proj_dir):
                continue
            
            if req.mode == "full":
                # Add entire project folder
                for root, dirs, files in os.walk(proj_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, PROJECTS_DIR)
                        zf.write(file_path, arcname)
            else:  # results only
                # Add only _masked_composite.png
                result_file = os.path.join(proj_dir, f"{proj_name}_masked_composite.png")
                if os.path.exists(result_file):
                    zf.write(result_file, f"{proj_name}_masked_composite.png")
    
    zip_buffer.seek(0)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "full" if req.mode == "full" else "results"
    filename = f"imgcraft_{mode_suffix}_{timestamp}.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
