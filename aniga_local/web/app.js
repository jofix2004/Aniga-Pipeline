// ── STATE ──
let CFile = null, CProj = null;
let pvIdx = -1, pvLayer = 'BLEND';
let pvImgs = {}, pvBoxes = [];
let pvZ = 1, pvX = 0, pvY = 0, pvW = 0, pvH = 0;
let dragging = false, dragSX = 0, dragSY = 0;

const COLORS = { text: '#00ff00', text2: '#88ff88', sfx: '#ff00ff', b1: '#ff0000', b2: '#0000ff', b3: '#ffff00', b4: '#00ffff', b5: '#ff8800' };

// Brush state
let tool = 'move', brushSize = 30, brushSoft = 50;
let maskCanvas = null, maskCtx = null;
let undoStack = [], redoStack = [];
let maskDirty = false, painting = false;
let rawPixels = null, cleanPixels = null;
let blendCanvas = null, blendCtx = null;

function toast(m, d = 2500) {
    const t = document.getElementById('toast');
    t.textContent = m; t.style.display = 'block';
    clearTimeout(t._t);
    t._t = setTimeout(() => t.style.display = 'none', d);
}
function showModal(id) { document.getElementById(id).classList.add('active'); }
function hideModals() { document.querySelectorAll('.modal-bg').forEach(m => m.classList.remove('active')); }

// ── HOME ──
async function loadList() {
    const r = await fetch('/api/projects'); const ps = await r.json();
    const el = document.getElementById('projectList');
    if (!ps.length) { el.innerHTML = '<div class="msg">Chưa có dự án</div>'; return; }
    el.innerHTML = ps.map(p => p.error
        ? `<div class="pcard"><div class="icon">⚠️</div><div class="info"><span class="name">${p.filename}</span><div class="meta">${p.error}</div></div></div>`
        : `<div class="pcard" onclick="openProj('${p.filename}')">
            <div class="icon">📁</div>
            <div class="info">
                <span class="name">${p.project_name}</span>
                <div class="meta"><span class="tag">${p.project_id}</span><span>${p.page_count} trang</span><span style="color:var(--green)">C:${p.clean_progress}</span><span style="color:var(--blue)">M:${p.mask_progress}</span></div>
            </div>
            <button class="del" onclick="event.stopPropagation();delProj('${p.filename}')" title="Xóa">🗑️</button>
        </div>`
    ).join('');
}

async function delProj(fn) { if (!confirm('Xóa dự án?')) return; await fetch('/api/projects/' + fn, { method: 'DELETE' }); loadList(); toast('✅ Đã xóa'); }

function goHome() {
    document.getElementById('homeView').style.display = '';
    document.getElementById('editorView').style.display = 'none';
    document.getElementById('backBtn').style.display = 'none';
    document.getElementById('topTitle').textContent = '🏠 Aniga Local';
    CFile = null; CProj = null; loadList();
}

async function openProj(fn) {
    CFile = fn;
    document.getElementById('topTitle').textContent = '⏳ Đang mở...';
    // Preload
    fetch('/api/projects/' + fn + '/preload', { method: 'POST' });
    const r = await fetch('/api/projects/' + fn + '/open', { method: 'POST' });
    CProj = await r.json();
    renderEditor();
    document.getElementById('homeView').style.display = 'none';
    document.getElementById('editorView').style.display = '';
    document.getElementById('backBtn').style.display = '';
    document.getElementById('topTitle').textContent = '📂 ' + CProj.project_name;
}

function renderEditor() {
    const p = CProj;
    document.getElementById('sbName').textContent = p.project_name;
    document.getElementById('sbMeta').innerHTML = `<span class="tag">${p.project_id}</span> &nbsp; ${p.page_count} trang`;
    const fs = (p.imgcraft_config && p.imgcraft_config.flux_size) || 1280;
    document.getElementById('fluxSel').value = fs;
    const g = document.getElementById('pageGrid');
    g.innerHTML = p.pages.map((pg, i) => {
        const dn = pg.display_name.split('_').pop();
        return `<div class="pgcard" onclick="pvOpen(${i})">
            <div class="thumb"><img src="${pg.urls.raw}" loading="lazy" onerror="this.style.display='none'"></div>
            <div class="pname">${dn}</div>
            <div class="badges">
                <span class="${pg.has_clean ? 'bd-ok' : 'bd-no'}">C</span>
                <span class="${pg.has_mask ? 'bd-ok' : 'bd-no'}">M</span>
                <span class="${pg.has_detections ? 'bd-ok' : 'bd-no'}">D</span>
            </div>
        </div>`;
    }).join('');

    // Khởi động poll status của Aniga3 liên tục khi ở trong Editor
    pollAniga3Status();
}

// ── ANIGA3 AUTO-MASK ──
let a3PollTimer = null;

async function startAniga3() {
    if (!CFile) return;
    const mode = prompt("Chế độ chạy:\n- Nhập 'resume' để chạy các trang chưa có mask.\n- Nhập 'all_force' để chạy lại toàn bộ.\n- Nhập 'errors' để chạy lại trang lỗi.", "resume");
    if (!mode) return;

    const config = {
        mask_classes: [],
        ocr_mode: document.getElementById('a3Ocr') ? document.getElementById('a3Ocr').value : "Không bật"
    };

    // AN TOÀN - CHECK NULL ĐỂ KHÔNG CHẾT JS DO Cache Trình Duyệt
    if (document.getElementById('a3Text') && document.getElementById('a3Text').checked) config.mask_classes.push('text');
    if (document.getElementById('a3SfxReal') && document.getElementById('a3SfxReal').checked) config.mask_classes.push('text2');

    // Check fallback catch id cũ đề phòng cache
    const bubblesNode = document.getElementById('a3Bubbles') || document.getElementById('a3Sfx');
    if (bubblesNode && bubblesNode.checked) config.mask_classes.push('b1', 'b2', 'b3', 'b4', 'b5');

    console.log("Bat dau goi API start với class: ", config.mask_classes);
    document.getElementById('btnA3Start').disabled = true;
    try {
        const r = await fetch(`/api/aniga3/${CFile}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, config })
        });
        const res = await r.json();
        if (!r.ok) { toast('⚠️ ' + (res.detail || "Error")); document.getElementById('btnA3Start').disabled = false; return; }
        toast('🚀 Đã bắt đầu tiến trình Auto-Mask');
        pollAniga3Status();
    } catch (e) {
        toast('❌ Lỗi kết nối AI Core');
        document.getElementById('btnA3Start').disabled = false;
    }
}

async function stopAniga3() {
    if (!CFile) return;
    document.getElementById('btnA3Stop').disabled = true;
    document.getElementById('a3StatusText').textContent = "Đang chờ dừng...";
    await fetch(`/api/aniga3/${CFile}/stop`, { method: 'POST' });
}

async function pollAniga3Status() {
    if (!CFile) return;
    clearTimeout(a3PollTimer);
    try {
        const r = await fetch(`/api/aniga3/${CFile}/status`);
        const st = await r.json();

        const panel = document.getElementById('aniga3Panel');
        if (!st.is_available) {
            panel.style.opacity = '0.5';
            document.getElementById('btnA3Start').disabled = true;
            document.getElementById('btnA3Start').textContent = "Thiếu Core";
            return;
        }

        const box = document.getElementById('a3StatusBox');
        const btnStart = document.getElementById('btnA3Start');
        const btnStop = document.getElementById('btnA3Stop');

        if (st.is_running) {
            box.style.display = 'block';
            btnStart.style.display = 'none';
            btnStop.style.display = 'block';
            btnStop.disabled = st.is_stopping;

            document.getElementById('a3Config').style.display = 'none';
            document.getElementById('a3StatusText').textContent = st.message;
            document.getElementById('a3ProgressText').textContent = `${st.done_pages}/${st.total_pages}`;

            const pct = st.total_pages > 0 ? Math.round((st.done_pages / st.total_pages) * 100) : 0;
            document.getElementById('a3Pct').textContent = `${pct}%`;
            document.getElementById('a3ProgressBar').style.width = `${pct}%`;

            a3PollTimer = setTimeout(pollAniga3Status, 2000);
        } else {
            // Vừa dừng xong
            if (box.style.display === 'block') {
                box.style.display = 'none';
                btnStart.style.display = 'block';
                btnStop.style.display = 'none';
                document.getElementById('a3Config').style.display = 'block';
                btnStart.disabled = false;

                // Refresh UI để lấy mask mới
                const scrollY = window.scrollY;
                await openProj(CFile);
                window.scrollTo(0, scrollY);
                if (st.message !== "Sẵn sàng") toast('ℹ️ Aniga3: ' + st.message);
            }
        }
    } catch (e) {
        console.error("Lỗi poll status:", e);
    }
}

// ── PREVIEW (DUAL PANE) ──
async function pvOpen(idx) {
    if (maskDirty) await autoSaveMask();

    pvIdx = idx;
    const pg = CProj.pages[idx];
    pvImgs = {}; pvBoxes = [];
    if (pvLayer === 'RAW') pvLayer = 'BLEND';
    pvZ = 1; pvX = 0; pvY = 0;
    setTool('move'); // CHỐNG VẼ LẦM: mặc định mỗi lần mở là move

    document.getElementById('pvTitle').textContent = pg.display_name;
    document.getElementById('pvOverlay').classList.add('active');

    const promises = [];
    promises.push(loadImg(pg.urls.raw).then(img => { pvImgs.RAW = img; pvW = img.width; pvH = img.height; }));
    if (pg.urls.clean) promises.push(loadImg(pg.urls.clean).then(img => pvImgs.CLEAN = img));
    if (pg.urls.mask) promises.push(loadImg(pg.urls.mask).then(img => pvImgs.MASK = img));
    if (pg.urls.detections) promises.push(fetch(pg.urls.detections).then(r => r.json()).then(d => { if (d && d.boxes) pvBoxes = d.boxes; }).catch(() => { }));
    await Promise.all(promises);

    if (pvImgs.RAW && pvBoxes.length) pvImgs.BBOX = buildBBox(pvImgs.RAW, pvBoxes);

    maskCanvas = document.createElement('canvas'); maskCanvas.width = pvW; maskCanvas.height = pvH;
    maskCtx = maskCanvas.getContext('2d');
    if (pvImgs.MASK) maskCtx.drawImage(pvImgs.MASK, 0, 0, pvW, pvH);
    else { maskCtx.fillStyle = '#000'; maskCtx.fillRect(0, 0, pvW, pvH); }
    undoStack = []; redoStack = []; maskDirty = false;
    pushUndo();

    rawPixels = null; cleanPixels = null; blendCanvas = null; blendCtx = null;
    if (pvImgs.RAW && pvImgs.CLEAN) {
        const tc = document.createElement('canvas'); tc.width = pvW; tc.height = pvH;
        const tctx = tc.getContext('2d');
        tctx.drawImage(pvImgs.RAW, 0, 0);
        rawPixels = tctx.getImageData(0, 0, pvW, pvH).data;
        tctx.drawImage(pvImgs.CLEAN, 0, 0, pvW, pvH);
        cleanPixels = tctx.getImageData(0, 0, pvW, pvH).data;
        blendCanvas = document.createElement('canvas'); blendCanvas.width = pvW; blendCanvas.height = pvH;
        blendCtx = blendCanvas.getContext('2d');
        rebuildFullBlend();
    }

    const layers = [];
    if (pvImgs.CLEAN) layers.push('CLEAN');
    if (pvImgs.MASK) layers.push('MASK');
    if (rawPixels && cleanPixels) layers.push('BLEND');
    if (pvImgs.BBOX) layers.push('BBOX');
    if (!layers.includes(pvLayer)) pvLayer = layers.length ? layers[Math.max(0, layers.length - 2)] : 'CLEAN';

    document.getElementById('layerBtns').innerHTML = layers.map(l => `<button class="pv-btn${l === pvLayer ? ' active' : ''}" onclick="pvSwitch('${l}',this)">${l}</button>`).join('');

    pvSwitch(pvLayer, null, true); // update UI
    pvFit(); pvDraw();
}

function loadImg(url) { return new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = no; i.src = url; }); }

function buildBBox(raw, boxes) {
    const w = raw.width, h = raw.height, c = document.createElement('canvas'); c.width = w; c.height = h;
    const ctx = c.getContext('2d'); ctx.drawImage(raw, 0, 0);
    for (const b of boxes) {
        const [x1, y1, x2, y2] = b.bbox; const cl = b['class']; const col = COLORS[cl] || '#fff';
        ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const lb = `${cl} ${(b.confidence * 100).toFixed(0)}%`; ctx.font = 'bold 14px Inter,sans-serif';
        const tw = ctx.measureText(lb).width; ctx.fillStyle = col; ctx.fillRect(x1, y1 - 20, tw + 8, 20);
        ctx.fillStyle = '#000'; ctx.fillText(lb, x1 + 4, y1 - 5);
    }
    return c;
}

function pvSwitch(layer, btn, skipDraw) {
    pvLayer = layer;
    document.querySelectorAll('#layerBtns .pv-btn').forEach(b => {
        if (b.textContent === layer) b.classList.add('active');
        else b.classList.remove('active');
    });
    document.getElementById('pvLabelR').textContent = layer;
    const showBrush = (layer === 'MASK' || layer === 'BLEND');
    document.getElementById('brushBar').classList.toggle('active', showBrush);

    // Xoá chức năng tự chọn bút để tránh user kéo thả bị dính vẽ nhầm
    if (!showBrush) setTool('move');
    if (layer === 'BLEND' && blendCanvas) rebuildFullBlend();
    if (!skipDraw) pvDraw();
}

let _pvRaf = 0;
let _pvCtxL = null, _pvCtxR = null;
let _pvCW_L = 0, _pvCH_L = 0, _pvCW_R = 0, _pvCH_R = 0;

function pvDraw() {
    if (_pvRaf) return;
    _pvRaf = requestAnimationFrame(_pvDrawNow);
}

function _pvDrawNow() {
    _pvRaf = 0;
    const cvL = document.getElementById('pvCanvasL'), cvR = document.getElementById('pvCanvasR');
    const paneL = document.getElementById('pvPaneL'), paneR = document.getElementById('pvPaneR');

    const awL = paneL.clientWidth, ah = paneL.clientHeight;
    const awR = paneR.clientWidth;

    if (_pvCW_L !== awL || _pvCH_L !== ah) { cvL.width = awL; cvL.height = ah; _pvCW_L = awL; _pvCH_L = ah; _pvCtxL = null; }
    if (_pvCW_R !== awR || _pvCH_R !== ah) { cvR.width = awR; cvR.height = ah; _pvCW_R = awR; _pvCH_R = ah; _pvCtxR = null; }

    if (!_pvCtxL) _pvCtxL = cvL.getContext('2d');
    if (!_pvCtxR) _pvCtxR = cvR.getContext('2d');

    const ctxL = _pvCtxL, ctxR = _pvCtxR;
    ctxL.clearRect(0, 0, awL, ah);
    ctxR.clearRect(0, 0, awR, ah);

    ctxL.imageSmoothingEnabled = ctxR.imageSmoothingEnabled = pvZ < 1;

    // Trái luôn là RAW
    if (pvImgs.RAW) {
        ctxL.save(); ctxL.translate(pvX, pvY); ctxL.scale(pvZ, pvZ);
        ctxL.drawImage(pvImgs.RAW, 0, 0);
        ctxL.restore();
    }

    // Phải là Layer hiện tại
    const srcR = pvImgs[pvLayer] || pvImgs.RAW;
    if (srcR) {
        ctxR.save(); ctxR.translate(pvX, pvY); ctxR.scale(pvZ, pvZ);
        if (pvLayer === 'MASK') {
            ctxR.drawImage(maskCanvas, 0, 0);
        } else if (pvLayer === 'BLEND' && blendCanvas) {
            ctxR.drawImage(blendCanvas, 0, 0);
        } else {
            ctxR.drawImage(srcR, 0, 0);
        }
        ctxR.restore();
    }

    document.getElementById('pvInfo').textContent = `${pvW}×${pvH}  |  ${Math.round(pvZ * 100)}%`;
}

function rebuildFullBlend() {
    if (!rawPixels || !cleanPixels || !blendCtx) return;
    const md = maskCtx.getImageData(0, 0, pvW, pvH).data;
    const out = blendCtx.createImageData(pvW, pvH);
    for (let i = 0; i < rawPixels.length; i += 4) {
        const m = md[i] / 255;
        out.data[i] = rawPixels[i] * (1 - m) + cleanPixels[i] * m;
        out.data[i + 1] = rawPixels[i + 1] * (1 - m) + cleanPixels[i + 1] * m;
        out.data[i + 2] = rawPixels[i + 2] * (1 - m) + cleanPixels[i + 2] * m;
        out.data[i + 3] = 255;
    }
    blendCtx.putImageData(out, 0, 0);
}

function updateBlendRegion(cx, cy, radius) {
    if (!rawPixels || !cleanPixels || !blendCtx) return;
    const pad = Math.ceil(radius) + 2;
    const x0 = Math.max(0, Math.floor(cx - pad));
    const y0 = Math.max(0, Math.floor(cy - pad));
    const x1 = Math.min(pvW, Math.ceil(cx + pad));
    const y1 = Math.min(pvH, Math.ceil(cy + pad));
    const rw = x1 - x0, rh = y1 - y0;
    if (rw <= 0 || rh <= 0) return;
    const md = maskCtx.getImageData(x0, y0, rw, rh).data;
    const out = blendCtx.createImageData(rw, rh);
    for (let y = 0; y < rh; y++) {
        for (let x = 0; x < rw; x++) {
            const si = ((y0 + y) * pvW + (x0 + x)) * 4;
            const di = (y * rw + x) * 4;
            const m = md[di] / 255;
            out.data[di] = rawPixels[si] * (1 - m) + cleanPixels[si] * m;
            out.data[di + 1] = rawPixels[si + 1] * (1 - m) + cleanPixels[si + 1] * m;
            out.data[di + 2] = rawPixels[si + 2] * (1 - m) + cleanPixels[si + 2] * m;
            out.data[di + 3] = 255;
        }
    }
    blendCtx.putImageData(out, x0, y0);
}

function pvFit() {
    // Fit theo paneR
    const paneR = document.getElementById('pvPaneR');
    const aw = paneR.clientWidth, ah = paneR.clientHeight;
    if (!pvW || !pvH) return;
    pvZ = Math.min(aw / pvW, ah / pvH, 1);
    pvX = (aw - pvW * pvZ) / 2; pvY = (ah - pvH * pvZ) / 2;
}

async function pvClose() {
    if (maskDirty) await autoSaveMask();
    document.getElementById('pvOverlay').classList.remove('active');
    document.getElementById('brushBar').classList.remove('active');
    tool = 'move';
}

function pvNav(d) { const n = pvIdx + d; if (n >= 0 && n < CProj.pages.length) pvOpen(n); }

// Dual pane resizer
(function () {
    let isResizing = false;
    const divider = document.getElementById('pvDivider');
    const paneL = document.getElementById('pvPaneL');
    const dual = document.getElementById('pvDual');
    divider.addEventListener('mousedown', e => { isResizing = true; e.preventDefault(); });
    window.addEventListener('mousemove', e => {
        if (!isResizing) return;
        const rect = dual.getBoundingClientRect();
        let pct = ((e.clientX - rect.left) / rect.width) * 100;
        pct = Math.max(10, Math.min(90, pct));
        paneL.style.flex = `0 0 ${pct}%`;
        pvDraw(); // force update size
    });
    window.addEventListener('mouseup', () => isResizing = false);
})();

// Evt cho cả 2 pane
const panes = [document.getElementById('pvPaneL'), document.getElementById('pvPaneR')];
panes.forEach(pane => {
    // Zoom
    pane.addEventListener('wheel', e => {
        if (!document.getElementById('pvOverlay').classList.contains('active')) return;
        e.preventDefault();
        const rect = pane.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const old = pvZ; pvZ = Math.max(0.1, Math.min(20, pvZ * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
        pvX = mx - (mx - pvX) * (pvZ / old); pvY = my - (my - pvY) * (pvZ / old); pvDraw();
    }, { passive: false });

    // Pan + Paint
    pane.addEventListener('mousedown', e => {
        if (e.button === 2) { // Chuột phải để Pan
            dragging = true; dragSX = e.clientX - pvX; dragSY = e.clientY - pvY;
            return;
        }
        if (e.button !== 0) return; // Chỉ cho phép chuột trái thảo tác Brush/Eraser
        // Chặn vẽ nếu đang ở paneL (RAW)
        if (pane.id === 'pvPaneR' && (tool === 'brush' || tool === 'eraser')) {
            painting = true; paintAt(e, pane);
        }
    });
    pane.addEventListener('contextmenu', e => e.preventDefault()); // Chặn menu màn hình khi click phải
    pane.addEventListener('mousemove', e => {
        // Brush cursor chỉ ở paneR
        if (tool === 'brush' || tool === 'eraser') {
            const bc = document.getElementById('brushCursor');
            if (pane.id !== 'pvPaneR') { bc.style.display = 'none'; }
            else {
                const rect = pane.getBoundingClientRect();
                const sz = brushSize * pvZ;
                bc.style.display = 'block';
                bc.style.width = sz + 'px'; bc.style.height = sz + 'px';
                bc.style.left = (e.clientX - rect.left - sz / 2) + 'px';
                bc.style.top = (e.clientY - rect.top - sz / 2) + 'px';
                bc.style.borderColor = tool === 'brush' ? 'rgba(0,120,255,0.9)' : 'rgba(255,50,50,0.9)';
            }
        } else {
            document.getElementById('brushCursor').style.display = 'none';
        }

        if (painting && pane.id === 'pvPaneR') { paintAt(e, pane); return; }
        if (!dragging) return;
        pvX = e.clientX - dragSX; pvY = e.clientY - dragSY; pvDraw();
    });

    // Tooltip chỉ pane R
    if (pane.id === 'pvPaneR') {
        pane.addEventListener('mousemove', e => {
            const tip = document.getElementById('pvTip');
            if (pvLayer !== 'BBOX' || !pvBoxes.length || dragging) { tip.style.display = 'none'; return; }
            const rect = pane.getBoundingClientRect();
            const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
            const ix = (cx - pvX) / pvZ, iy = (cy - pvY) / pvZ;
            let found = null;
            for (const b of pvBoxes) { const [x1, y1, x2, y2] = b.bbox; if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) { found = b; break; } }
            if (found && found.ocr_text) { tip.style.display = 'block'; tip.style.left = (cx + 10) + 'px'; tip.style.top = (cy - 6) + 'px'; tip.textContent = `[${found['class']}] ${found.ocr_text}`; }
            else { tip.style.display = 'none'; }
        });
    }
});

window.addEventListener('mouseup', () => {
    if (painting) { painting = false; pushUndo(); }
    dragging = false;
});

// ── BRUSH ENGINE ──
function setTool(t) {
    tool = t;
    document.getElementById('btnBrush').classList.toggle('active', t === 'brush');
    document.getElementById('btnEraser').classList.toggle('active', t === 'eraser');
    document.getElementById('pvPaneR').style.cursor = (t === 'brush' || t === 'eraser') ? 'none' : 'grab';
    if (t === 'move') document.getElementById('brushCursor').style.display = 'none';
}

function paintAt(e, pane) {
    const rect = pane.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const ix = (cx - pvX) / pvZ, iy = (cy - pvY) / pvZ;
    const r = brushSize / 2;
    const hardness = (100 - brushSoft) / 100;

    if (hardness >= 0.95) {
        maskCtx.fillStyle = tool === 'brush' ? '#ffffff' : '#000000';
        maskCtx.beginPath(); maskCtx.arc(ix, iy, r, 0, Math.PI * 2); maskCtx.fill();
    } else {
        const grad = maskCtx.createRadialGradient(ix, iy, r * hardness, ix, iy, r);
        if (tool === 'brush') {
            grad.addColorStop(0, 'rgba(255,255,255,1)');
            grad.addColorStop(1, 'rgba(255,255,255,0)');
            maskCtx.globalCompositeOperation = 'lighter';
        } else {
            grad.addColorStop(0, 'rgba(0,0,0,1)');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            maskCtx.globalCompositeOperation = 'source-over';
        }
        maskCtx.fillStyle = grad;
        maskCtx.fillRect(ix - r, iy - r, r * 2, r * 2);
        maskCtx.globalCompositeOperation = 'source-over';
    }
    maskDirty = true;
    if (pvLayer === 'BLEND') updateBlendRegion(ix, iy, r);
    pvDraw();
}

function pushUndo() {
    undoStack.push(maskCtx.getImageData(0, 0, pvW, pvH));
    if (undoStack.length > 50) undoStack.shift();
    redoStack = [];
}
function maskUndo() {
    if (undoStack.length <= 1) return;
    redoStack.push(undoStack.pop());
    maskCtx.putImageData(undoStack[undoStack.length - 1], 0, 0);
    maskDirty = true;
    if (pvLayer === 'BLEND') rebuildFullBlend();
    pvDraw();
}
function maskRedo() {
    if (!redoStack.length) return;
    const d = redoStack.pop();
    undoStack.push(d);
    maskCtx.putImageData(d, 0, 0);
    maskDirty = true;
    if (pvLayer === 'BLEND') rebuildFullBlend();
    pvDraw();
}

// Auto save wrapper
async function autoSaveMask() {
    if (!maskDirty) return;
    document.getElementById('pvTitle').textContent = '💾 Saving...';
    await _doSaveMask();
    maskDirty = false;
}
function saveMaskBtn() { autoSaveMask(); toast('✅ Đã lưu mask'); }

async function _doSaveMask() {
    const pg = CProj.pages[pvIdx];
    const dataUrl = maskCanvas.toDataURL('image/png');
    await fetch(`/api/projects/${CFile}/pages/${pg.hidden_id}/save-mask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mask_base64: dataUrl })
    });
}

// Keyboard
document.addEventListener('keydown', e => {
    if (!document.getElementById('pvOverlay').classList.contains('active')) return;
    // Bỏ chặn ctrl/shift để người dùng có thể xài tắt khác, chỉ chặn default cho undo
    if (e.key === 'ArrowLeft') pvNav(-1);
    else if (e.key === 'ArrowRight') pvNav(1);
    else if (e.key === 'Escape') pvClose();
    else if (document.activeElement.tagName !== 'INPUT') { // Tránh gõ tắt khi đang focus input (vd. rename)
        if (e.key === 'b' || e.key === 'B') setTool('brush');
        else if (e.key === 'e' || e.key === 'E') setTool('eraser');
        else if (e.key === 'x' || e.key === 'X') setTool(tool === 'brush' ? 'eraser' : 'brush');
        else if (e.key === 'v' || e.key === 'V') setTool('move');
        else if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); maskUndo(); }
        else if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); maskRedo(); }
        else if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) { e.preventDefault(); saveMaskBtn(); }
    }
});

// ── ACTIONS ──
async function pvReset() {
    if (!confirm('Reset clean/mask/detections?')) return; const pg = CProj.pages[pvIdx];
    await fetch(`/api/projects/${CFile}/pages/${pg.hidden_id}/reset`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ layers: ['clean', 'mask', 'detections'] }) });
    pvClose(); openProj(CFile); toast('✅ Đã reset');
}
async function pvDelete() {
    if (!confirm('Xóa trang này?')) return; const pg = CProj.pages[pvIdx];
    await fetch(`/api/projects/${CFile}/pages/${pg.hidden_id}`, { method: 'DELETE' }); pvClose(); openProj(CFile); toast('✅ Đã xóa');
}

function exportAniga() {
    toast('Đang nén file .aniga...', 5000);
    location.href = '/api/projects/' + CFile + '/download';
}

async function doCreate() {
    const name = document.getElementById('cName').value.trim(); const files = document.getElementById('cFiles').files;
    if (!name) { toast('⚠️ Nhập tên!'); return; } if (!files.length) { toast('⚠️ Chọn ảnh!'); return; }
    document.getElementById('cStatus').textContent = `Đang tạo... (${files.length} ảnh)`;
    const fd = new FormData(); fd.append('project_name', name); for (const f of files) fd.append('files', f);
    await fetch('/api/projects/create', { method: 'POST', body: fd });
    hideModals(); loadList(); toast('✅ Đã tạo');
}

async function doAdd() {
    const files = document.getElementById('addFiles').files; if (!files.length) return;
    const fd = new FormData(); for (const f of files) fd.append('files', f);
    await fetch(`/api/projects/${CFile}/add-pages`, { method: 'POST', body: fd });
    hideModals(); openProj(CFile); toast(`✅ Thêm ${files.length} trang`);
}

async function doRename() {
    const name = document.getElementById('rnName').value.trim(); if (!name) return;
    await fetch(`/api/projects/${CFile}/rename`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
    hideModals(); openProj(CFile); toast('✅ Đã đổi tên');
}

async function doUpdate() {
    const file = document.getElementById('updFile').files[0]; if (!file) { toast('⚠️ Chọn file!'); return; }
    document.getElementById('updStatus').textContent = 'Đang merge...';
    const fd = new FormData(); fd.append('update_file', file);
    const r = await fetch(`/api/projects/${CFile}/update`, { method: 'POST', body: fd });
    const data = await r.json(); hideModals();
    openProj(CFile);
    if (data.errors && data.errors.length) toast(`⚠️ ${data.errors.join(', ')}`, 4000);
    else toast(`✅ Sync ${data.synced} trang`);
}

function doExport() {
    const layerChecks = document.querySelectorAll('#exportModal .export-cb input[type="checkbox"]:not(#cbCropResize):checked');
    const layers = Array.from(layerChecks).map(c => c.value).join(',');
    if (!layers) { toast('⚠️ Chọn ít nhất 1 layer!'); return; }

    // Tùy chọn Crop & Resize
    const cbCropResize = document.getElementById('cbCropResize');
    const bCrop = cbCropResize ? cbCropResize.checked : true;

    location.href = `/api/projects/${CFile}/resolve?layers=${layers}&crop_resize=${bCrop}`;
    hideModals(); toast('📤 Đang xuất ZIP...');
}

async function setFlux(v) {
    await fetch(`/api/projects/${CFile}/imgcraft-config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ flux_size: parseInt(v) }) });
    toast(`✅ Flux Size → ${v}`);
}

// ── INIT ──
loadList();
