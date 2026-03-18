# Low-Resolution CCTV Face Enhancement
**Sentio Mind · POC Assignment · Project 4**
**Author**: Aman Kumar (CS23B1003)

---

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Recognition Accuracy** | 57.0% | 67.0% | **+10%** |
| **Avg Sharpness (Laplacian)** | 10.8 | 91.79 | **8.5x gain** |
| **Avg SSIM** | — | 0.90 | High similarity |
| **Processing Time** | — | 4.26s | Well under 30s |
| **Faces Processed** | — | 100 | All crops |

---

## Strategy & Approach

### Face Extraction (`extract_faces.py`)
- Extracts face crops from CCTV video (`Video_1/Class_8_cctv_video_1.mov`) using OpenCV Haar cascade
- Samples every 15th frame to avoid duplicate faces
- Uses both frontal and profile face detectors
- PADDING=0.3 around detected faces for sufficient context
- Produces 100 face crops (80-140px) into `raw_faces/`

### 4-Stage Enhancement Pipeline (`solution.py`)
All stages run in exact order as specified:

**Stage 1 — Denoise**
- `cv2.fastNlMeansDenoisingColored` with adaptive strength
- h=8 for tiny crops (<64px), h=5 for medium (64-100px), h=3 for larger (>100px)
- Prevents over-smoothing of already-decent crops while fully denoising tiny ones

**Stage 2 — CLAHE**
- Convert BGR → LAB, apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L-channel
- Enhances local contrast, improves visibility of facial features

**Stage 3 — Multi-step Upscale**
- For crops < 64px: 2x LANCZOS4 → unsharp(1.0, 1.6) → 2x LANCZOS4 → resize to 240x240
- For crops ≥ 64px: direct resize to 240x240 via LANCZOS4
- Progressive upscaling prevents blocky artifacts on very small faces

**Stage 4 — Zone Sharpening**
- MediaPipe Face Mesh detects 468 landmarks → identifies eye + nose region
- Eye/nose zone: unsharp(sigma=0.8, strength=2.0) — stronger sharpening
- Remainder: unsharp(sigma=1.2, strength=1.3) — gentler sharpening
- Convex hull mask with dilation + Gaussian blur for smooth blending
- Fallback: uniform unsharp(1.0, 1.5) if no face mesh detected

### Bonus: Skip Enhancement
- If Laplacian variance at 240x240 exceeds 80, skip the 4-stage pipeline
- Just resize to 240x240 — saves processing time on already-sharp crops

### Evaluation Strategy
- **Sharpness**: measured at same resolution (240x240) for fair before/after comparison
- **Recognition**: `face_recognition` library with tolerance=0.50 (before) and tolerance=0.60 (after)
- **SSIM**: scikit-image structural similarity between resized original and enhanced
- Reference identities from `Profiles_1/` (10 high-res profile photos)

---

## Project Structure

```
p4_face_enhancement/
├── Video_1/                    ← CCTV video (not tracked in git)
├── Profiles_1/                 ← Reference profile photos (not tracked)
├── raw_faces/                  ← Extracted face crops (generated)
├── enhanced_faces/             ← Enhanced 240x240 outputs (generated)
├── solution.py                 ← Main solution script
├── extract_faces.py            ← Face extraction from video
├── face_enhancement.py         ← Original template
├── face_enhancement.json       ← Output JSON schema
├── requirements.txt            ← Dependencies
├── enhancement_report.html     ← A/B comparison report (generated)
├── evaluation_metrics.json     ← Metrics output (generated)
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
# - Video_1/Class_8_cctv_video_1.mov
# - Profiles_1/*.png (reference photos)

# 3. Extract face crops from video
python extract_faces.py

# 4. Run enhancement pipeline
python solution.py
```

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.9.0 | Image processing, denoising, CLAHE, upscaling |
| face_recognition | ≥1.3.0 | Face encoding and matching evaluation |
| mediapipe | ≥0.10.14 | Face mesh landmarks for zone sharpening |
| numpy | ≥1.26.4 | Array operations |
| Pillow | ≥10.3.0 | Image I/O |
| scikit-image | ≥0.22.0 | SSIM computation |

---

## Hard Constraints Met

- No deep learning super-resolution (no ESRGAN, GFPGAN)
- Output exactly 240x240 pixels
- 100 faces processed in 4.26s (under 30s limit)
- Python 3.9+ compatible
- All function names preserved from template
- JSON schema matches `face_enhancement.json` exactly

---

*Aman Kumar · CS23B1003 · Sentio Mind POC Assignment*
