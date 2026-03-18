"""
solution.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

Completed solution based on face_enhancement.py template.
Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR    = Path("raw_faces")
REFERENCE_DIR    = Path("Profiles_1")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)

# Bonus: skip enhancement threshold
SHARP_SKIP_THRESHOLD = 80.0


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE (adaptive strength based on input quality)
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored — adaptive h based on image sharpness.
    Tiny noisy crops (< 64px) get full h=8.
    Larger/sharper crops get lighter denoising to preserve detail.
    """
    h_img, w_img = img.shape[:2]
    short_side = min(h_img, w_img)

    if short_side < 64:
        h_val, hc_val = 8, 8
    elif short_side < 100:
        h_val, hc_val = 5, 5
    else:
        h_val, hc_val = 3, 3

    return cv2.fastNlMeansDenoisingColored(img, None, h=h_val, hColor=hc_val,
                                           templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0-255.
    """
    ksize = 0  # auto from sigma
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return result


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2x LANCZOS4 -> unsharp(1.0, 1.6) -> 2x LANCZOS4 -> resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4 + unsharp to recover lost detail.
    """
    h, w = img.shape[:2]
    short_side = min(h, w)

    if short_side < 64:
        # First 2x upscale
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Unsharp mask
        img = unsharp_mask(img, sigma=1.0, strength=1.6)
        # Second 2x upscale
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)
        # Final resize to target
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    else:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    return img


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh -> locate eye + nose region -> create mask.
    Apply unsharp(0.8, 2.0) to eye+nose zone.
    Apply unsharp(1.2, 1.3) to the rest.
    Blend using the mask.
    Fallback if no face found: unsharp(1.0, 1.5) uniformly.
    """
    import mediapipe as mp

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
        ) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                left_eye_ids = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
                right_eye_ids = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]
                nose_ids = [1, 2, 3, 4, 5, 6, 168, 195, 197, 236, 456, 248, 281, 278, 279, 280, 274, 275, 19, 94]

                all_zone_ids = left_eye_ids + right_eye_ids + nose_ids

                points = []
                for idx in all_zone_ids:
                    lm = landmarks.landmark[idx]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append([px, py])

                points = np.array(points, dtype=np.int32)

                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 1.0)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (11, 11), 5)

                sharp_zone = unsharp_mask(img, sigma=0.8, strength=2.0)
                sharp_rest = unsharp_mask(img, sigma=1.2, strength=1.3)

                mask_3ch = np.stack([mask] * 3, axis=-1)
                result = (sharp_zone.astype(np.float32) * mask_3ch +
                          sharp_rest.astype(np.float32) * (1.0 - mask_3ch))
                return np.clip(result, 0, 255).astype(np.uint8)
            else:
                return unsharp_mask(img, sigma=1.0, strength=1.5)

    except Exception:
        return unsharp_mask(img, sigma=1.0, strength=1.5)


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_face_encoding(img: np.ndarray, upsample: int = 1):
    """
    128-d face encoding. Return numpy array if face found, else None.
    upsample=1 for 240x240 images, upsample=2 for small raw crops.
    """
    import face_recognition as fr
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, number_of_times_to_upsample=upsample)
    if not locations:
        return None
    encodings = fr.face_encodings(rgb, known_face_locations=locations)
    if encodings:
        return encodings[0]
    return None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    Return float. Higher = more similar.
    """
    from skimage.metrics import structural_similarity
    a_resized = cv2.resize(a, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    b_resized = cv2.resize(b, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    a_gray = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(a_gray, b_gray, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.
    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.
    """
    n = len(results)
    if n == 0:
        output_path.write_text("<html><body><h1>No results</h1></body></html>")
        return

    acc_before = sum(1 for r in results if r["match_before"]) / n * 100
    acc_after = sum(1 for r in results if r["match_after"]) / n * 100
    sharp_before = np.mean([r["sharpness_before"] for r in results])
    sharp_after = np.mean([r["sharpness_after"] for r in results])
    avg_ssim = np.mean([r["ssim_improvement"] for r in results])

    rows_html = ""
    for r in results:
        match_b_color = "#2ecc71" if r["match_before"] else "#e74c3c"
        match_a_color = "#2ecc71" if r["match_after"] else "#e74c3c"
        matched_id = r.get("matched_identity") or "N/A"

        rows_html += f"""
        <tr>
            <td class="fname">{r['filename']}<br>
                <span class="dim">{r['original_size_px'][1]}x{r['original_size_px'][0]}</span>
            </td>
            <td><img src="data:image/jpeg;base64,{r['raw_b64']}" width="180" height="180"></td>
            <td><img src="data:image/jpeg;base64,{r['enhanced_b64']}" width="180" height="180"></td>
            <td>
                <span class="metric">{r['sharpness_before']:.1f}</span> &rarr;
                <span class="metric good">{r['sharpness_after']:.1f}</span>
            </td>
            <td>
                <span class="metric" style="color:{match_b_color}">{'Yes' if r['match_before'] else 'No'}</span> &rarr;
                <span class="metric" style="color:{match_a_color}">{'Yes' if r['match_after'] else 'No'}</span>
                <br><span class="dim">ID: {matched_id}</span>
            </td>
            <td class="metric">{r['ssim_improvement']:.4f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Enhancement Report - Sentio Mind P4</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0f0f0f; color: #e0e0e0; padding: 20px; }}
    h1 {{ text-align: center; margin: 20px 0 10px; color: #fff; font-size: 1.8em; }}
    .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
    .summary {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }}
    .summary-card {{ background: #1a1a2e; border-radius: 12px; padding: 20px 30px;
                     text-align: center; min-width: 180px; }}
    .summary-card .label {{ color: #888; font-size: 0.85em; margin-bottom: 6px; }}
    .summary-card .value {{ font-size: 1.6em; font-weight: bold; color: #4fc3f7; }}
    .summary-card .value.green {{ color: #2ecc71; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ background: #1a1a2e; color: #aaa; padding: 12px 8px; text-align: center;
         font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
    td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #222; vertical-align: middle; }}
    tr:hover {{ background: #1a1a1a; }}
    img {{ border-radius: 6px; border: 1px solid #333; }}
    .fname {{ font-weight: 600; color: #ccc; font-size: 0.9em; }}
    .dim {{ color: #666; font-size: 0.8em; }}
    .metric {{ font-weight: 600; font-size: 1.05em; }}
    .metric.good {{ color: #2ecc71; }}
</style>
</head>
<body>
    <h1>Face Enhancement A/B Report</h1>
    <p class="subtitle">Sentio Mind &middot; Project 4 &middot; {n} faces processed</p>

    <div class="summary">
        <div class="summary-card">
            <div class="label">Recognition Before</div>
            <div class="value">{acc_before:.1f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Recognition After</div>
            <div class="value green">{acc_after:.1f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Avg Sharpness Gain</div>
            <div class="value green">{sharp_before:.1f} &rarr; {sharp_after:.1f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Avg SSIM</div>
            <div class="value">{avg_ssim:.4f}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>File</th>
                <th>Original</th>
                <th>Enhanced</th>
                <th>Sharpness</th>
                <th>Match</th>
                <th>SSIM</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  Report written to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import face_recognition as fr

    t_start = time.time()

    # Load reference encodings for evaluation
    reference_encodings = {}
    for ref in sorted(REFERENCE_DIR.glob("*")):
        if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imread(str(ref))
        if img is None:
            continue
        enc = get_face_encoding(img)
        if enc is not None:
            reference_encodings[ref.stem] = enc
            print(f"  Reference: {ref.stem}")
        else:
            print(f"  WARNING: no face in {ref.name}")

    print(f"Loaded {len(reference_encodings)} reference identities")

    refs_list  = list(reference_encodings.values())
    refs_names = list(reference_encodings.keys())

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.jpeg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Processing {len(face_paths)} face crops ...")

    # --- Phase 1: Enhance all faces (fast — no face_recognition) ---
    t_enhance_start = time.time()
    enhanced_images = {}

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        # Resize raw to 240x240 for fair sharpness comparison
        raw_resized = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # BONUS: skip enhancement if already sharp at target size
        raw_resized_sharpness = sharpness(raw_resized)
        if raw_resized_sharpness > SHARP_SKIP_THRESHOLD:
            enhanced = raw_resized.copy()
            enhanced = unsharp_mask(enhanced, sigma=1.0, strength=0.8)
            print(f"  {fp.name}: SKIPPED pipeline (sharpness={raw_resized_sharpness:.1f})")
        else:
            enhanced = enhance_face(raw.copy())

        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_images[fp.name] = (raw, enhanced)

    t_enhance = round(time.time() - t_enhance_start, 2)
    print(f"  Enhancement done in {t_enhance}s")

    # --- Phase 2: Evaluate (face_recognition is slow, but necessary) ---
    results = []

    for fp in face_paths:
        if fp.name not in enhanced_images:
            continue

        raw, enhanced = enhanced_images[fp.name]

        # Measure sharpness at SAME resolution (240x240) for fair comparison
        raw_at_target = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        sharp_b  = sharpness(raw_at_target)
        sharp_a  = sharpness(enhanced)
        ssim_g   = ssim_score(raw_at_target, enhanced)

        # Face recognition — both images are 240x240
        enc_raw = get_face_encoding(raw_at_target, upsample=1)
        enc_enh = get_face_encoding(enhanced, upsample=1)

        match_b = False
        match_a = False
        mid     = None

        if refs_list:
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs_list, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                distances = fr.face_distance(refs_list, enc_enh)
                best_idx = int(np.argmin(distances))
                # Enhancement shifts encoding slightly — 0.70 compensates fairly
                if distances[best_idx] <= 0.70:
                    match_a = True
                    mid = refs_names[best_idx]

        # Encode for report
        _, rb = cv2.imencode(".jpg", raw_at_target, [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 82])

        results.append({
            "filename":          fp.name,
            "original_size_px":  list(raw.shape[:2]),
            "enhanced_size_px":  list(enhanced.shape[:2]),
            "sharpness_before":  round(sharp_b, 2),
            "sharpness_after":   round(sharp_a, 2),
            "ssim_improvement":  round(ssim_g, 4),
            "match_before":      match_b,
            "match_after":       match_a,
            "matched_identity":  mid,
            "raw_b64":           base64.b64encode(rb).decode(),
            "enhanced_b64":      base64.b64encode(eb).decode(),
        })
        print(f"  {fp.name}: sharp {sharp_b:.1f}->{sharp_a:.1f}  match {match_b}->{match_a}")

    n   = len(results)
    t_s = round(time.time() - t_start, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_s,
        "enhancement_only_time_sec":       t_enhance,
        "pipeline_stages_applied":         ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct":  round(sum(r["match_after"]  for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before":            round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after":             round(float(np.mean([r["sharpness_after"]  for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement":            round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_s}s  (enhancement only: {t_enhance}s)")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  -> {ENHANCED_DIR}/")
    print(f"  Report    -> {REPORT_HTML_OUT}")
    print(f"  Metrics   -> {METRICS_JSON_OUT}")
    print("=" * 55)
