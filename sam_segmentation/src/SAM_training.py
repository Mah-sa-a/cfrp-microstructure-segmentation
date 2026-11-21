# -*- coding: utf-8 -*-
"""
SAM-assisted semantic labeling for CFRP real micrographs.

Classes: 0 = matrix, 1 = fiber, 2 = void.
Save full-image labels, QC visualizations, optional 512x512 image/mask
patches, patch overlays, and a manifest for later dataset preparation.

Patch overlays are for visual QC only; never use them as training masks.
The manifest is recreated on every run: keep reviewed decisions in a
separate reviewed CSV. Deleting an overlay does not reject a training pair.
"""

import os
import csv
import math
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.color import rgb2gray
from skimage.filters import threshold_multiotsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_opening,
    disk,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_FOLDER = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/data/real_images"
)

# Separate output folder to preserve earlier results during the dilation test.
OUTPUT_ROOT = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/"
    "outputs/sam_no_dilation_with_patch_overlays"
)

SAM_CHECKPOINT = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/"
    "checkpoints/sam/sam_vit_h_4b8939.pth"
)

SAM_MODEL_TYPE = "vit_h"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Internal segmentation patches
SEGMENT_PATCH_SIZE = 512
SEGMENT_OVERLAP = 64
SEGMENT_PATCH_THRESHOLD_PX = 600 * 600

# Training-patch export
EXPORT_PATCHES = True
TRAIN_PATCH_SIZE = 512
TRAIN_PATCH_OVERLAP = 0
KEEP_ONLY_FULL_PATCHES = True

# Patch overlays: original weight + mask weight must equal 1.
PATCH_OVERLAY_IMAGE_WEIGHT = 0.65
PATCH_OVERLAY_MASK_WEIGHT = 0.35

COLORS = {
    0: [128, 128, 128],
    1: [255, 0, 0],
    2: [0, 255, 0],
}
CLASS_NAMES = {0: "matrix", 1: "fiber", 2: "void"}

# SAM automatic-mask generation
SAM_POINTS_PER_SIDE = 64
SAM_POINTS_PER_BATCH = 64
SAM_PRED_IOU_THRESH = 0.70
SAM_STABILITY_THRESH = 0.80
SAM_BOX_NMS_THRESH = 0.50
SAM_CROP_N_LAYERS = 1
SAM_CROP_NMS_THRESH = 0.50
SAM_CROP_OVERLAP_RATIO = 0.35
SAM_MIN_MASK_REGION_AREA = 5

# Fiber candidate filtering
FIBER_AREA_MIN = 8
FIBER_AREA_MAX = 2500
FIBER_CIRCULARITY_MIN = 0.45
FIBER_SOLIDITY_MIN = 0.60
FIBER_MASK_IOU_MAX = 0.20

# Radius 0 disables expansion for the false-bridge diagnostic test.
FIBER_INTERFACE_DILATION_RADIUS = 0
FIBER_INTERFACE_INTENSITY_OFFSET = 35

# Void filtering
VOID_AREA_FRACTION_OF_MEDIAN_FIBER = 1.0
VOID_FALLBACK_MIN_AREA = 40
VOID_MIN_SOLIDITY = 0.35
VOID_MIN_CIRCULARITY = 0.01

# Final cleanup: unchanged from the supplied script.
POST_FIBER_OPENING_RADIUS = 1
POST_VOID_OPENING_RADIUS = 1
POST_REMOVE_SMALL_FIBERS = 8
POST_REMOVE_SMALL_VOIDS = 20
POST_FILL_HOLES_AREA = 64

# =============================================================================
# OUTPUT FOLDERS
# =============================================================================

FULL_ORIGINAL_DIR = os.path.join(OUTPUT_ROOT, "full_images")
FULL_RAW_MASK_DIR = os.path.join(OUTPUT_ROOT, "full_masks_raw")
FULL_COLOR_MASK_DIR = os.path.join(OUTPUT_ROOT, "full_masks_color")
FULL_OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "overlays")
FULL_PANEL_DIR = os.path.join(OUTPUT_ROOT, "panels")
FULL_INSTANCE_RAW_DIR = os.path.join(OUTPUT_ROOT, "fiber_instances_raw")
FULL_INSTANCE_VIS_DIR = os.path.join(OUTPUT_ROOT, "fiber_instances_vis")

PATCH_IMAGE_DIR = os.path.join(OUTPUT_ROOT, "patches", "images")
PATCH_MASK_DIR = os.path.join(OUTPUT_ROOT, "patches", "masks")
PATCH_OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "patches", "overlays")
MANIFEST_PATH = os.path.join(OUTPUT_ROOT, "sam_manifest.csv")

for folder in [
    FULL_ORIGINAL_DIR, FULL_RAW_MASK_DIR, FULL_COLOR_MASK_DIR,
    FULL_OVERLAY_DIR, FULL_PANEL_DIR, FULL_INSTANCE_RAW_DIR,
    FULL_INSTANCE_VIS_DIR,
]:
    os.makedirs(folder, exist_ok=True)

if EXPORT_PATCHES:
    for folder in [PATCH_IMAGE_DIR, PATCH_MASK_DIR, PATCH_OVERLAY_DIR]:
        os.makedirs(folder, exist_ok=True)

# =============================================================================
# LOAD SAM
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if not os.path.isfile(SAM_CHECKPOINT):
    raise FileNotFoundError(f"SAM checkpoint not found:\n{SAM_CHECKPOINT}")

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=SAM_POINTS_PER_SIDE,
    points_per_batch=SAM_POINTS_PER_BATCH,
    pred_iou_thresh=SAM_PRED_IOU_THRESH,
    stability_score_thresh=SAM_STABILITY_THRESH,
    box_nms_thresh=SAM_BOX_NMS_THRESH,
    crop_n_layers=SAM_CROP_N_LAYERS,
    crop_nms_thresh=SAM_CROP_NMS_THRESH,
    crop_overlap_ratio=SAM_CROP_OVERLAP_RATIO,
    min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
)

# =============================================================================
# IMAGE HELPERS
# =============================================================================

def rgb_to_gray_uint8(img_rgb):
    """Convert RGB image to uint8 grayscale."""
    return (rgb2gray(img_rgb) * 255).astype(np.uint8)


def mask_iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union if union > 0 else 0.0


def get_global_thresholds(gray):
    smoothed = gaussian(gray.astype(np.float32), sigma=1.0)
    try:
        thresholds = threshold_multiotsu(smoothed, classes=3)
        void_threshold = float(thresholds[0])
        fiber_threshold = float(thresholds[1])
    except ValueError:
        void_threshold = float(np.percentile(gray, 20))
        fiber_threshold = float(np.percentile(gray, 70))
    return void_threshold, fiber_threshold

# =============================================================================
# PATCH HELPERS
# =============================================================================

def get_patch_positions(length, patch_size, overlap):
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("patch_size must be larger than overlap")
    last = max(length - patch_size, 0)
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions


def extract_patches(img, patch_size=512, overlap=64):
    h, w = img.shape[:2]
    xs = get_patch_positions(w, patch_size, overlap)
    ys = get_patch_positions(h, patch_size, overlap)
    patches = []
    for y in ys:
        for x in xs:
            patch = img[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            patches.append((x, y, patch))
    return patches, (h, w)


def merge_patches_vote(patch_results, full_size, n_classes=3):
    h, w = full_size
    votes = np.zeros((n_classes, h, w), dtype=np.uint16)
    for x, y, patch_mask in patch_results:
        patch_h, patch_w = patch_mask.shape
        for class_id in range(n_classes):
            votes[class_id, y:y + patch_h, x:x + patch_w] += (
                patch_mask == class_id
            )
    return np.argmax(votes, axis=0).astype(np.uint8)

# =============================================================================
# FIBER SELECTION
# =============================================================================

def select_sam_fiber_masks(sam_masks, gray, fiber_threshold):
    candidates = []
    for sam_mask in sam_masks:
        proposal = sam_mask["segmentation"].astype(bool)
        components = label(proposal, connectivity=2)
        for prop in regionprops(components, intensity_image=gray):
            if prop.area < FIBER_AREA_MIN or prop.area > FIBER_AREA_MAX:
                continue
            perimeter = max(prop.perimeter, 1.0)
            circularity = (4.0 * np.pi * prop.area) / (perimeter ** 2)
            if circularity < FIBER_CIRCULARITY_MIN:
                continue
            if prop.solidity < FIBER_SOLIDITY_MIN:
                continue
            if prop.mean_intensity < fiber_threshold:
                continue
            component = components == prop.label
            score = (
                0.50 * float(sam_mask["predicted_iou"])
                + 0.30 * float(sam_mask["stability_score"])
                + 0.20 * min(float(circularity), 1.0)
            )
            candidates.append({
                "mask": component, "score": score, "area": prop.area,
            })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    accepted = []
    for candidate in candidates:
        is_duplicate = any(
            mask_iou(candidate["mask"], selected["mask"]) >= FIBER_MASK_IOU_MAX
            for selected in accepted
        )
        if not is_duplicate:
            accepted.append(candidate)
    return accepted


def build_fiber_instances(selected_fibers, gray, fiber_threshold):
    h, w = gray.shape
    fiber_instances = np.zeros((h, w), dtype=np.int32)
    occupied = np.zeros((h, w), dtype=bool)
    interface_threshold = max(
        0, int(fiber_threshold - FIBER_INTERFACE_INTENSITY_OFFSET)
    )
    footprint = disk(FIBER_INTERFACE_DILATION_RADIUS)
    next_id = 1
    for item in selected_fibers:
        candidate = item["mask"].copy()
        if FIBER_INTERFACE_DILATION_RADIUS > 0:
            candidate = binary_dilation(candidate, footprint=footprint)
        candidate &= gray >= interface_threshold
        candidate &= ~occupied
        if np.count_nonzero(candidate) < FIBER_AREA_MIN:
            continue
        fiber_instances[candidate] = next_id
        occupied[candidate] = True
        next_id += 1
    return fiber_instances, interface_threshold


def relabel_instances(instance_mask):
    """Renumber existing positive labels consecutively."""
    relabeled = np.zeros_like(instance_mask, dtype=np.int32)
    next_id = 1
    for prop in regionprops(instance_mask):
        relabeled[instance_mask == prop.label] = next_id
        next_id += 1
    return relabeled

# =============================================================================
# VOID DETECTION
# =============================================================================

def get_min_void_area(fiber_instances):
    fiber_areas = np.array([
        prop.area for prop in regionprops(fiber_instances)
        if prop.area >= FIBER_AREA_MIN
    ])
    if len(fiber_areas) == 0:
        return int(VOID_FALLBACK_MIN_AREA), 0.0
    median_fiber_area = float(np.median(fiber_areas))
    min_void_area = max(
        VOID_FALLBACK_MIN_AREA,
        int(VOID_AREA_FRACTION_OF_MEDIAN_FIBER * median_fiber_area),
    )
    return min_void_area, median_fiber_area


def detect_and_filter_voids(gray, void_threshold, min_void_area):
    candidates = gray < void_threshold
    void_labels = label(candidates, connectivity=2)
    void_mask = np.zeros(gray.shape, dtype=bool)
    for prop in regionprops(void_labels):
        if prop.area < min_void_area:
            continue
        perimeter = max(prop.perimeter, 1.0)
        circularity = (4.0 * np.pi * prop.area) / (perimeter ** 2)
        if prop.solidity < VOID_MIN_SOLIDITY:
            continue
        if circularity < VOID_MIN_CIRCULARITY:
            continue
        void_mask[void_labels == prop.label] = True
    return void_mask


def filter_existing_voids_by_fiber_area(semantic_mask, fiber_instances):
    min_void_area, median_fiber_area = get_min_void_area(fiber_instances)
    labels_void = label(semantic_mask == 2, connectivity=2)
    kept_voids = np.zeros(semantic_mask.shape, dtype=bool)
    for prop in regionprops(labels_void):
        if prop.area >= min_void_area:
            kept_voids[labels_void == prop.label] = True
    output = semantic_mask.copy()
    output[output == 2] = 0
    output[kept_voids] = 2
    return output, min_void_area, median_fiber_area

# =============================================================================
# POSTPROCESSING
# =============================================================================

def postprocess_semantic_mask(semantic_mask):
    fiber = semantic_mask == 1
    void = semantic_mask == 2
    if POST_FIBER_OPENING_RADIUS > 0:
        fiber = binary_opening(fiber, footprint=disk(POST_FIBER_OPENING_RADIUS))
    if POST_VOID_OPENING_RADIUS > 0:
        void = binary_opening(void, footprint=disk(POST_VOID_OPENING_RADIUS))
    fiber = remove_small_objects(fiber, min_size=POST_REMOVE_SMALL_FIBERS)
    void = remove_small_objects(void, min_size=POST_REMOVE_SMALL_VOIDS)
    fiber = remove_small_holes(fiber, area_threshold=POST_FILL_HOLES_AREA)
    void = remove_small_holes(void, area_threshold=POST_FILL_HOLES_AREA)
    cleaned = np.zeros_like(semantic_mask, dtype=np.uint8)
    cleaned[fiber] = 1
    cleaned[void] = 2
    cleaned[fiber] = 1
    return cleaned

# =============================================================================
# TILE SEGMENTATION
# =============================================================================

def segment_tile(img_rgb, gray, void_threshold, fiber_threshold):
    sam_masks = mask_generator.generate(img_rgb)
    selected_fibers = select_sam_fiber_masks(
        sam_masks=sam_masks, gray=gray, fiber_threshold=fiber_threshold,
    )
    fiber_instances, interface_threshold = build_fiber_instances(
        selected_fibers=selected_fibers,
        gray=gray,
        fiber_threshold=fiber_threshold,
    )
    fiber_instances = relabel_instances(fiber_instances)
    min_void_area, median_fiber_area = get_min_void_area(fiber_instances)
    void_mask = detect_and_filter_voids(
        gray=gray, void_threshold=void_threshold, min_void_area=min_void_area,
    )
    semantic_mask = np.zeros(gray.shape, dtype=np.uint8)
    semantic_mask[fiber_instances > 0] = 1
    semantic_mask[void_mask] = 2
    semantic_mask = postprocess_semantic_mask(semantic_mask)
    fiber_instances = label(semantic_mask == 1, connectivity=2)
    fiber_instances = relabel_instances(fiber_instances)
    info = {
        "sam_proposals": len(sam_masks),
        "fiber_count": int(fiber_instances.max()),
        "interface_threshold": int(interface_threshold),
        "median_fiber_area": float(median_fiber_area),
        "min_void_area": int(min_void_area),
    }
    print(
        f" SAM proposals={len(sam_masks)}, "
        f"fibers={fiber_instances.max()}, "
        f"interface threshold={interface_threshold}, "
        f"median fiber area={median_fiber_area:.1f}px, "
        f"minimum void area={min_void_area}px"
    )
    return semantic_mask, fiber_instances, info

# =============================================================================
# FULL IMAGE / PATCH DISPATCHER
# =============================================================================

def segment_image(img_rgb):
    gray = rgb_to_gray_uint8(img_rgb)
    h, w = gray.shape
    void_threshold, fiber_threshold = get_global_thresholds(gray)
    print(
        f" Thresholds: void < {void_threshold:.1f}, "
        f"fiber core >= {fiber_threshold:.1f}, "
        f"dilation radius={FIBER_INTERFACE_DILATION_RADIUS}"
    )
    if h * w <= SEGMENT_PATCH_THRESHOLD_PX:
        print(f" Image {w}x{h}: full-image mode")
        semantic_mask, fiber_instances, tile_info = segment_tile(
            img_rgb=img_rgb, gray=gray,
            void_threshold=void_threshold, fiber_threshold=fiber_threshold,
        )
        info = {
            "mode": "full",
            "void_threshold": float(void_threshold),
            "fiber_threshold": float(fiber_threshold),
            **tile_info,
        }
        return semantic_mask, fiber_instances, info

    patches, full_size = extract_patches(
        img_rgb, patch_size=SEGMENT_PATCH_SIZE, overlap=SEGMENT_OVERLAP,
    )
    print(
        f" Image {w}x{h}: patch mode ({len(patches)} patches, "
        f"size={SEGMENT_PATCH_SIZE}, overlap={SEGMENT_OVERLAP})"
    )
    patch_results = []
    patch_sam_props = []
    for patch_index, (x, y, patch_rgb) in enumerate(patches, start=1):
        patch_gray = rgb_to_gray_uint8(patch_rgb)
        print(f" Patch {patch_index}/{len(patches)} at ({x}, {y})")
        patch_semantic, _, patch_info = segment_tile(
            img_rgb=patch_rgb, gray=patch_gray,
            void_threshold=void_threshold, fiber_threshold=fiber_threshold,
        )
        patch_results.append((x, y, patch_semantic))
        patch_sam_props.append(patch_info["sam_proposals"])

    semantic_mask = merge_patches_vote(
        patch_results=patch_results, full_size=full_size, n_classes=3,
    )
    semantic_mask = postprocess_semantic_mask(semantic_mask)
    fiber_instances = label(semantic_mask == 1, connectivity=2)
    fiber_instances = relabel_instances(fiber_instances)
    semantic_mask, min_void_area, median_fiber_area = (
        filter_existing_voids_by_fiber_area(
            semantic_mask=semantic_mask, fiber_instances=fiber_instances,
        )
    )
    semantic_mask[fiber_instances > 0] = 1
    semantic_mask = postprocess_semantic_mask(semantic_mask)
    fiber_instances = label(semantic_mask == 1, connectivity=2)
    fiber_instances = relabel_instances(fiber_instances)
    print(
        f" Final fibers={fiber_instances.max()}, "
        f"median fiber area={median_fiber_area:.1f}px, "
        f"minimum void area={min_void_area}px"
    )
    info = {
        "mode": "patch",
        "void_threshold": float(void_threshold),
        "fiber_threshold": float(fiber_threshold),
        "sam_proposals": int(sum(patch_sam_props)),
        "fiber_count": int(fiber_instances.max()),
        "median_fiber_area": float(median_fiber_area),
        "min_void_area": int(min_void_area),
        "n_internal_patches": len(patches),
    }
    return semantic_mask, fiber_instances, info

# =============================================================================
# VISUALIZATION
# =============================================================================

def semantic_to_rgb(semantic_mask):
    color_mask = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        color_mask[semantic_mask == class_id] = color
    return color_mask


def make_instance_visualization(fiber_instances):
    n_instances = int(fiber_instances.max())
    if n_instances == 0:
        return np.zeros((*fiber_instances.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    color_lut = rng.integers(
        low=35, high=256, size=(n_instances + 1, 3), dtype=np.uint8,
    )
    color_lut[0] = [0, 0, 0]
    return color_lut[fiber_instances]

# =============================================================================
# SAVING FULL RESULTS
# =============================================================================

def save_full_outputs(img_rgb, semantic_mask, fiber_instances, stem):
    color_mask = semantic_to_rgb(semantic_mask)
    overlay = cv2.addWeighted(img_rgb, 0.60, color_mask, 0.40, 0)
    instance_vis = make_instance_visualization(fiber_instances)
    original_path = os.path.join(FULL_ORIGINAL_DIR, f"{stem}_original.png")
    raw_mask_path = os.path.join(FULL_RAW_MASK_DIR, f"{stem}_mask.png")
    color_mask_path = os.path.join(FULL_COLOR_MASK_DIR, f"{stem}_mask_color.png")
    overlay_path = os.path.join(FULL_OVERLAY_DIR, f"{stem}_overlay.png")
    instance_raw_path = os.path.join(
        FULL_INSTANCE_RAW_DIR, f"{stem}_fiber_instances.tiff",
    )
    instance_vis_path = os.path.join(
        FULL_INSTANCE_VIS_DIR, f"{stem}_fiber_instances_vis.png",
    )
    panel_path = os.path.join(FULL_PANEL_DIR, f"{stem}_panel.png")
    plt.imsave(original_path, img_rgb)
    plt.imsave(color_mask_path, color_mask)
    plt.imsave(overlay_path, overlay)
    plt.imsave(instance_vis_path, instance_vis)
    cv2.imwrite(raw_mask_path, semantic_mask)
    if fiber_instances.max() > np.iinfo(np.uint16).max:
        raise ValueError("More than 65,535 fiber instances; uint16 TIFF is insufficient.")
    cv2.imwrite(instance_raw_path, fiber_instances.astype(np.uint16))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(stem, fontsize=14)
    panels = [
        (img_rgb, "Original"), (color_mask, "Semantic mask"),
        (overlay, "Overlay"), (instance_vis, "Fiber instances"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    total_pixels = semantic_mask.size
    print(f" Saved -> {panel_path}")
    print(f" Fiber instances: {fiber_instances.max():,}")
    for class_id, class_name in CLASS_NAMES.items():
        pixel_count = int(np.sum(semantic_mask == class_id))
        percentage = 100.0 * pixel_count / total_pixels
        print(f" {class_name:8s}: {pixel_count:>10,} px ({percentage:5.1f}%)")
    return {
        "original_path": original_path,
        "raw_mask_path": raw_mask_path,
        "color_mask_path": color_mask_path,
        "overlay_path": overlay_path,
        "instance_raw_path": instance_raw_path,
        "instance_vis_path": instance_vis_path,
        "panel_path": panel_path,
    }

# =============================================================================
# TRAINING PATCH EXPORT WITH QC OVERLAYS
# =============================================================================

def export_training_patches(img_rgb, semantic_mask, stem):
    """Export image/mask pairs and matching overlays after full-image labeling."""
    patch_rows = []
    patches_img, _ = extract_patches(
        img_rgb, patch_size=TRAIN_PATCH_SIZE, overlap=TRAIN_PATCH_OVERLAP,
    )
    for x, y, img_patch in patches_img:
        patch_h, patch_w = img_patch.shape[:2]
        mask_patch = semantic_mask[y:y + patch_h, x:x + patch_w]
        if KEEP_ONLY_FULL_PATCHES:
            if patch_h != TRAIN_PATCH_SIZE or patch_w != TRAIN_PATCH_SIZE:
                continue
        patch_name = f"{stem}_y{y:05d}_x{x:05d}"
        image_patch_path = os.path.join(PATCH_IMAGE_DIR, f"{patch_name}.png")
        mask_patch_path = os.path.join(PATCH_MASK_DIR, f"{patch_name}_mask.png")
        overlay_patch_path = os.path.join(
            PATCH_OVERLAY_DIR, f"{patch_name}_overlay.png",
        )
        cv2.imwrite(
            image_patch_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(mask_patch_path, mask_patch)

        patch_color_mask = semantic_to_rgb(mask_patch)
        patch_overlay = cv2.addWeighted(
            img_patch, PATCH_OVERLAY_IMAGE_WEIGHT,
            patch_color_mask, PATCH_OVERLAY_MASK_WEIGHT, 0,
        )
        cv2.imwrite(
            overlay_patch_path,
            cv2.cvtColor(patch_overlay, cv2.COLOR_RGB2BGR),
        )

        class_counts = {
            class_id: int(np.sum(mask_patch == class_id))
            for class_id in CLASS_NAMES
        }
        patch_rows.append({
            "sample_id": patch_name,
            "parent_id": stem,
            "domain": "real",
            "image_path": image_patch_path,
            "mask_path": mask_patch_path,
            "overlay_path": overlay_patch_path,
            "image_type": "patch",
            "patch_x": x,
            "patch_y": y,
            "width": patch_w,
            "height": patch_h,
            "matrix_pixels": class_counts[0],
            "fiber_pixels": class_counts[1],
            "void_pixels": class_counts[2],
            "needs_review": "yes",
            "quality_flag": "unreviewed",
        })
    return patch_rows

# =============================================================================
# MANIFEST
# =============================================================================

MANIFEST_FIELDS = [
    "sample_id", "parent_id", "domain", "image_path", "mask_path",
    "overlay_path", "image_type", "patch_x", "patch_y", "width", "height",
    "matrix_pixels", "fiber_pixels", "void_pixels", "needs_review",
    "quality_flag", "label_mode", "segmentation_mode", "void_threshold",
    "fiber_threshold", "sam_proposals", "fiber_count", "median_fiber_area",
    "min_void_area",
]


def init_manifest():
    if os.path.isfile(MANIFEST_PATH):
        print(
            " WARNING: Recreating sam_manifest.csv. "
            "Keep reviewed decisions in a separate reviewed CSV."
        )
    with open(MANIFEST_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def append_manifest(rows):
    if not rows:
        return
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writerows(rows)

# =============================================================================
# SUMMARY GRID
# =============================================================================

def save_summary_grid(panel_folder):
    panel_files = sorted(
        filename for filename in os.listdir(panel_folder)
        if filename.endswith("_panel.png")
    )
    if not panel_files:
        return
    ncols = min(2, len(panel_files))
    nrows = math.ceil(len(panel_files) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 14, nrows * 6))
    axes = np.atleast_1d(axes).ravel()
    for ax, panel_file in zip(axes, panel_files):
        panel_path = os.path.join(panel_folder, panel_file)
        ax.imshow(plt.imread(panel_path))
        ax.set_title(panel_file.replace("_panel.png", ""), fontsize=10)
        ax.axis("off")
    for ax in axes[len(panel_files):]:
        ax.axis("off")
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_ROOT, "ALL_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary grid -> {summary_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"Input folder not found:\n{IMAGE_FOLDER}")
    image_files = sorted(
        filename for filename in os.listdir(IMAGE_FOLDER)
        if filename.lower().endswith(IMAGE_EXTS)
    )
    if not image_files:
        raise FileNotFoundError(f"No supported image files found in:\n{IMAGE_FOLDER}")
    init_manifest()
    print(f"\nFound {len(image_files)} image(s).\n")
    for filename in image_files:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        stem = os.path.splitext(filename)[0]
        print(f"> {filename}")
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(" WARNING: Could not read image; skipped.\n")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            semantic_mask, fiber_instances, info = segment_image(img_rgb)
            paths = save_full_outputs(
                img_rgb=img_rgb, semantic_mask=semantic_mask,
                fiber_instances=fiber_instances, stem=stem,
            )
            manifest_rows = []
            if EXPORT_PATCHES:
                patch_rows = export_training_patches(
                    img_rgb=img_rgb, semantic_mask=semantic_mask, stem=stem,
                )
                for row in patch_rows:
                    row.update({
                        "label_mode": "sam_assisted_full_image_then_patch",
                        "segmentation_mode": info["mode"],
                        "void_threshold": info["void_threshold"],
                        "fiber_threshold": info["fiber_threshold"],
                        "sam_proposals": info["sam_proposals"],
                        "fiber_count": info["fiber_count"],
                        "median_fiber_area": info["median_fiber_area"],
                        "min_void_area": info["min_void_area"],
                    })
                manifest_rows.extend(patch_rows)
            else:
                class_counts = {
                    class_id: int(np.sum(semantic_mask == class_id))
                    for class_id in CLASS_NAMES
                }
                manifest_rows.append({
                    "sample_id": stem,
                    "parent_id": stem,
                    "domain": "real",
                    "image_path": image_path,
                    "mask_path": paths["raw_mask_path"],
                    "overlay_path": paths["overlay_path"],
                    "image_type": "full",
                    "patch_x": 0,
                    "patch_y": 0,
                    "width": semantic_mask.shape[1],
                    "height": semantic_mask.shape[0],
                    "matrix_pixels": class_counts[0],
                    "fiber_pixels": class_counts[1],
                    "void_pixels": class_counts[2],
                    "needs_review": "yes",
                    "quality_flag": "unreviewed",
                    "label_mode": "sam_assisted_full_image",
                    "segmentation_mode": info["mode"],
                    "void_threshold": info["void_threshold"],
                    "fiber_threshold": info["fiber_threshold"],
                    "sam_proposals": info["sam_proposals"],
                    "fiber_count": info["fiber_count"],
                    "median_fiber_area": info["median_fiber_area"],
                    "min_void_area": info["min_void_area"],
                })
            append_manifest(manifest_rows)
            print(f" Done -> {len(manifest_rows)} training sample(s)\n")
        except Exception as error:
            print(f" FAILED: {error}\n")
    save_summary_grid(FULL_PANEL_DIR)
    if EXPORT_PATCHES:
        print(f"\nPatch overlays for QC -> {PATCH_OVERLAY_DIR}")
    print(f"Manifest -> {MANIFEST_PATH}")
    print(
        "Review overlays, then mark quality_flag as accepted or rejected "
        "in a separate reviewed manifest."
    )


if __name__ == "__main__":
    main()
