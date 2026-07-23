import os
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
    binary_dilation,
    disk,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_FOLDER = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/data/real_images"
)

OUTPUT_FOLDER = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/outputs/sam_masks"
)

SAM_CHECKPOINT = (
    "/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/"
    "Computations/CFRP-microstructure-segmentation/"
    "checkpoints/sam/sam_vit_h_4b8939.pth"
)

SAM_MODEL_TYPE = "vit_h"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# -----------------------------------------------------------------------------
# Patch settings
# -----------------------------------------------------------------------------

PATCH_SIZE = 512
OVERLAP = 64
PATCH_THRESHOLD_PX = 600 * 600

# -----------------------------------------------------------------------------
# Semantic classes
# -----------------------------------------------------------------------------

COLORS = {
    0: [128, 128, 128],  # matrix = gray
    1: [255,   0,   0],  # fiber = red
    2: [  0, 255,   0],  # void = green
}

# -----------------------------------------------------------------------------
# SAM automatic-mask generation
# -----------------------------------------------------------------------------

SAM_POINTS_PER_SIDE = 64
SAM_POINTS_PER_BATCH = 64
SAM_PRED_IOU_THRESH = 0.70
SAM_STABILITY_THRESH = 0.80
SAM_BOX_NMS_THRESH = 0.50
SAM_CROP_N_LAYERS = 1
SAM_CROP_NMS_THRESH = 0.50
SAM_CROP_OVERLAP_RATIO = 0.35
SAM_MIN_MASK_REGION_AREA = 5

# -----------------------------------------------------------------------------
# Fiber candidate filtering
# -----------------------------------------------------------------------------

FIBER_AREA_MIN = 8
FIBER_AREA_MAX = 2500
FIBER_CIRCULARITY_MIN = 0.45
FIBER_SOLIDITY_MIN = 0.60

FIBER_MASK_IOU_MAX = 0.20

# -----------------------------------------------------------------------------
# Fiber-interface expansion
#
# Radius 0: no enlargement
# Radius 1: recommended starting point, adds roughly one-pixel outer ring
# Radius 2: use only if fibers remain visibly undersized
# -----------------------------------------------------------------------------

FIBER_INTERFACE_DILATION_RADIUS = 1

# Used only to stop the new outer ring entering very dark pixels.
# This is not the main expansion mechanism.
FIBER_INTERFACE_INTENSITY_OFFSET = 35

# -----------------------------------------------------------------------------
# Void filtering
#
# Void must be at least this fraction of the median accepted fiber area.
# -----------------------------------------------------------------------------

VOID_AREA_FRACTION_OF_MEDIAN_FIBER = 1.0
VOID_FALLBACK_MIN_AREA = 40

# Reject thin or fragmented dark components.
VOID_MIN_SOLIDITY = 0.35
VOID_MIN_CIRCULARITY = 0.01


# =============================================================================
# LOAD SAM
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if not os.path.isfile(SAM_CHECKPOINT):
    raise FileNotFoundError(
        f"SAM checkpoint not found:\n{SAM_CHECKPOINT}"
    )

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
    """Compute intersection-over-union between two Boolean masks."""
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)

    return intersection / union if union > 0 else 0.0


def get_global_thresholds(gray):
    """
    Estimate per-image intensity thresholds.

    void threshold: dark-component candidate
    fiber threshold: proposal-level bright fiber criterion
    """
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

def extract_patches(img, patch_size=512, overlap=64):
    """Split an image into overlapping patches."""
    h, w = img.shape[:2]
    step = patch_size - overlap

    if step <= 0:
        raise ValueError("PATCH_SIZE must be larger than OVERLAP.")

    x_last = max(w - patch_size, 0)
    y_last = max(h - patch_size, 0)

    x_positions = list(range(0, x_last + 1, step))
    y_positions = list(range(0, y_last + 1, step))

    if x_positions[-1] != x_last:
        x_positions.append(x_last)

    if y_positions[-1] != y_last:
        y_positions.append(y_last)

    patches = []

    for y in y_positions:
        for x in x_positions:
            patch = img[
                y:min(y + patch_size, h),
                x:min(x + patch_size, w),
            ]
            patches.append((x, y, patch))

    return patches, (h, w)


def merge_patches_vote(patch_results, full_size, n_classes=3):
    """Merge semantic patch masks using class-wise majority voting."""
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
    """
    Select plausible SAM fiber proposals and remove duplicate masks.

    Accepted masks are not merged here.
    """
    candidates = []

    for sam_mask in sam_masks:
        proposal = sam_mask["segmentation"].astype(bool)
        components = label(proposal, connectivity=2)

        for prop in regionprops(components, intensity_image=gray):
            if prop.area < FIBER_AREA_MIN:
                continue

            if prop.area > FIBER_AREA_MAX:
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

            candidates.append(
                {
                    "mask": component,
                    "score": score,
                    "area": prop.area,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    accepted = []

    for candidate in candidates:
        is_duplicate = any(
            mask_iou(candidate["mask"], selected["mask"])
            >= FIBER_MASK_IOU_MAX
            for selected in accepted
        )

        if not is_duplicate:
            accepted.append(candidate)

    return accepted


def build_fiber_instances(selected_fibers, gray, fiber_threshold):
    """
    Create separate labeled fiber instances.

    Each fiber:
    1. Starts from its accepted SAM mask.
    2. Expands independently by FIBER_INTERFACE_DILATION_RADIUS.
    3. Is constrained to avoid very dark pixels.
    4. Claims non-occupied pixels, preserving higher-score masks first.

    No global union or dilation is used.
    """
    h, w = gray.shape

    fiber_instances = np.zeros((h, w), dtype=np.int32)
    occupied = np.zeros((h, w), dtype=bool)

    interface_threshold = max(
        0,
        int(fiber_threshold - FIBER_INTERFACE_INTENSITY_OFFSET),
    )

    footprint = disk(FIBER_INTERFACE_DILATION_RADIUS)

    next_id = 1

    for item in selected_fibers:
        candidate = item["mask"].copy()

        # Expand this object alone to recover fiber-interface pixels.
        if FIBER_INTERFACE_DILATION_RADIUS > 0:
            candidate = binary_dilation(
                candidate,
                footprint=footprint,
            )

        # The expanded ring may not enter very dark regions / true voids.
        candidate &= gray >= interface_threshold

        # A better-scored fiber already owns overlap pixels.
        candidate &= ~occupied

        # Discard tiny remnants after overlap resolution.
        if np.count_nonzero(candidate) < FIBER_AREA_MIN:
            continue

        fiber_instances[candidate] = next_id
        occupied[candidate] = True
        next_id += 1

    return fiber_instances, interface_threshold


def relabel_instances(instance_mask):
    """Relabel all positive connected components consecutively."""
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
    """Compute minimum valid void area from observed fiber areas."""
    fiber_areas = np.array([
        prop.area
        for prop in regionprops(fiber_instances)
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
    """
    Detect dark void components, rejecting small and line-like artifacts.
    """
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
    """Apply final void-size filtering after patch merging."""
    min_void_area, median_fiber_area = get_min_void_area(fiber_instances)

    labels = label(semantic_mask == 2, connectivity=2)
    kept_voids = np.zeros(semantic_mask.shape, dtype=bool)

    for prop in regionprops(labels):
        if prop.area >= min_void_area:
            kept_voids[labels == prop.label] = True

    output = semantic_mask.copy()
    output[output == 2] = 0
    output[kept_voids] = 2

    return output, min_void_area, median_fiber_area


# =============================================================================
# TILE SEGMENTATION
# =============================================================================

def segment_tile(img_rgb, gray, void_threshold, fiber_threshold):
    """
    Segment one full image or one patch.

    Class labels:
      0 = matrix
      1 = fiber
      2 = void
    """
    sam_masks = mask_generator.generate(img_rgb)

    selected_fibers = select_sam_fiber_masks(
        sam_masks=sam_masks,
        gray=gray,
        fiber_threshold=fiber_threshold,
    )

    fiber_instances, interface_threshold = build_fiber_instances(
        selected_fibers=selected_fibers,
        gray=gray,
        fiber_threshold=fiber_threshold,
    )

    fiber_instances = relabel_instances(fiber_instances)

    min_void_area, median_fiber_area = get_min_void_area(fiber_instances)

    void_mask = detect_and_filter_voids(
        gray=gray,
        void_threshold=void_threshold,
        min_void_area=min_void_area,
    )

    semantic_mask = np.zeros(gray.shape, dtype=np.uint8)
    semantic_mask[fiber_instances > 0] = 1
    semantic_mask[void_mask] = 2

    print(
        f"      SAM proposals={len(sam_masks)}, "
        f"fibers={fiber_instances.max()}, "
        f"interface threshold={interface_threshold}, "
        f"median fiber area={median_fiber_area:.1f}px, "
        f"minimum void area={min_void_area}px"
    )

    return semantic_mask, fiber_instances


# =============================================================================
# FULL IMAGE / PATCH DISPATCHER
# =============================================================================

def segment_image(img_rgb):
    """
    Return:
      semantic_mask: 0=matrix, 1=fiber, 2=void
      fiber_instances: integer fiber-label map
    """
    gray = rgb_to_gray_uint8(img_rgb)
    h, w = gray.shape

    void_threshold, fiber_threshold = get_global_thresholds(gray)

    print(
        f"  Thresholds: "
        f"void < {void_threshold:.1f}, "
        f"fiber core >= {fiber_threshold:.1f}, "
        f"dilation radius={FIBER_INTERFACE_DILATION_RADIUS}"
    )

    # Full-image mode
    if h * w <= PATCH_THRESHOLD_PX:
        print(f"  Image {w}x{h}: full-image mode")

        return segment_tile(
            img_rgb=img_rgb,
            gray=gray,
            void_threshold=void_threshold,
            fiber_threshold=fiber_threshold,
        )

    # Patch mode
    patches, full_size = extract_patches(
        img_rgb,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
    )

    print(
        f"  Image {w}x{h}: patch mode "
        f"({len(patches)} patches, size={PATCH_SIZE}, overlap={OVERLAP})"
    )

    patch_results = []

    for patch_index, (x, y, patch_rgb) in enumerate(patches, start=1):
        patch_gray = rgb_to_gray_uint8(patch_rgb)

        print(
            f"    Patch {patch_index}/{len(patches)} "
            f"at ({x}, {y})"
        )

        patch_semantic, _ = segment_tile(
            img_rgb=patch_rgb,
            gray=patch_gray,
            void_threshold=void_threshold,
            fiber_threshold=fiber_threshold,
        )

        patch_results.append((x, y, patch_semantic))

    semantic_mask = merge_patches_vote(
        patch_results=patch_results,
        full_size=full_size,
        n_classes=3,
    )

    # After patch voting, each connected red area is an instance.
    fiber_instances = label(semantic_mask == 1, connectivity=2)
    fiber_instances = relabel_instances(fiber_instances)

    semantic_mask, min_void_area, median_fiber_area = (
        filter_existing_voids_by_fiber_area(
            semantic_mask=semantic_mask,
            fiber_instances=fiber_instances,
        )
    )

    # Preserve fibers if a patch-voted void overlaps a fiber.
    semantic_mask[fiber_instances > 0] = 1

    print(
        f"  Final fibers={fiber_instances.max()}, "
        f"median fiber area={median_fiber_area:.1f}px, "
        f"minimum void area={min_void_area}px"
    )

    return semantic_mask, fiber_instances


# =============================================================================
# VISUALIZATION AND SAVING
# =============================================================================

def semantic_to_rgb(semantic_mask):
    """Convert class labels into an RGB display image."""
    color_mask = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)

    for class_id, color in COLORS.items():
        color_mask[semantic_mask == class_id] = color

    return color_mask


def make_instance_visualization(fiber_instances):
    """Create a deterministic random-color image for fiber IDs."""
    n_instances = int(fiber_instances.max())

    if n_instances == 0:
        return np.zeros((*fiber_instances.shape, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed=42)

    color_lut = rng.integers(
        low=35,
        high=256,
        size=(n_instances + 1, 3),
        dtype=np.uint8,
    )

    color_lut[0] = [0, 0, 0]

    return color_lut[fiber_instances]


def save_results(img_rgb, semantic_mask, fiber_instances, out_dir, stem):
    """Write semantic output, instance output, overlay, and panel image."""
    color_mask = semantic_to_rgb(semantic_mask)

    overlay = cv2.addWeighted(
        img_rgb,
        0.60,
        color_mask,
        0.40,
        0,
    )

    instance_vis = make_instance_visualization(fiber_instances)

    original_path = os.path.join(out_dir, f"{stem}_original.png")
    semantic_path = os.path.join(out_dir, f"{stem}_semantic_mask.png")
    raw_path = os.path.join(out_dir, f"{stem}_semantic_mask_raw.png")
    overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
    instances_path = os.path.join(out_dir, f"{stem}_fiber_instances.tiff")
    instances_vis_path = os.path.join(
        out_dir,
        f"{stem}_fiber_instances_vis.png",
    )
    panel_path = os.path.join(out_dir, f"{stem}_comparison.png")

    plt.imsave(original_path, img_rgb)
    plt.imsave(semantic_path, color_mask)
    plt.imsave(overlay_path, overlay)
    plt.imsave(instances_vis_path, instance_vis)

    cv2.imwrite(raw_path, semantic_mask)

    if fiber_instances.max() > np.iinfo(np.uint16).max:
        raise ValueError(
            "More than 65,535 fiber instances; uint16 TIFF is insufficient."
        )

    cv2.imwrite(
        instances_path,
        fiber_instances.astype(np.uint16),
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(stem, fontsize=14)

    panels = [
        (img_rgb, "Original"),
        (color_mask, "Semantic Mask"),
        (overlay, "Semantic Overlay"),
        (instance_vis, "Fiber Instances"),
    ]

    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    total_pixels = semantic_mask.size

    print(f"  Saved -> {panel_path}")
    print(f"    Fiber instances: {fiber_instances.max():,}")

    for class_id, class_name in [
        (0, "matrix"),
        (1, "fiber"),
        (2, "void"),
    ]:
        pixel_count = int(np.sum(semantic_mask == class_id))
        percentage = 100.0 * pixel_count / total_pixels

        print(
            f"    {class_name:8s}: "
            f"{pixel_count:>10,} px ({percentage:5.1f}%)"
        )


# =============================================================================
# SUMMARY GRID
# =============================================================================

def save_summary_grid(output_folder):
    """Create one grid image containing all comparison panels."""
    panel_files = sorted(
        filename
        for filename in os.listdir(output_folder)
        if filename.endswith("_comparison.png")
    )

    if not panel_files:
        return

    ncols = min(2, len(panel_files))
    nrows = math.ceil(len(panel_files) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 14, nrows * 6),
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, panel_file in zip(axes, panel_files):
        panel_path = os.path.join(output_folder, panel_file)
        ax.imshow(plt.imread(panel_path))
        ax.set_title(
            panel_file.replace("_comparison.png", ""),
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(panel_files):]:
        ax.axis("off")

    plt.tight_layout()

    summary_path = os.path.join(
        output_folder,
        "ALL_summary.png",
    )

    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSummary grid -> {summary_path}")


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(
            f"Input folder not found:\n{IMAGE_FOLDER}"
        )

    image_files = sorted(
        filename
        for filename in os.listdir(IMAGE_FOLDER)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    if not image_files:
        raise FileNotFoundError(
            f"No supported image files found in:\n{IMAGE_FOLDER}"
        )

    print(f"\nFound {len(image_files)} image(s).\n")

    for filename in image_files:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        stem = os.path.splitext(filename)[0]

        print(f"▶ {filename}")

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("  WARNING: Could not read image; skipped.\n")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            semantic_mask, fiber_instances = segment_image(img_rgb)

            save_results(
                img_rgb=img_rgb,
                semantic_mask=semantic_mask,
                fiber_instances=fiber_instances,
                out_dir=OUTPUT_FOLDER,
                stem=stem,
            )

            print("  Done\n")

        except Exception as error:
            print(f"  FAILED: {error}\n")

    save_summary_grid(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()