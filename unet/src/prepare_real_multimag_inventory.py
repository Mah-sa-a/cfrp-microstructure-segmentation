#!/usr/bin/env python3
"""Build an audited, non-destructive inventory of reviewed real SAM patches."""
import argparse
import csv
import hashlib
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BATCHES = {
    "sam_l40s_parallel_799154": ["1", "R1_1", "R2", "R3_1a", "R3_1b"],
    "sam_700x_808474": ["R1_1", "R2", "R3_1a", "R3_1b"],
}
BOUNDARY = "20250722_162407"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    project = args.project.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output or project / "data" / "datasets" /
           f"real_multimag_inventory_{stamp}_{uuid.uuid4().hex[:6]}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    records, errors, manifests, parents = [], [], [], []
    duplicate_images = defaultdict(list)
    def error(batch, group, sid, message):
        errors.append(dict(batch=batch, source_group=group,
                           sample_id=sid, issue=message))
    for batch, groups in BATCHES.items():
        for group in groups:
            root = project / "outputs" / batch / group
            manifest = root / "sam_manifest.csv"
            if not manifest.is_file():
                error(batch, group, "", f"Missing manifest: {manifest}")
                continue
            manifests.append(dict(path=str(manifest), sha256=sha256(manifest)))
            with manifest.open(newline="") as f:
                reader = csv.DictReader(f)
                required = {"sample_id", "parent_id", "image_type"}
                if not required.issubset(reader.fieldnames or []):
                    error(batch, group, "", "Missing required manifest columns")
                    continue
                source_rows = list(reader)
            reject = root / "patches" / "reject"
            rejected = defaultdict(lambda: {"image": [], "mask": []})
            if not reject.is_dir():
                error(batch, group, "", "Reject directory is missing")
            else:
                for p in sorted(reject.rglob("*")):
                    if not p.is_file() or p.suffix.lower() != ".png":
                        continue
                    stem = p.stem
                    if stem.endswith("_overlay"):
                        continue
                    kind = "mask" if stem.endswith("_mask") else "image"
                    sid = stem[:-5] if kind == "mask" else stem
                    rejected[sid][kind].append(p.resolve())
            seen = set()
            parent_counts = Counter()
            for src in source_rows:
                sid, parent = src["sample_id"], src["parent_id"]
                if src["image_type"] != "patch":
                    error(batch, group, sid, "Unexpected non-patch manifest row")
                    continue
                if not sid or Path(sid).name != sid or not parent:
                    error(batch, group, sid, "Invalid sample_id or parent_id")
                    continue
                if sid in seen:
                    error(batch, group, sid, "Duplicate sample_id within manifest")
                    continue
                seen.add(sid)
                parent_counts[parent] += 1
                magnification = 700 if batch == "sam_700x_808474" else 300
                if batch == "sam_l40s_parallel_799154" and group == "1":
                    magnification = 700 if parent >= BOUNDARY else 300
                active_i = root / "patches" / "images" / (sid + ".png")
                active_m = root / "patches" / "masks" / (sid + "_mask.png")
                rejected_pair = sid in rejected
                choices = rejected.get(sid, {"image": [], "mask": []})
                # Rejection takes precedence over normal-folder copies.
                if rejected_pair:
                    image_options = choices["image"]
                    mask_options = choices["mask"]

                    copy_mismatch = False
                    for active, rejected_options, kind in (
                        (active_i, image_options, "image"),
                        (active_m, mask_options, "mask"),
                    ):
                        if active.is_file() and len(rejected_options) == 1:
                            if sha256(active) != sha256(rejected_options[0]):
                                error(
                                    batch, group, sid,
                                    f"Normal-folder and rejected {kind} copies differ"
                                )
                                copy_mismatch = True
                else:
                    image_options = (
                        [active_i.resolve()] if active_i.is_file() else []
                    )
                    mask_options = (
                        [active_m.resolve()] if active_m.is_file() else []
                    )
                    copy_mismatch = False
                row = dict(sample_uid=f"{batch}::{group}::{sid}",
                           annotation_batch=batch, source_group=group,
                           split_group_id=group, split_unit="physical_sample",
                           parent_id=parent, parent_uid=f"{group}::{parent}",
                           sample_id=sid, magnification=magnification,
                           magnification_basis="user_confirmed_parent_acquisition_mapping",
                           quality_flag="reject" if rejected_pair else "good",
                           needs_review="no", review_basis="user_confirmed_complete_visual_screening",
                           rejection_reason="not_recorded" if rejected_pair else "",
                           original_image_path=src.get("image_path", ""),
                           original_mask_path=src.get("mask_path", ""),
                           image_path="", mask_path="", integrity_status="error",
                           eligible=False, width="", height="", matrix_pixels="",
                           fiber_pixels="", void_pixels="", n_present_classes="",
                           single_class="", image_sha256="", mask_sha256="",
                           patch_x=src.get("patch_x", ""), patch_y=src.get("patch_y", ""))
                records.append(row)
                if copy_mismatch:
                    continue
                if len(image_options) != 1 or len(mask_options) != 1:
                    error(batch, group, sid,
                          f"Expected exactly one image and mask; found {len(image_options)}, {len(mask_options)}")
                    continue
                if rejected_pair and (len(choices["image"]) != 1 or len(choices["mask"]) != 1):
                    error(batch, group, sid, "Rejected pair is incomplete inside reject directory")
                    continue
                ip, mp = image_options[0], mask_options[0]
                row.update(image_path=str(ip), mask_path=str(mp))
                try:
                    with Image.open(ip) as im:
                        im.load()
                        size = im.size
                        if im.mode != "RGB":
                            raise ValueError(f"Expected RGB image; found {im.mode}")
                    with Image.open(mp) as im:
                        mask = np.asarray(im)
                    if mask.ndim != 2:
                        raise ValueError(f"Raw mask must be 2D; shape={mask.shape}")
                    if size != (512, 512) or mask.shape != (512, 512):
                        raise ValueError(f"Unexpected image/mask sizes: {size}, {mask.shape}")
                    labels = np.unique(mask)
                    if not np.isin(labels, [0, 1, 2]).all():
                        raise ValueError(f"Invalid mask values: {labels.tolist()}")
                    counts = np.bincount(mask.astype(np.int64).ravel(), minlength=3)
                    ih, mh = sha256(ip), sha256(mp)
                    row.update(width=size[0], height=size[1],
                               matrix_pixels=int(counts[0]), fiber_pixels=int(counts[1]),
                               void_pixels=int(counts[2]), n_present_classes=len(labels),
                               single_class=len(labels) == 1, image_sha256=ih,
                               mask_sha256=mh, integrity_status="ok", eligible=not rejected_pair)
                    if not rejected_pair:
                        duplicate_images[ih].append(row)
                except Exception as exc:
                    error(batch, group, sid, str(exc))
            for sid in rejected:
                if sid not in seen:
                    error(batch, group, sid, "Rejected ID is absent from this group's manifest")
            for parent, count in sorted(parent_counts.items()):
                mag = 700 if batch == "sam_700x_808474" else 300
                if batch == "sam_l40s_parallel_799154" and group == "1":
                    mag = 700 if parent >= BOUNDARY else 300
                parents.append(dict(annotation_batch=batch, source_group=group,
                                    parent_id=parent, magnification=mag,
                                    split_group_id=group, manifest_patch_count=count))
            print(f"{batch}/{group}: indexed {len(seen)} patches", flush=True)
    duplicates = []
    for digest, matches in duplicate_images.items():
        if len(matches) > 1:
            cross_group = len({r["source_group"] for r in matches}) > 1
            for r in matches:
                duplicates.append(dict(image_sha256=digest, sample_uid=r["sample_uid"],
                                       source_group=r["source_group"],
                                       cross_sample=cross_group, mask_sha256=r["mask_sha256"]))
    summary = []
    keys = sorted({(r["annotation_batch"], r["source_group"], r["magnification"]) for r in records})
    for batch, group, mag in keys:
        rs = [r for r in records if (r["annotation_batch"], r["source_group"], r["magnification"]) == (batch, group, mag)]
        summary.append(dict(annotation_batch=batch, source_group=group, magnification=mag,
                            parents=len({r["parent_id"] for r in rs}), indexed_patches=len(rs),
                            rejected=sum(r["quality_flag"] == "reject" for r in rs),
                            integrity_errors=sum(r["integrity_status"] != "ok" for r in rs),
                            eligible_candidates=sum(r["eligible"] for r in rs),
                            eligible_single_class=sum(r["eligible"] and r["single_class"] is True for r in rs)))
    if records:
        write_csv(out / "master_index.csv", records, list(records[0]))
        write_csv(out / "rejected_pairs.csv", [r for r in records if r["quality_flag"] == "reject"], list(records[0]))
    write_csv(out / "parent_metadata.csv", parents,
              ["annotation_batch", "source_group", "parent_id", "magnification", "split_group_id", "manifest_patch_count"])
    write_csv(out / "integrity_report.csv", errors, ["batch", "source_group", "sample_id", "issue"])
    write_csv(out / "dataset_summary.csv", summary,
              ["annotation_batch", "source_group", "magnification", "parents", "indexed_patches", "rejected", "integrity_errors", "eligible_candidates", "eligible_single_class"])
    write_csv(out / "duplicate_image_report.csv", duplicates,
              ["image_sha256", "sample_uid", "source_group", "cross_sample", "mask_sha256"])
    provenance = dict(created_utc=stamp, project=str(project), batches=BATCHES,
                      excluded_source_groups=["R1_2a"], group1_700x_start=BOUNDARY,
                      review_complete="all nine batch/sample combinations, user confirmed",
                      split_unit="physical_sample; tube independence not established",
                      single_class_policy="retained in inventory; training-only filtering deferred",
                      no_splits_created=True, source_manifests=manifests,
                      script_sha256=sha256(Path(__file__)),
                      integrity_error_count=len(errors), duplicate_image_rows=len(duplicates),
                      status="BLOCKED_INTEGRITY" if errors else ("REVIEW_DUPLICATES" if duplicates else "INVENTORY_OK"))
    (out / "preparation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nInventory directory: {out}")
    print(f"Indexed: {len(records)}; rejected: {sum(r['quality_flag'] == 'reject' for r in records)}")
    print(f"Eligible candidates: {sum(r['eligible'] for r in records)}")
    print(f"Integrity errors: {len(errors)}; duplicate-image rows: {len(duplicates)}")
    print(f"Status: {provenance['status']}; no folds or training jobs created.")
    sys.exit(2 if errors else 0)

if __name__ == "__main__":
    main()
