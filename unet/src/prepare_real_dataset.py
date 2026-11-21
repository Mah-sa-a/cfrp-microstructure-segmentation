"""Refresh SAM patch manifests and create grouped cross-validation CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["source_folder", "sample_id", "preparation_status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def basename(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="Exploratory only: allow unreviewed labels in all splits.")
    parser.add_argument("--run-name", default="real_5fold")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config or root / "unet/configs/dataset_preparation.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run-name must be a single directory name")
    source = root / cfg["sam_output_root"]
    manifest_dir = root / "data/manifests" / args.run_name
    split_dir = root / "data/splits" / args.run_name
    if manifest_dir.exists() or split_dir.exists():
        raise FileExistsError("Run outputs already exist. Choose a new --run-name.")
    samples = cfg["included_samples"]
    if len(samples) != len(set(samples)):
        raise ValueError("Duplicate included sample names")
    metadata, _ = read_csv(root / cfg["sample_metadata"])
    meta = {}
    for row in metadata:
        s = row["sample_id"].strip()
        if s in meta:
            raise ValueError(f"Duplicate metadata: {s}")
        meta[s] = row
    for s in samples:
        if s not in meta or not meta[s]["split_group_id"].strip():
            raise ValueError(f"Missing group metadata for {s}")
    groups = sorted({meta[s]["split_group_id"].strip() for s in samples})
    if len(groups) < 3:
        raise ValueError("At least three groups are required for this train/validation/test scheme.")
    allowed = {"good", "unreviewed"} if args.allow_unreviewed else {"good"}
    all_rows, eligible, rejects, missing, quality_excluded = [], [], [], [], []
    seen_ids, seen_paths, image_hashes = set(), set(), {}
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    input_hashes = {}
    for sample in samples:
        folder = source / sample
        manifest = folder / "sam_manifest.csv"
        rows, fields = read_csv(manifest)
        input_hashes[sample] = digest(manifest)
        required = {"sample_id", "parent_id", "image_type", "image_path", "mask_path", "quality_flag"}
        if not required.issubset(fields):
            raise ValueError(f"{sample}: missing columns {sorted(required-set(fields))}")
        patches = [r for r in rows if r["image_type"].strip().lower() == "patch"]
        if not patches:
            raise ValueError(f"{sample}: no patch records")
        lookup = {}
        for row in patches:
            for column in ("image_path", "mask_path"):
                name = basename(row[column])
                if not name:
                    raise ValueError(f"{sample}: empty {column}")
                if name in lookup and lookup[name] != row["sample_id"]:
                    raise ValueError(f"{sample}: ambiguous filename {name}")
                lookup[name] = row["sample_id"]
        reject_files = set()
        for d in folder.rglob("*"):
            if d.is_dir() and d.name.lower() in cfg["reject_folder_names"]:
                reject_files.update(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        unknown = [str(p) for p in reject_files if p.name not in lookup]
        if unknown:
            raise ValueError(f"{sample}: unmatched rejected filenames; check folder contents: {unknown[:10]}")
        rejected_ids = {lookup[p.name] for p in reject_files}
        for original in patches:
            row = original.copy()
            uid = sample + "::" + row["sample_id"]
            if uid in seen_ids:
                raise ValueError(f"Duplicate ID: {uid}")
            seen_ids.add(uid)
            group = meta[sample]["split_group_id"].strip()
            row.update(source_folder=sample, dataset_sample_id=uid, split_group_id=group,
                       winding_configuration=meta[sample].get("winding_configuration", "unknown"),
                       original_image_path=row["image_path"], original_mask_path=row["mask_path"])
            image = folder / "patches/images" / basename(row["image_path"])
            mask = folder / "patches/masks" / basename(row["mask_path"])
            row["image_path"] = image.relative_to(root).as_posix()
            row["mask_path"] = mask.relative_to(root).as_posix()
            flag = row["quality_flag"].strip().lower()
            if row["sample_id"] in rejected_ids or flag == "reject":
                row.update(quality_flag="reject", needs_review="no", preparation_status="rejected")
                rejects.append(row)
            elif not image.is_file() or not mask.is_file():
                row["preparation_status"] = "missing_image_or_mask"
                missing.append(row)
            elif flag not in allowed:
                row["preparation_status"] = "excluded_quality_status"
                quality_excluded.append(row)
            else:
                if str(image.resolve()) in seen_paths:
                    raise ValueError(f"Repeated image path: {image}")
                seen_paths.add(str(image.resolve()))
                sha = digest(image)
                if sha in image_hashes and image_hashes[sha][0] != group:
                    raise ValueError(f"Identical image bytes across groups: {uid}, {image_hashes[sha][1]}")
                image_hashes[sha] = (group, uid)
                row.update(image_sha256=sha, preparation_status="eligible")
                eligible.append(row)
            all_rows.append(row)
    absent = set(samples) - {r["source_folder"] for r in eligible}
    if absent:
        raise ValueError(f"No eligible patches for {sorted(absent)}. Default requires quality_flag=good. "
                         "Use --allow-unreviewed only for exploratory runs; check missing files too.")
    # Do not silently split an incomplete inventory.
    if missing:
        raise ValueError(f"{len(missing)} non-rejected records have missing files. "
                         f"First examples: {[r['dataset_sample_id'] for r in missing[:10]]}")
    write_csv(manifest_dir / "all_patches_reviewed.csv", all_rows)
    write_csv(manifest_dir / "eligible_patches.csv", eligible)
    write_csv(manifest_dir / "rejected_patches.csv", rejects)
    write_csv(manifest_dir / "excluded_quality_records.csv", quality_excluded)
    summaries = []
    for i, test in enumerate(groups):
        val = groups[(i+1) % len(groups)]
        assigned = {"train": [g for g in groups if g not in (test, val)],
                    "validation": [val], "test": [test]}
        fold_dir = split_dir / f"fold_{i+1:02d}"
        for partition, partition_groups in assigned.items():
            subset = [dict(r, split=partition) for r in eligible if r["split_group_id"] in partition_groups]
            write_csv(fold_dir / f"{partition}.csv", subset)
            summaries.append(dict(fold=i+1, split=partition, groups="|".join(partition_groups),
                                  samples="|".join(sorted({r["source_folder"] for r in subset})),
                                  patches=len(subset),
                                  configurations="|".join(sorted({r["winding_configuration"] for r in subset}))))
        (fold_dir / "assignment.json").write_text(json.dumps(assigned, indent=2), encoding="utf-8")
    write_csv(split_dir / "split_summary.csv", summaries)
    audit = dict(configuration=cfg, input_manifest_sha256=input_hashes,
                 metadata_sha256=digest(root / cfg["sample_metadata"]),
                 allow_unreviewed=args.allow_unreviewed, groups=groups,
                 path_convention="image_path and mask_path relative to repository root",
                 scheme="leave-one-group-out test; cyclic held-out validation group",
                 eligible=len(eligible), rejected=len(rejects), quality_excluded=len(quality_excluded))
    (split_dir / "preparation_config.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Prepared {len(groups)} folds, {len(eligible)} eligible patches, {len(rejects)} rejected patches.")
    print(f"Inspect {split_dir / 'split_summary.csv'} before training.")
    print("WARNING: verify parent-tube relationships and winding-configuration coverage.")
    if args.allow_unreviewed:
        print("EXPLORATORY: validation/test may contain unreviewed annotations.")

if __name__ == "__main__":
    main()
