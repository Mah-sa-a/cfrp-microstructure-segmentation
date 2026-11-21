#!/usr/bin/env python3
"""Freeze reviewed multimag splits using the previous sample-wise fold assignments."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

GROUPS = {"1", "R1_1", "R2", "R3_1a", "R3_1b"}

def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def truth(v):
    return str(v).strip().lower() in {"true", "1", "yes"}

def write_csv(p, rows, fields):
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("--inventory", type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    inv = args.inventory.resolve()
    master = inv / "master_index.csv"
    provenance = json.loads((inv / "preparation_provenance.json").read_text())
    require(provenance["status"] == "INVENTORY_OK", "Inventory has not passed its audit")
    _, all_rows = read_csv(master)
    rows = [r for r in all_rows if truth(r["eligible"])]
    require(len(rows) > 0, "No eligible rows")
    require({r["source_group"] for r in rows} == GROUPS, "Unexpected eligible sample groups")
    require(len({r["sample_uid"] for r in rows}) == len(rows), "Duplicate sample_uid")
    require(len({r["image_path"] for r in rows}) == len(rows), "Repeated image path")
    require(len({r["image_sha256"] for r in rows}) == len(rows), "Repeated image-file hash")
    for r in rows:
        require(r["quality_flag"] == "good" and r["needs_review"] == "no", "Unapproved eligible row")
        require(r["integrity_status"] == "ok", "Invalid eligible row")
        require(r["split_group_id"] == r["source_group"], "Unexpected split grouping")
        require(r["magnification"] in {"300", "700"}, "Unknown magnification")
        require(sum(int(r[k]) for k in ("matrix_pixels", "fiber_pixels", "void_pixels")) == 512*512,
                "Invalid pixel counts")
        for key in ("image_path", "mask_path"):
            require(Path(r[key]).is_file(), f"Missing file: {r[key]}")
    for g in GROUPS:
        require({r["magnification"] for r in rows if r["source_group"] == g} == {"300", "700"},
                f"Missing magnification for {g}")

    old = project / "data/splits/mixedclass_train_only_20261106_180359_877663"
    old_config = project / "unet/configs/real_only_mixedclass_fp32_v1.json"
    config = json.loads(old_config.read_text())
    require(config.get("precision") == "float32", "Previous configuration is not float32")
    plans, old_sources, template_fields = [], [], None
    for i in range(1,6):
        groups_by_split = {}
        for split in ("train", "validation", "test"):
            p = old / f"fold_{i:02d}" / f"{split}.csv"
            fields, previous = read_csv(p)
            if template_fields is None:
                template_fields = fields
            require(fields == template_fields, f"CSV schema differs in {p}")
            gs = {r["split_group_id"] for r in previous}
            groups_by_split[split] = gs
            old_sources.append({"path":str(p), "sha256":digest(p)})
        tr, va, te = (groups_by_split[k] for k in ("train","validation","test"))
        require(len(tr)==3 and len(va)==1 and len(te)==1, f"Unexpected fold sizes: {i}")
        require(not(tr & va or tr & te or va & te) and tr | va | te == GROUPS,
                f"Invalid previous fold grouping: {i}")
        plans.append(groups_by_split)
    require(Counter(next(iter(p["test"])) for p in plans) == Counter({g:1 for g in GROUPS}),
            "Each sample must be test exactly once")
    require(Counter(next(iter(p["validation"])) for p in plans) == Counter({g:1 for g in GROUPS}),
            "Each sample must be validation exactly once")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"reviewed_multimag_mixedclass_fp32_{stamp}_{uuid.uuid4().hex[:6]}"
    output = project / "data/splits" / tag
    output.mkdir(parents=True, exist_ok=False)
    config_path = project / "unet/configs" / f"{tag}.json"
    extras = ["sample_uid", "source_group", "annotation_batch", "magnification", "parent_uid",
              "split_unit", "mask_sha256", "n_present_classes", "single_class", "review_basis",
              "integrity_status", "eligible"]
    fields = template_fields + [k for k in extras if k not in template_fields]
    generated, summary, excluded, audit = [], [], [], []
    test_seen = Counter()
    try:
        for i, plan in enumerate(plans,1):
            fold_dir = output / f"fold_{i:02d}"
            fold_dir.mkdir()
            chosen = {}
            for split in ("train", "validation", "test"):
                candidates = [r for r in rows if r["source_group"] in plan[split]]
                kept = [r for r in candidates if split != "train" or int(r["n_present_classes"]) > 1]
                removed = [r for r in candidates if split == "train" and int(r["n_present_classes"]) == 1]
                for r in removed:
                    excluded.append(dict(fold=f"fold_{i:02d}", sample_uid=r["sample_uid"],
                                         source_group=r["source_group"], magnification=r["magnification"],
                                         reason="training_only_single_class"))
                converted = []
                for r in sorted(kept,key=lambda x:x["sample_uid"]):
                    x = dict(r)
                    x.update(domain="real", image_type="patch", split=split,
                             dataset_sample_id=r["sample_uid"],
                             source_folder=r["source_group"],
                             preparation_status="eligible")
                    # Unavailable legacy diagnostics stay blank, rather than being fabricated.
                    converted.append(x)
                    if split == "test":
                        test_seen[r["sample_uid"]] += 1
                path = fold_dir / f"{split}.csv"
                write_csv(path, converted, fields)
                generated.append(dict(path=str(path.relative_to(project)),sha256=digest(path)))
                chosen[split] = converted
                summary.append(dict(fold=f"fold_{i:02d}",split=split,
                                    samples=";".join(sorted(plan[split])),
                                    before_training_filter=len(candidates),count=len(converted),
                                    excluded_single_class=len(removed),
                                    patches_300=sum(r["magnification"]=="300" for r in converted),
                                    patches_700=sum(r["magnification"]=="700" for r in converted)))
            for key in ("source_group","parent_uid","image_path","image_sha256"):
                sets = [{r[key] for r in chosen[s]} for s in ("train","validation","test")]
                require(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),
                        f"Within-fold overlap detected: fold {i}, {key}")
            audit.append(dict(fold=f"fold_{i:02d}",train_groups=sorted(plan["train"]),
                              validation_groups=sorted(plan["validation"]),test_groups=sorted(plan["test"]),
                              within_fold_checks="passed"))
        require(len(test_seen)==len(rows) and all(n==1 for n in test_seen.values()),
                "Test coverage is not exactly once per eligible patch")
        write_csv(output / "split_summary.csv",summary,list(summary[0]))
        write_csv(output / "excluded_training_rows.csv",excluded,
                  ["fold","sample_uid","source_group","magnification","reason"])
        shutil.copy2(inv / "parent_metadata.csv",output / "parent_metadata.csv")
        config.update(real_splits=str(output.relative_to(project)),
                      run_name=f"study_v3_{tag}",allow_unreviewed=False,precision="float32")
        with config_path.open("x") as f:
            json.dump(config,f,indent=2)
            f.write("\n")
        report = dict(status="SPLITS_OK",created_utc=stamp,inventory=str(inv),
                      inventory_sha256=digest(master),script_sha256=digest(Path(__file__)),
                      previous_config_sha256=digest(old_config),previous_split_files=old_sources,
                      generated_split_files=generated,config_path=str(config_path),
                      config_sha256=digest(config_path),folds=audit,
                      eligible_patches=len(rows),single_class_unique=sum(int(r["n_present_classes"])==1 for r in rows),
                      training_exclusion_rows=len(excluded),partition_unit="physical_sample",
                      tube_independence_established=False,
                      legacy_fields_not_in_inventory="blank, except derived domain/image_type/source_folder/dataset_sample_id/preparation_status/split",
                      class_filter="training only; fewer than two present classes excluded",
                      sampling="natural patch counts; no balancing or duplication",
                      test_coverage="each eligible patch once across folds")
        (output / "split_provenance.json").write_text(json.dumps(report,indent=2)+"\n")
    except Exception:
        (output / "FAILED_DO_NOT_USE.txt").write_text("Preparation failed; do not train from these partial outputs.\n")
        raise
    print(f"Split directory: {output}")
    print(f"New configuration: {config_path}")
    for s in summary:
        print(f"{s['fold']} {s['split']:10s} n={s['count']:5d} excluded={s['excluded_single_class']} groups={s['samples']}")
    print(f"Eligible patches: {len(rows)}; unique single-class patches: {report['single_class_unique']}")
    print(f"Training exclusion rows across folds: {len(excluded)}")
    print("Status: SPLITS_OK. No training jobs submitted.")

if __name__ == "__main__":
    main()
