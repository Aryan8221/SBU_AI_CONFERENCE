#!/usr/bin/env python3
import argparse, os, json, re
from pathlib import Path

# Filenames:
#   CT/CT10-res.nii.gz
#   PT/PT10-res.nii.gz
#   Mask/Seg10.seg_aligned.nii.gz
#   Mask/Seg1_merged.seg_aligned.nii.gz

IDX_CT_RE   = re.compile(r"^CT(\d+)-res\.nii\.gz$", re.IGNORECASE)
IDX_PT_RE   = re.compile(r"^PT(\d+)-res\.nii\.gz$", re.IGNORECASE)
IDX_LBL1_RE = re.compile(r"^(\d+)_seg\.nii\.gz$", re.IGNORECASE)  # LABELS/{idx}_seg.nii.gz
IDX_LBL2_RE = re.compile(r"^Seg(\d+)\.nii$", re.IGNORECASE)

# Hard-coded center ranges:
CENTER1_RANGE = range(1, 12)   # 1..11
CENTER2_RANGE = range(12, 25)  # 12..24


def index_from_name(name: str, kind: str):
    if kind == "ct":
        m = IDX_CT_RE.match(name)
    elif kind == "pt":
        m = IDX_PT_RE.match(name)
    elif kind == "label1":
        m = IDX_LBL1_RE.match(name)
    elif kind == "label2":
        m = IDX_LBL2_RE.match(name)
    else:
        return None
    return int(m.group(1)) if m else None


def scan_modalities(root: Path):
    root = Path(root)
    ct_dir  = root / "CT"
    pet_dir = root / "PT"

    print(f"[INFO] CT dir:  {ct_dir}")
    print(f"[INFO] PT dir: {pet_dir}")

    if not ct_dir.is_dir() or not pet_dir.is_dir():
        raise SystemExit("Expected subfolders CT/ and PT/ under --path")

    # labels in LABELS/ (idx_seg.nii.gz) or Mask/ (Seg{idx}[_merged].seg_aligned.nii.gz)
    if (root / "LABELS").is_dir():
        labels_dir = root / "LABELS"
        label_mode = "labels"
        print("[INFO] Using LABELS/ for ground truth.")
    elif (root / "Mask").is_dir():
        labels_dir = root / "Mask"
        label_mode = "mask"
        print("[INFO] Using Mask/ for ground truth.")
    else:
        raise SystemExit("Expected LABELS/ or Mask/ folder under --path")

    ct_map, pt_map, lbl_map = {}, {}, {}

    # CT
    for f in sorted(os.listdir(ct_dir)):
        idx = index_from_name(f, "ct")
        if idx is not None:
            ct_map[idx] = f

    # PT
    for f in sorted(os.listdir(pet_dir)):
        idx = index_from_name(f, "pt")
        if idx is not None:
            pt_map[idx] = f

    # LABELS / Mask
    for f in sorted(os.listdir(labels_dir)):
        idx = index_from_name(f, "label1" if label_mode == "labels" else "label2")
        if idx is not None:
            lbl_map[idx] = f

    return ct_dir, pet_dir, labels_dir, ct_map, pt_map, lbl_map


def make_entry(idx: int,
               ct_dir: Path, pet_dir: Path, lbl_dir: Path,
               ct_map: dict, pt_map: dict, lbl_map: dict):
    # ensure files for this index
    if idx not in ct_map or idx not in pt_map or idx not in lbl_map:
        missing = []
        if idx not in ct_map:
            missing.append("CT")
        if idx not in pt_map:
            missing.append("PT")
        if idx not in lbl_map:
            missing.append("LABEL")
        raise ValueError(f"Index {idx} missing: {', '.join(missing)}")

    ct_path  = (ct_dir  / ct_map[idx]).resolve()
    pt_path  = (pet_dir / pt_map[idx]).resolve()
    lbl_path = (lbl_dir / lbl_map[idx]).resolve()

    return {
        "image": [str(ct_path), str(pt_path)],   # absolute paths
        "label": str(lbl_path),                  # absolute path
        "index": idx
    }


def write_loco_two_centers(
    center1_indices,
    center2_indices,
    ct_dir: Path, pet_dir: Path, lbl_dir: Path,
    ct_map: dict, pt_map: dict, lbl_map: dict,
    out_dir: Path
):
    """
    Leave-one-center-out with 2 centers.
    - fold0.json: train on center1, val on center2
    - fold1.json: train on center2, val on center1
    """
    out_dir = Path(out_dir)
    center1_indices = list(sorted(center1_indices))
    center2_indices = list(sorted(center2_indices))

    print(f"[INFO] Center1 indices: {center1_indices}")
    print(f"[INFO] Center2 indices: {center2_indices}")

    folds = [
        ("fold0.json", center1_indices, center2_indices),  # train C1, val C2
        ("fold1.json", center2_indices, center1_indices),  # train C2, val C1
    ]

    for fold_name, train_idxs, val_idxs in folds:
        training, validation = [], []

        # training set
        for i in train_idxs:
            try:
                training.append(
                    make_entry(i, ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map)
                )
            except ValueError as e:
                print(f"[WARN] Skipping train idx {i}: {e}")

        # validation set
        for j in val_idxs:
            try:
                validation.append(
                    make_entry(j, ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map)
                )
            except ValueError as e:
                print(f"[WARN] Skipping val idx {j}: {e}")

        data = {"training": training, "validation": validation}

        out_path = (out_dir / fold_name).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"[OK] Wrote {out_path}: "
            f"train={len(training)} cases, val={len(validation)} cases"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate LOCO (leave-one-center-out) folds for 2 centers with ABSOLUTE paths."
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Dataset root containing CT/, PT/, and LABELS/ or Mask/",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Folder to write JSON folds",
    )
    args = parser.parse_args()

    ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map = scan_modalities(args.path)

    all_indices = sorted(set(ct_map) & set(pt_map) & set(lbl_map))
    if not all_indices:
        raise SystemExit("No common indices found across CT, PT, and LABELS/Mask.")

    print(f"[INFO] Found {len(all_indices)} indices with complete CT/PT/LABEL.")

    # Intersect with our assumed center ranges
    center1_indices = sorted(set(CENTER1_RANGE) & set(all_indices))
    center2_indices = sorted(set(CENTER2_RANGE) & set(all_indices))

    if not center1_indices:
        print("[WARN] No complete cases found for Center 1 (1–11).")
    if not center2_indices:
        print("[WARN] No complete cases found for Center 2 (12–24).")

    if not center1_indices or not center2_indices:
        raise SystemExit("Need at least one case in BOTH centers for LOCO.")

    write_loco_two_centers(
        center1_indices,
        center2_indices,
        ct_dir, pet_dir, lbl_dir,
        ct_map, pt_map, lbl_map,
        args.out,
    )

    print("[DONE] LOCO JSONs written to:", Path(args.out).resolve())


if __name__ == "__main__":
    main()
