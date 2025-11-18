#!/usr/bin/env python3
import argparse, os, json, re
from pathlib import Path

# Filenames:
#   CT/CT10-res.nii.gz
#   PET/PT10-res.nii.gz
#   Mask/Seg10.seg_aligned.nii.gz
#   Mask/Seg1_merged.seg_aligned.nii.gz

IDX_CT_RE   = re.compile(r"^CT(\d+)-res\.nii\.gz$", re.IGNORECASE)
IDX_PT_RE   = re.compile(r"^PT(\d+)-res\.nii\.gz$", re.IGNORECASE)
IDX_LBL1_RE = re.compile(r"^(\d+)_seg\.nii\.gz$", re.IGNORECASE)  # for LABELS/{idx}_seg.nii.gz (if ever used)

# *** CHANGED: match Seg{idx}.seg_aligned.nii.gz and Seg{idx}_merged.seg_aligned.nii.gz ***
IDX_LBL2_RE = re.compile(r"^Seg(\d+)(?:_merged)?\.seg_aligned\.nii\.gz$", re.IGNORECASE)

CENTER1 = list(range(1, 12))      # 1..11
CENTER2 = list(range(12, 25))     # 12..24


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
    ct_dir  = root / "CT"
    pet_dir = root / "PET"
    if not ct_dir.is_dir() or not pet_dir.is_dir():
        raise SystemExit("Expected subfolders CT/ and PET/ under --path")

    # labels in LABELS/ (idx_seg.nii.gz) or Mask/ (Seg{idx}[_merged].seg_aligned.nii.gz)
    if (root / "LABELS").is_dir():
        labels_dir = root / "LABELS"
        label_mode = "labels"
    elif (root / "Mask").is_dir():
        labels_dir = root / "Mask"
        label_mode = "mask"
    else:
        raise SystemExit("Expected LABELS/ or Mask/ folder under --path")

    ct_map, pt_map, lbl_map = {}, {}, {}

    # CT
    for f in sorted(os.listdir(ct_dir)):
        idx = index_from_name(f, "ct")
        if idx is not None:
            ct_map[idx] = f

    # PET
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
            missing.append("PET")
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


def write_loocv_for_center(center_indices,
                           out_prefix: str,
                           ct_dir: Path, pet_dir: Path, lbl_dir: Path,
                           ct_map: dict, pt_map: dict, lbl_map: dict,
                           out_dir: Path):
    for val_idx in center_indices:
        train_idxs = [i for i in center_indices if i != val_idx]

        training, validation = [], []

        for i in train_idxs:
            try:
                training.append(
                    make_entry(i, ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map)
                )
            except ValueError as e:
                print(f"[WARN] Skipping train idx {i}: {e}")

        try:
            validation.append(
                make_entry(val_idx, ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map)
            )
        except ValueError as e:
            print(f"[WARN] Skipping val idx {val_idx}: {e}")

        data = {"training": training, "validation": validation}

        out_path = (out_dir / f"{out_prefix}_fold_{val_idx}.json").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Wrote {out_path}: train={len(training)} val={len(validation)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LOOCV folds per center (1–11, 12–24) with ABSOLUTE paths."
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Dataset root containing CT/, PET/, and LABELS/ or Mask/",
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
        raise SystemExit("No common indices found across CT, PET, and LABELS/Mask.")

    center1 = [i for i in range(1, 12) if i in all_indices]
    center2 = [i for i in range(12, 25) if i in all_indices]

    if center1:
        write_loocv_for_center(
            center1, "center1", ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map, args.out
        )
    else:
        print("[WARN] No complete cases in Center 1 (1–11).")

    if center2:
        write_loocv_for_center(
            center2, "center2", ct_dir, pet_dir, lbl_dir, ct_map, pt_map, lbl_map, args.out
        )
    else:
        print("[WARN] No complete cases in Center 2 (12–24).")

    print("[DONE] LOOCV JSONs written to:", args.out.resolve())


if __name__ == "__main__":
    main()
