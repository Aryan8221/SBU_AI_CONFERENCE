#!/usr/bin/env python3
import argparse, os, json, re
from pathlib import Path

# Filenames:
#   PT/PT10-res.nii.gz
#   LABELS/10_seg.nii.gz
#   Mask/Seg10.nii

IDX_PT_RE   = re.compile(r"^PT(\d+)-res\.nii(?:\.gz)?$", re.IGNORECASE)
IDX_LBL1_RE = re.compile(r"^(\d+)_seg\.nii\.gz$", re.IGNORECASE)      # LABELS/
IDX_LBL2_RE = re.compile(r"^Seg(\d+)\.nii$", re.IGNORECASE)          # Mask/


def index_from_name(name: str, kind: str):
    kind = kind.lower()
    if kind == "pt":
        m = IDX_PT_RE.match(name)
    elif kind == "label1":
        m = IDX_LBL1_RE.match(name)
    elif kind == "label2":
        m = IDX_LBL2_RE.match(name)
    else:
        return None
    return int(m.group(1)) if m else None


def scan_modalities(root: Path):
    print(f"root: {Path(root) / 'PT'}")
    pet_dir = Path(root) / "PT"
    print(f"pet_dir.is_dir(): {pet_dir.is_dir()}")
    print(f"pet_dir TYPE: {type(pet_dir)}")
    if not pet_dir.is_dir():
        raise SystemExit("Expected subfolder PT/ under --path")

    # labels in LABELS/ (idx_seg.nii.gz) or Mask/ (Seg{idx}.nii)
    if (Path(root) / "LABELS").is_dir():
        labels_dir = Path(root) / "LABELS"
        label_mode = "labels"
    elif (Path(root) / "Mask").is_dir():
        labels_dir = Path(root) / "Mask"
        label_mode = "mask"
    else:
        raise SystemExit("Expected LABELS/ or Mask/ folder under --path")

    pt_map, lbl_map = {}, {}

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

    return pet_dir, labels_dir, pt_map, lbl_map


def make_entry(idx: int, pet_dir: Path, lbl_dir: Path, pt_map: dict, lbl_map: dict):
    if idx not in pt_map or idx not in lbl_map:
        missing = []
        if idx not in pt_map:
            missing.append("PT")
        if idx not in lbl_map:
            missing.append("LABEL")
        raise ValueError(f"Index {idx} missing: {', '.join(missing)}")

    pt_path  = (pet_dir / pt_map[idx]).resolve()
    lbl_path = (lbl_dir / lbl_map[idx]).resolve()

    return {
        "image": str(pt_path),   # PET only (absolute path)
        "label": str(lbl_path),  # absolute path
        "index": idx
    }


def write_loocv_all(indices, pet_dir: Path, lbl_dir: Path, pt_map: dict, lbl_map: dict, out_dir: Path):
    """
    Single LOOCV over all indices in `indices`.
    Output: fold0.json, fold1.json, fold2.json, ...
    """
    indices = list(indices)

    for fold_id, val_idx in enumerate(indices):
        train_idxs = [i for i in indices if i != val_idx]

        training, validation = [], []

        for i in train_idxs:
            try:
                training.append(make_entry(i, pet_dir, lbl_dir, pt_map, lbl_map))
            except ValueError as e:
                print(f"[WARN] Skipping train idx {i}: {e}")

        try:
            validation.append(make_entry(val_idx, pet_dir, lbl_dir, pt_map, lbl_map))
        except ValueError as e:
            print(f"[WARN] Skipping val idx {val_idx}: {e}")

        data = {"training": training, "validation": validation}

        out_path = (Path(out_dir) / f"fold{fold_id}.json").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Wrote {out_path}: fold_id={fold_id}, val_idx={val_idx}, train={len(training)} val={len(validation)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-center LOOCV folds with ABSOLUTE paths (PET-only input)."
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Dataset root containing PT/ and LABELS/ or Mask/",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Folder to write JSON folds",
    )
    args = parser.parse_args()

    pet_dir, lbl_dir, pt_map, lbl_map = scan_modalities(args.path)

    print(f"PT map indices: {sorted(pt_map.keys())}")
    print(f"Label map indices: {sorted(lbl_map.keys())}")

    all_indices = sorted(set(pt_map) & set(lbl_map))
    if not all_indices:
        raise SystemExit("No common indices found across PT and LABELS/Mask.")

    print(f"[INFO] Found {len(all_indices)} total cases: {all_indices}")

    # ----- Center splitting logic -----
    center1_indices = [i for i in all_indices if 1 <= i <= 11]
    center2_indices = [i for i in all_indices if 12 <= i <= 24]

    print(f"[INFO] Center1 indices: {center1_indices}")
    print(f"[INFO] Center2 indices: {center2_indices}")

    if center1_indices:
        out_c1 = args.out / "center1_pet_only"
        print(f"[INFO] Using {len(center1_indices)} cases for Center 1 LOOCV.")
        write_loocv_all(center1_indices, pet_dir, lbl_dir, pt_map, lbl_map, out_c1)
    else:
        print("[WARN] No indices found for Center 1. Skipping.")

    if center2_indices:
        out_c2 = args.out / "center2_pet_only"
        print(f"[INFO] Using {len(center2_indices)} cases for Center 2 LOOCV.")
        write_loocv_all(center2_indices, pet_dir, lbl_dir, pt_map, lbl_map, out_c2)
    else:
        print("[WARN] No indices found for Center 2. Skipping.")

    print("[DONE] Per-center LOOCV JSONs written under:", args.out.resolve())


if __name__ == "__main__":
    main()
