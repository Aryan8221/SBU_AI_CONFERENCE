#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm


# ------------------------------- Config ---------------------------------------
VALID_IMG_EXTS = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}
# labels typically are .seg.nrrd from 3D Slicer, but we also allow plain nrrd/nii
VALID_LABEL_HINT = {".seg.nrrd", ".nrrd", ".nii", ".nii.gz"}

CT_TOKENS  = ("CT", "_CT")
PET_TOKENS = ("PT", "PET", "_PT", "_PET")


# ------------------------------ Path helpers ----------------------------------
def stem_nii(p: Path) -> str:
    """Path.stem that handles .nii.gz nicely."""
    if "".join(p.suffixes[-2:]).lower() == ".nii.gz":
        return p.name[:-7]
    return p.stem


def is_file_with_ext(p: Path, allowed_exts: set[str]) -> bool:
    if not p.is_file():
        return False
    name = p.name.lower()
    if ".nii.gz" in name and (".nii.gz" in allowed_exts or ".nii" in allowed_exts):
        return True
    return any(name.endswith(ext) for ext in allowed_exts)


def is_nifti_like(p: Path) -> bool:
    return is_file_with_ext(p, VALID_IMG_EXTS)


def is_label_file(p: Path) -> bool:
    # Accept explicit .seg.nrrd first
    if p.is_file() and p.name.lower().endswith(".seg.nrrd"):
        return True
    # Otherwise allow general label containers (nrrd/nii)
    return is_file_with_ext(p, VALID_LABEL_HINT)


def normalize_case_key_for_images(p: Path) -> str:
    """
    Build a pairing key for CT/PET:
    - drop suffixes
    - remove CT/PT/PET tokens
    """
    name = stem_nii(p)
    for tok in CT_TOKENS + PET_TOKENS:
        name = name.replace(tok, "")
    return name


def normalize_case_key_for_labels(p: Path) -> str:
    """
    Build a pairing key for labels:
    - drop suffixes
    - remove CT/PT/PET tokens if present
    - strip trailing _S# / -S# so key matches the image key
    """
    name = stem_nii(p)
    for tok in CT_TOKENS + PET_TOKENS:
        name = name.replace(tok, "")
    name = re.sub(r"([_-]?S\d+)$", "", name, flags=re.IGNORECASE)
    return name


# --------------------------- Geometry / resampling ----------------------------
def sitk_to_nib_affine(img: sitk.Image) -> np.ndarray:
    """
    Construct a NIfTI-style RAS affine from SimpleITK image:
    SimpleITK uses LPS; NIfTI typically uses RAS.
    """
    spacing = np.array(list(img.GetSpacing()), dtype=float)  # (sx, sy, sz)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)  # LPS
    origin = np.array(list(img.GetOrigin()), dtype=float)  # LPS

    # Convert LPS -> RAS by flipping first two axes
    lps_to_ras = np.diag([-1, -1, 1])
    R = lps_to_ras @ direction @ np.diag(spacing)
    t = lps_to_ras @ origin.reshape(3,)

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = R
    affine[:3, 3] = t
    return affine


def resample_to_reference(image: sitk.Image, reference_image: sitk.Image, interpolator) -> np.ndarray:
    """
    Resample 'image' onto 'reference_image' geometry, return numpy array in (x,y,z) nib order.
    """
    resampled = sitk.Resample(
        image,
        reference_image,
        sitk.Transform(),
        interpolator,
        0.0,
        image.GetPixelID()
    )
    arr_zyx = sitk.GetArrayFromImage(resampled)  # (z, y, x)
    return np.transpose(arr_zyx, (2, 1, 0))      # -> (x, y, z)


def save_nifti_like(data_xyz: np.ndarray, ref_nib: nib.Nifti1Image, out_path: Path, dtype=None):
    if dtype is not None:
        data_xyz = data_xyz.astype(dtype, copy=False)
    img = nib.Nifti1Image(data_xyz, ref_nib.affine, ref_nib.header)
    nib.save(img, str(out_path))


# ------------------------------ Discovery -------------------------------------
def discover(in_dir: Path):
    """
    Returns:
      pairs: list of (case_key, ct_path, pet_path)
      labels_map: dict case_key -> list[Path]
    Works whether files are mixed together or under CT/ and PET/ subfolders.
    """
    root = Path(in_dir)
    all_files = [p for p in root.rglob("*") if p.is_file()]

    # Prefer explicit CT/ PET/ subdirs if present
    sub_ct = root / "CT"
    sub_pet = root / "PET"

    if sub_ct.is_dir() and sub_pet.is_dir():
        ct_files = [p for p in sub_ct.rglob("*") if is_nifti_like(p)]
        pet_files = [p for p in sub_pet.rglob("*") if is_nifti_like(p)]
    else:
        ct_files = [p for p in all_files if is_nifti_like(p) and any(tok in p.name for tok in CT_TOKENS)]
        pet_files = [p for p in all_files if is_nifti_like(p) and any(tok in p.name for tok in PET_TOKENS)]

    # Labels anywhere
    label_files = [p for p in all_files if is_label_file(p)]

    # Build maps
    pet_map = {normalize_case_key_for_images(p): p for p in pet_files}
    pairs = []
    for ct in ct_files:
        key = normalize_case_key_for_images(ct)
        if key not in pet_map:
            raise RuntimeError(f"No PET match for CT: {ct.name} (key='{key}')")
        pairs.append((key, ct, pet_map[key]))

    labels_map = {}
    for lp in label_files:
        key = normalize_case_key_for_labels(lp)
        labels_map.setdefault(key, []).append(lp)

    return pairs, labels_map


def find_pet_reference(pairs):
    """
    Choose PET with smallest (x,y,z) as reference.
    Returns (ref_path, ref_sitk, ref_nib) where ref_nib is a NIfTI with a valid affine/header.
    """
    if not pairs:
        raise RuntimeError("No CT/PET pairs found.")
    sizes = []
    for key, _, pet_p in pairs:
        img = sitk.ReadImage(str(pet_p))
        sz = img.GetSize()  # (x,y,z)
        sizes.append((sz, pet_p, img))

    sz_min, ref_path, ref_sitk = min(sizes, key=lambda t: t[0])

    # Try to use nib from PET if NIfTI, otherwise synthesize affine from SITK
    pet_path = Path(ref_path)
    if "".join(pet_path.suffixes[-2:]).lower() == ".nii.gz" or pet_path.suffix.lower() == ".nii":
        ref_nib = nib.load(str(pet_path))
    else:
        affine = sitk_to_nib_affine(ref_sitk)
        ref_nib = nib.Nifti1Image(np.zeros(sz_min[::-1], dtype=np.float32), affine)

    return str(ref_path), ref_sitk, ref_nib


# ------------------------------ Labels utils ----------------------------------
def read_label_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def resample_label_to_ref(label_img: sitk.Image, ref_sitk: sitk.Image) -> np.ndarray:
    return resample_to_reference(label_img, ref_sitk, interpolator=sitk.sitkNearestNeighbor)


def merge_labels(label_arrays_xyz: dict[str, np.ndarray]) -> np.ndarray:
    """
    Merge single-class masks { 'S1':mask, ... } -> single labelmap:
    S1->1, S2->2, ... Overlaps resolved by later segments winning (sorted by name).
    """
    if not label_arrays_xyz:
        return None
    any_arr = next(iter(label_arrays_xyz.values()))
    merged = np.zeros_like(any_arr, dtype=np.uint16)
    for seg_name, arr in sorted(label_arrays_xyz.items(), key=lambda t: t[0]):  # S1..S11
        m = re.search(r"S(\d+)", seg_name, flags=re.IGNORECASE)
        sid = int(m.group(1)) if m else 1
        merged[arr.astype(bool)] = sid
    return merged


def extract_seg_name(p: Path) -> str:
    m = re.search(r"(S\d+)", p.name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "S1"


# --------------------------------- Main ---------------------------------------
def preprocess(in_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    (out_dir / "CT").mkdir(parents=True, exist_ok=True)
    (out_dir / "PET").mkdir(parents=True, exist_ok=True)
    (out_dir / "Labels").mkdir(parents=True, exist_ok=True)

    pairs, labels_map = discover(Path(in_dir))
    pet_ref_path, pet_ref_sitk, pet_ref_nib = find_pet_reference(pairs)
    print(f"[INFO] PET_REFERENCE: {pet_ref_path} | size={pet_ref_sitk.GetSize()} spacing={pet_ref_sitk.GetSpacing()}")

    ref_xyz = tuple(pet_ref_sitk.GetSize())

    for case_key, ct_path, pet_path in tqdm(pairs, desc="Resampling"):
        # 1) Load
        ct_sitk  = sitk.ReadImage(str(ct_path))
        pet_sitk = sitk.ReadImage(str(pet_path))

        # 2) Resample to PET ref
        ct_xyz  = resample_to_reference(ct_sitk,  pet_ref_sitk, interpolator=sitk.sitkBSpline)
        pet_xyz = resample_to_reference(pet_sitk, pet_ref_sitk, interpolator=sitk.sitkBSpline)

        # 3) Optional window/clipping (customize)
        ct_xyz  = np.clip(ct_xyz,  -200, 1000)
        pet_xyz = np.clip(pet_xyz,    0,  100)

        # 4) Save CT/PET
        if ct_xyz.shape != ref_xyz or pet_xyz.shape != ref_xyz:
            print(f"[WARN] Shape mismatch for {ct_path.name}/{pet_path.name}: "
                  f"CT={ct_xyz.shape}, PET={pet_xyz.shape}, REF={ref_xyz}")
            continue

        ct_out  = out_dir / "CT"  / f"{stem_nii(ct_path)}.nii.gz"
        pet_out = out_dir / "PET" / f"{stem_nii(pet_path)}.nii.gz"
        save_nifti_like(ct_xyz,  pet_ref_nib, ct_out)
        save_nifti_like(pet_xyz, pet_ref_nib, pet_out)

        # 5) Labels for this case (if any)
        case_labels = labels_map.get(case_key, [])
        if not case_labels:
            continue

        per_label_arrays = {}
        for lp in sorted(case_labels):
            seg_name = extract_seg_name(lp)
            lab_img  = read_label_sitk(lp)
            lab_xyz  = resample_label_to_ref(lab_img, pet_ref_sitk).astype(np.uint16, copy=False)

            # Save per-segment
            lab_out = out_dir / "Labels" / f"{case_key}_{seg_name}.nii.gz"
            save_nifti_like(lab_xyz, pet_ref_nib, lab_out, dtype=np.uint16)
            per_label_arrays[seg_name] = lab_xyz

        # Also save merged labelmap
        merged = merge_labels(per_label_arrays)
        if merged is not None:
            merged_out = out_dir / "Labels" / f"{case_key}_ALL.nii.gz"
            save_nifti_like(merged, pet_ref_nib, merged_out, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT/PET and label volumes; resample to PET reference.")
    parser.add_argument("--in_dir",  required=True, help="Input directory containing CT/PET (and labels). Mixed or CT/, PET/.")
    parser.add_argument("--out_dir", required=True, help="Output directory. Creates CT/, PET/, Labels/ inside.")
    args = parser.parse_args()
    preprocess(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
