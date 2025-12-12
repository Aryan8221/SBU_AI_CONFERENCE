import numpy as np
import SimpleITK as sitk
import nibabel as nib
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def resample_image(image, new_spacing, reference_image):
    """
    Resamples an image to a new spacing, using the reference image to determine
    the size of the output image.
    """
    output_origin = reference_image.GetOrigin()
    output_spacing = new_spacing
    output_direction = reference_image.GetDirection()

    resampled_image = sitk.Resample(
        image,
        reference_image.GetSize(),
        sitk.Transform(),       # identity transform
        sitk.sitkBSpline,
        output_origin,
        output_spacing,
        output_direction,
    )

    resampled_array = sitk.GetArrayFromImage(resampled_image)
    # SimpleITK returns (z, y, x); nibabel expects (x, y, z)
    return np.transpose(resampled_array, (2, 1, 0))


def extract_id(fname: str, modality: str) -> str:
    """
    Extract the shared numeric ID from filename, independent of extension.
    Assumes names like:
        CT416.nii, CT416.nii.gz, PET416.nii, PET416.nii.gz, etc.
    """
    name = Path(fname).name

    if modality.upper() == "CT":
        # remove first occurrence of 'CT'
        if "CT" not in name:
            raise ValueError(f"Expected 'CT' in CT filename, got: {fname}")
        name = name.replace("CT", "", 1)
    elif modality.upper() == "PT":
        if "PT" not in name:
            raise ValueError(f"Expected 'PT' in PT filename, got: {fname}")
        name = name.replace("PT", "", 1)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # strip known extensions
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    return name  # e.g. "416"


def preprocess_nifti_images(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "PT"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "CT"), exist_ok=True)

    all_files = [
        f for f in os.listdir(in_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    pet_files = [f for f in all_files if "PT" in f]
    ct_files = [f for f in all_files if "CT" in f]

    if not pet_files or not ct_files:
        raise RuntimeError("No PT or CT files found in input directory.")

    # Build maps: id -> filename
    ct_map = {}
    for f in ct_files:
        idx = extract_id(f, "CT")
        ct_map[idx] = f

    pet_map = {}
    for f in pet_files:
        idx = extract_id(f, "PT")
        pet_map[idx] = f

    common_ids = sorted(set(ct_map.keys()) & set(pet_map.keys()))
    if not common_ids:
        raise RuntimeError("No matching CT/PT IDs found.")

    if len(common_ids) != len(ct_map) or len(common_ids) != len(pet_map):
        print("[WARN] Some CT or PT files do not have a matching pair. "
              "Only intersecting IDs will be processed.")

    for idx in tqdm(common_ids):
        ct_filename = ct_map[idx]
        pet_filename = pet_map[idx]

        ct_path = os.path.join(in_dir, ct_filename)
        pet_path = os.path.join(in_dir, pet_filename)

        # PT as nibabel for affine/header
        pet_nib = nib.load(pet_path)

        # SimpleITK for resampling & spacing
        ct_image = sitk.ReadImage(ct_path)
        pet_image = sitk.ReadImage(pet_path)

        pet_spacing = pet_image.GetSpacing()

        ct_downsampled_image = resample_image(
            ct_image, pet_spacing, pet_image
        )

        ct_hu_min, ct_hu_max = -1000, 1000
        pet_hu_min, pet_hu_max = 0, 100

        clipped_ct_data = np.clip(ct_downsampled_image, ct_hu_min, ct_hu_max)
        clipped_pet_data = np.clip(pet_nib.get_fdata(), pet_hu_min, pet_hu_max)

        # Create NIfTI with PT affine/header so CT & PT match in space
        final_ct_image = nib.Nifti1Image(
            clipped_ct_data, pet_nib.affine, pet_nib.header
        )
        final_pet_image = nib.Nifti1Image(
            clipped_pet_data, pet_nib.affine, pet_nib.header
        )

        # Make sure shapes actually match before saving
        if clipped_ct_data.shape != clipped_pet_data.shape:
            print(f"[WARN] Shape mismatch for ID {idx}: "
                  f"CT {clipped_ct_data.shape} vs PT {clipped_pet_data.shape}. Skipping.")
            continue

        ct_output_path = os.path.join(out_dir, "CT", f"CT{idx}.nii.gz")
        pet_output_path = os.path.join(out_dir, "PT", f"PT{idx}.nii.gz")

        nib.save(final_ct_image, ct_output_path)
        nib.save(final_pet_image, pet_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Nifti Images")
    parser.add_argument("--in_dir", type=str, help="Input directory containing all the images (CT and PT, mixed)")
    parser.add_argument("--out_dir", type=str, help="Directory to save the images")
    args = parser.parse_args()

    preprocess_nifti_images(args.in_dir, args.out_dir)
