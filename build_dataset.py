import os
import numpy as np
from skimage import io, transform, measure
from albumentations import Compose, RandomBrightnessContrast, RGBShift, Rotate
import albumentations as A
import shutil
from tqdm import tqdm

# Define augmentation transformations
def get_augmentations():
    return Compose([
        Rotate(limit=45, p=0.5)
    ])

# Helper functions
def get_hor(segment):
    """Calculate Homogeneity of Region as the proportion of the majority class."""
    segment = segment.flatten().astype(np.uint8)
    if len(segment) == 0:
        return 0
    bincount = np.bincount(segment)
    majority_count = np.max(bincount)
    total_count = len(segment)
    HoR = majority_count / total_count if total_count > 0 else 0
    return HoR

# Class mapping for multiclass deforestation detection
CLASS_NAMES = {
    0: "forest",
    1: "desmatamento_cr",        # Desmatamento corte raso
    2: "desmatamento_veg",       # Desmatamento com vegetação
    3: "mineracao",              # Mineração
    4: "degradacao",             # Degradação
    5: "cicatriz_de_queimada",   # Cicatriz de incêndio florestal
    6: "cs_desordenado",         # Corte seletivo Desordenado
    7: "cs_geometrico",          # Corte seletivo Geométrico
    8: "nao_definido",           # Non-defined deforestation
}

def get_major_class(segment):
    """Return the class name of the majority class in the segment."""
    segment = segment.flatten().astype(np.uint8)
    majority_class = np.argmax(np.bincount(segment))
    return CLASS_NAMES.get(majority_class, "unknown")

def correct_band_indexing(bands):
    return [ch-1 for ch in bands]

def prepare_and_save_segments(bands):
    # Load files
    corrected_bands = correct_band_indexing(bands)
    scenes = [np.load(os.path.join(scenes_dir, f))[:, :, corrected_bands] for f in sorted(os.listdir(scenes_dir)) if 'x05' not in f]
    gts = [np.load(os.path.join(gt_dir, f)).squeeze() for f in sorted(os.listdir(gt_dir)) if 'x05' not in f]
    segmentations = [load_superpixels(os.path.join(seg_dir, f)) for f in sorted(os.listdir(seg_dir)) if 'x05' not in f and os.path.isfile(os.path.join(seg_dir, f))]
    region_ids = [f"x{i + 1 :02d}" for i in range(len(scenes) + 1) if i != 4]

    os.makedirs(output_dir, exist_ok=True)

    resizes = dict()

    for scene, gt, seg, region_id in zip(scenes, gts, segmentations, region_ids):
        rsz = 0
        unique_superpixels = np.unique(seg)

        # Create folders for each region
        region_folder = os.path.join(output_dir, region_id)
        os.makedirs(os.path.join(region_folder, "original"), exist_ok=True)
        os.makedirs(os.path.join(region_folder, "augmented"), exist_ok=True)

        for superpixel_id in tqdm(unique_superpixels, desc=f"Processing {region_id}"):
            # Create mask for the superpixel
            mask = seg == superpixel_id

            # Get bounding box for the superpixel
            region = measure.regionprops(mask.astype(np.uint8))[0]
            minr, minc, maxr, maxc = region.bbox

            # Crop the image and ground truth
            cropped_image = scene[minr:maxr, minc:maxc] * mask[minr:maxr, minc:maxc, np.newaxis]
            cropped_gt = gt[minr:maxr, minc:maxc] * mask[minr:maxr, minc:maxc]

            # Check if HoR is sufficient
            if get_hor(cropped_gt) < 0.7 or cropped_image.size < 70:
                continue

            # Skip unknown class segments
            if get_major_class(cropped_gt) == "unknown":
                continue

            # Put segment into a [segment_size] shaped zero-padded image
            resized_image = np.zeros((*segment_size, cropped_image.shape[-1]), dtype=np.uint8)
            resized_gt = np.zeros(segment_size, dtype=np.uint8)

            # Insert image and ground truth into the center of the zero-padded image (check if segment dimensions are smaller than segment_size)
            if cropped_image.shape[0] < segment_size[0] and cropped_image.shape[1] < segment_size[1]:
                offset = (np.array(segment_size) - np.array(cropped_gt.shape)) // 2
                resized_image[offset[0]:offset[0] + cropped_image.shape[0], offset[1]:offset[1] + cropped_image.shape[1]] = cropped_image
                resized_gt[offset[0]:offset[0] + cropped_gt.shape[0], offset[1]:offset[1] + cropped_gt.shape[1]] = cropped_gt
            else:
                rsz += 1
                resized_image = transform.resize(cropped_image, segment_size, anti_aliasing=True)
                resized_gt = transform.resize(cropped_gt, segment_size, anti_aliasing=False, order=0)

            # Determine class
            segment_class = get_major_class(cropped_gt)

            # print(f"unique segment: {np.unique(resized_image)}")
            # print(f"unique gt: {np.unique(resized_gt)}")

            if resized_image.max() > 1:
                resized_image = resized_image.astype(np.uint8)
            else:
                resized_image = (resized_image * 255).astype(np.uint8)

            # Save original image
            original_path = os.path.join(region_folder, "original", f"{superpixel_id}_{segment_class}.png")
            io.imsave(original_path, resized_image, check_contrast=False)

            # Perform augmentations
            for i in range(num_augmentations):
                augmented_image = get_augmentations()(image=resized_image)["image"]
                augmented_path = os.path.join(region_folder, "augmented", f"{superpixel_id}_{segment_class}_aug_{i}.png")
                io.imsave(augmented_path, augmented_image, check_contrast=False)

        resizes[region_id] = rsz

    with open(os.path.join(output_dir, "resizes.txt"), "w") as f:
        for region_id, rsz in resizes.items():
            f.write(f"{region_id}: {rsz}\n")

def load_superpixels(seg_path):
    if seg_path.endswith(".npy"):
        return np.load(seg_path)
    elif seg_path.endswith((".png", ".pgm")):
        return io.imread(seg_path)
    else:
        raise ValueError("Unsupported file format")

if __name__ == "__main__":
    # Run preparation and saving
    # Parameters
    scenes_dir = "/home/ebneto/foresteyes/scenes_sentinel2/"
    gt_dir = "/home/ebneto/foresteyes/truth_masks_sentinel2-DETERMulticlass/deter_multiclass_truth/"
    seg_dir = "/home/ebneto/bandselection/slics_sentinel_pca/"
    output_dir = "dml_dataset_sentinelGECCO-256sq"
    segment_size = (256, 256)
    num_augmentations = 5  # Set number of augmented images per segment
    # Digit_2: 13 (65.00%)
    # Digit_8: 7 (35.00%)
    # Digit_6: 5 (25.00%)
    bands = [2, 8, 6]
    prepare_and_save_segments(bands)
