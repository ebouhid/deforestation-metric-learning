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
        RandomBrightnessContrast(p=0.5),
        RGBShift(p=0.5),
        Rotate(limit=45, p=0.5)
    ])

# Parameters
scenes_dir = "/datasets/eduardo/foresteyes/landsat_8/allbands_8b/"
gt_dir = "/datasets/eduardo/foresteyes/truth_masks/"
seg_dir = "/datasets/eduardo/foresteyes/superpixels_pucmg/SegmPCA_original/SLIC/scenes_pca/4000"
output_dir = "segment_embeddings_classification_dataset_norsz"
segment_size = (28, 28)
num_augmentations = 5  # Set number of augmented images per segment

# Helper functions
def get_hor(segment):
    segment = segment.flatten().astype(np.uint8)
    NFP = np.count_nonzero(segment == 2)
    NP = np.count_nonzero(segment)
    NNP = NP - NFP
    HoR = max([NFP, NNP]) / NP if NP > 0 else 0
    return HoR

def get_major_class(segment):
    segment = segment.flatten().astype(np.uint8)
    majority_class = np.argmax(np.bincount(segment))
    return "forest" if majority_class == 2 else "recent-deforestation" if majority_class == 1 else "not-analyzed"

def prepare_and_save_segments():
    # Load files
    scenes = [np.load(os.path.join(scenes_dir, f))[:, :, [3, 2, 1]] for f in sorted(os.listdir(scenes_dir)) if 'x05' not in f]
    gts = [np.load(os.path.join(gt_dir, f)).squeeze() for f in sorted(os.listdir(gt_dir)) if 'x05' not in f]
    segmentations = [load_superpixels(os.path.join(seg_dir, f)) for f in sorted(os.listdir(seg_dir)) if 'x05' not in f]
    region_ids = [f"x{i + 1 :02d}" for i in range(len(scenes) + 1) if i != 4]

    os.makedirs(output_dir, exist_ok=True)

    for scene, gt, seg, region_id in zip(scenes, gts, segmentations, region_ids):
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

            # Skip not_analyzed segments
            if get_major_class(cropped_gt) == "not_analyzed":
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

def load_superpixels(seg_path):
    if seg_path.endswith(".npy"):
        return np.load(seg_path)
    elif seg_path.endswith((".png", ".pgm")):
        return io.imread(seg_path)
    else:
        raise ValueError("Unsupported file format")

# Run preparation and saving
prepare_and_save_segments()
