from pathlib import Path
import numpy as np
import scipy.io
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters.rank import modal
from skimage.morphology import disk
from skimage.segmentation import find_boundaries

def load_mat_array(path):
    """Loads a .mat file and returns its array data."""
    data = scipy.io.loadmat(path)
    real_keys = [k for k in data if not k.startswith("__")]

    print(f"Keys found in {path}: {real_keys}")
    for k in real_keys:
        print(f"  '{k}': shape={data[k].shape}, dtype={data[k].dtype}")

    if len(real_keys) == 1:
        return data[real_keys[0]]

    # More than one variable -- change this to the key you actually want.
    chosen_key = real_keys[0]
    print(f"Multiple keys found, using '{chosen_key}' -- edit load_mat_array() to change this.")
    return data[chosen_key]


def resize_mask_to(mask, target_shape):
    """Aligns mask onto target_shape (most likely case here: transpose)."""
    if mask.shape == target_shape:
        return mask
 
    if mask.shape == target_shape[::-1]:
        return mask.T
 
    print(f"Mask shape {mask.shape} does not match target {target_shape} -- "
          f"resizing with nearest-neighbor interpolation.")
    from scipy.ndimage import zoom
    zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
    return zoom(mask, zoom_factors, order=0)


def handle_nan_labels(mask):
    """Replaces NaN pixels with an explicit sentinel label."""
    nan_mask = np.isnan(mask)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        sentinel = np.nanmax(mask) + 1 if np.isfinite(np.nanmax(mask)) else -1
        print(f"Found {n_nan} NaN pixels ({100 * n_nan / mask.size:.1f}%) -- "
              f"replacing with sentinel label {sentinel}.")
        mask = np.where(nan_mask, sentinel, mask)
    return mask


def denoise_labels(mask, radius=3):
    """Majority/mode filter over a disk neighborhood to clean speckle noise."""
    unique_vals, remapped = np.unique(mask, return_inverse=True)
    remapped = remapped.reshape(mask.shape).astype(np.uint16)
    denoised_remapped = modal(remapped, disk(radius))
    return unique_vals[denoised_remapped]


def remove_small_regions(mask, min_size=50):
    """
    Removes small "inclusions" by merging each one into whichever real label surrounds it.
    Used to "clean up" from the denoise labels, as it don't pick up on bigger clusters.
    """
    mask = mask.copy()
    struct = ndi.generate_binary_structure(2, 2)  # 8-connectivity
 
    while True:
        removed_any = False
        for val in np.unique(mask):
            binary = (mask == val)
            components, n_components = ndi.label(binary, structure=struct)
            if n_components == 0:
                continue
            sizes = ndi.sum(binary, components, index=range(1, n_components + 1))
 
            for comp_id, size in enumerate(sizes, start=1):
                if size >= min_size:
                    continue
                island = components == comp_id
                # look at the immediate neighbors just outside the island
                dilated = ndi.binary_dilation(island, structure=struct) & ~island
                neighbor_vals = mask[dilated]
                neighbor_vals = neighbor_vals[neighbor_vals != val]
                if neighbor_vals.size == 0:
                    continue  # isolated with no valid neighbor (shouldn't normally happen)
                vals, counts = np.unique(neighbor_vals, return_counts=True)
                new_val = vals[np.argmax(counts)]
                mask[island] = new_val
                removed_any = True
 
        if not removed_any:
            break
 
    return mask


def mask_to_binary_boundary(mask):
    """Returns a strictly binary image: 1.0 (white) = background, 0.0 (black) = boundary. Using a basic scikit-image function."""
    boundaries = find_boundaries(mask, mode="outer")
    binary_img = np.ones(mask.shape, dtype=float)
    binary_img[boundaries] = 0.0
    return binary_img


def find_mask_files(mask_root, suffix=".mat"):
    """Recursively find mask files (default: .mat) under mask_root."""
    return sorted(Path(mask_root).rglob(f"*{suffix}"))
 
 
def find_modality_files(processed_dir, iq_pattern="iq_sdan", adp_pattern="adp_1Dsig"):
    """Finds the iq/adp .npy files inside one x*_processed folder by substring match."""
    iq_path, adp_path = None, None
    for f in Path(processed_dir).glob("*.npy"):
        if iq_pattern in f.name:
            iq_path = f
        elif adp_pattern in f.name:
            adp_path = f
    return iq_path, adp_path


def build_pairs(mask_root, image_root, mask_suffix=".mat", verbose=True):
    """
    Pairs every mask file under mask_root with its iq/adp .npy files under
    image_root, matched by <sample>/<location> folder path (NOT filename --
    the mask and the images never share a filename).
 
    Two mask layouts are supported, detected per-file by path depth:
      - PGs/<sample>/<location>/<mask_file>              (no magnification
        subfolder -- the mask is paired with EVERY x*_processed folder
        found under that location, same as before)
      - PGs/<sample>/<location>/<magnification>/<mask_file>  (an explicit
        magnification subfolder, e.g. "x100" -- the mask is paired ONLY
        with the matching SDA/<sample>/<location>/<magnification>_processed
        folder, never with other magnifications at that location)
 
    Returns a list of dicts:
        {"sample", "location", "magnification", "mask_path", "iq_path", "adp_path"}
    """
    mask_root = Path(mask_root)
    pairs = []
    unmatched = []
 
    for mask_path in find_mask_files(mask_root, mask_suffix):
        rel = mask_path.relative_to(mask_root)
 
        if len(rel.parts) == 3:
            sample, location = rel.parts[0], rel.parts[1]
            required_magnification = None
        elif len(rel.parts) == 4:
            sample, location, required_magnification = rel.parts[0], rel.parts[1], rel.parts[2]
        else:
            unmatched.append(str(mask_path))
            continue
 
        loc_dir = Path(image_root) / sample / location
        if not loc_dir.exists():
            unmatched.append(str(mask_path))
            continue
 
        found_any = False
        for processed_dir in sorted(loc_dir.glob("x*_processed")):
            magnification = processed_dir.name.replace("_processed", "")
 
            # If this mask lives under an explicit magnification subfolder,
            # only pair it with that same magnification's images.
            if required_magnification is not None and magnification != required_magnification:
                continue
 
            iq_path, adp_path = find_modality_files(processed_dir)
            if iq_path is None or adp_path is None:
                continue
 
            pairs.append({
                "sample": sample, "location": location, "magnification": magnification,
                "mask_path": mask_path, "iq_path": iq_path, "adp_path": adp_path,
            })
            found_any = True
 
        if not found_any:
            unmatched.append(str(mask_path))
 
    if verbose:
        print(f"Built {len(pairs)} mask/iq/adp pair(s).")
        if unmatched:
            print(f"Warning: {len(unmatched)} mask file(s) had no matching "
                  f"SDA images and were skipped: {unmatched}")
 
    return pairs


def crop_non_overlapping(array, tile_size=128):
    """
    Splits array into non-overlapping tile_size x tile_size tiles, starting
    from (0, 0). Any leftover strip on the right/bottom that doesn't fill a
    full tile is dropped, not padded. Returns a list of (index, tile), with
    index starting at 1 and increasing in row-major (top-to-bottom,
    left-to-right) order.
    """
    h, w = array.shape[:2]
    n_rows = h // tile_size
    n_cols = w // tile_size
 
    tiles = []
    idx = 1
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size
            tiles.append((idx, array[y0:y1, x0:x1]))
            idx += 1
    return tiles


def save_float01_as_png(array, path):
    """Saves a float array (assumed roughly [0, 1], or binary 0/1) as an 8-bit PNG."""
    arr = np.clip(array, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)
 
 
def normalize_for_display(array):
    """Per-image min-max scaling to [0, 1], for saving IQ/ADP as viewable PNGs, since the arrays are raw floats."""
    lo, hi = np.nanmin(array), np.nanmax(array)
    return (array - lo) / (hi - lo + 1e-6)