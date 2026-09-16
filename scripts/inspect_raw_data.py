from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = Path("datasets/raw")
PGS_DIR = DATASET_DIR / "PGs"
SDA_DIR = DATASET_DIR / "SDA"

# Set to None to visualize the first complete matched sample.
# Example: VISUALIZE_SAMPLE = ("2", "950")
VISUALIZE_SAMPLE = None

# Figures are saved here because the cluster may not have a graphical display.
OUTPUT_DIR = Path("experiments/inspect_raw_data")


# ---------------------------------------------------------------------------
# Load PGS .mat data
# ---------------------------------------------------------------------------

def get_mat_arrays(mat_path):
    """Return all non-metadata arrays stored in a MATLAB file."""
    contents = loadmat(mat_path)

    return {
        name: np.asarray(value)
        for name, value in contents.items()
        if not name.startswith("__") and np.asarray(value).size > 0
    }


def choose_pgs_array(mat_path):
    """
    Choose the most likely previous-austenite array.

    Prefer 2D variables whose names contain parent/grain/id.
    Otherwise use the largest 2D variable.
    """
    arrays = get_mat_arrays(mat_path)

    if not arrays:
        raise ValueError(f"No data arrays found in {mat_path}")

    preferred = [
        (name, array)
        for name, array in arrays.items()
        if array.ndim == 2
        and any(word in name.lower() for word in ("parent", "grain", "id"))
    ]

    if preferred:
        return max(preferred, key=lambda item: item[1].size)

    candidates = [
        (name, array)
        for name, array in arrays.items()
        if array.ndim == 2
    ]

    if candidates:
        return max(candidates, key=lambda item: item[1].size)

    return max(arrays.items(), key=lambda item: item[1].size)


# ---------------------------------------------------------------------------
# Find corresponding SDA data
# ---------------------------------------------------------------------------

def find_iq_file(directory):
    """Find an IQ .npy file in a directory."""
    files = sorted(
        path for path in directory.glob("*.npy")
        if "iq" in path.name.lower()
    )
    return files[0] if files else None


def find_adp_file(directory):
    """Find an ADP .npy file in a directory."""
    files = sorted(
        path for path in directory.glob("*.npy")
        if "adp" in path.name.lower()
    )
    return files[0] if files else None


def find_sda_directory(pgs_path):
    """
    Find the corresponding SDA processing directory.

    Normally:
        PGS/2/950/file.mat
        -> SDA/2/950/x200_processed/

    For PGS folders with an extra processing level, e.g.
        PGS/4/midt/x100/file.mat
        PGS/4/midt/x200/file.mat

    the function also tries the shorter parent path:
        SDA/4/midt/

    This lets us inspect the structure without assuming that PGS and SDA
    have exactly the same number of folders.
    """
    relative = pgs_path.relative_to(PGS_DIR)
    pgs_parts = relative.parts[:-1]

    # First try the complete PGS folder path.
    candidate_roots = [
        SDA_DIR.joinpath(*pgs_parts),
    ]

    # Then progressively remove the deepest folder. This handles cases such
    # as PGS/4/midt/x100 while SDA only has SDA/4/midt/.
    for n in range(len(pgs_parts) - 1, 0, -1):
        candidate_roots.append(SDA_DIR.joinpath(*pgs_parts[:n]))

    for sample_directory in candidate_roots:
        if not sample_directory.exists():
            continue

        processing_directories = sorted(
            directory
            for directory in sample_directory.rglob("*")
            if directory.is_dir()
            and list(directory.glob("*.npy"))
        )

        # Prefer a folder containing both IQ and ADP.
        for directory in processing_directories:
            if find_iq_file(directory) and find_adp_file(directory):
                return directory

    return None


def make_sample_key(pgs_path):
    """Return the folder path identifying a sample."""
    relative = pgs_path.relative_to(PGS_DIR)
    return tuple(relative.parts[:-1])


def find_matches():
    """Find PGS files and their corresponding SDA IQ/ADP files."""
    matches = []

    for pgs_path in sorted(PGS_DIR.rglob("*.mat")):
        sda_directory = find_sda_directory(pgs_path)

        matches.append({
            "key": make_sample_key(pgs_path),
            "pgs": pgs_path,
            "sda": sda_directory,
            "iq": find_iq_file(sda_directory) if sda_directory else None,
            "adp": find_adp_file(sda_directory) if sda_directory else None,
        })

    return matches


# ---------------------------------------------------------------------------
# Printing / inspection
# ---------------------------------------------------------------------------

def print_mat_information(mat_path):
    """Print variables, shapes, dtypes and useful value information."""
    print(f"\nPGS: {mat_path}")

    arrays = get_mat_arrays(mat_path)

    for name, array in arrays.items():
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")

        if np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]

            if finite.size:
                print(f"      min={finite.min()}, max={finite.max()}")

                if array.size < 10_000_000:
                    unique = np.unique(array)
                    print(f"      unique values={len(unique)}")
                    if len(unique) <= 20:
                        print(f"      values={unique}")


def print_matches(matches):
    """Print all discovered PGS/SDA matches."""
    print("\n" + "=" * 90)
    print("PGS <-> SDA MATCHES")
    print("=" * 90)

    for match in matches:
        print("-" * 90)
        print(f"Sample: {' / '.join(match['key'])}")
        print(f"  PGS: {match['pgs']}")

        if match["sda"] is None:
            print("  SDA: NOT FOUND")
        else:
            print(f"  SDA: {match['sda']}")
            print(f"  IQ:  {match['iq'] or 'NOT FOUND'}")
            print(f"  ADP: {match['adp'] or 'NOT FOUND'}")


def print_shapes(match):
    """Print shapes and check whether PGS.T matches the EBSD shape."""
    pgs_name, pgs = choose_pgs_array(match["pgs"])
    iq = np.load(match["iq"])
    adp = np.load(match["adp"])

    print("\n" + "=" * 90)
    print("SELECTED SAMPLE")
    print("=" * 90)
    print(f"Sample: {' / '.join(match['key'])}")

    print(f"\nPGS variable: {pgs_name}")
    print(f"PGS shape:    {pgs.shape}")
    print(f"IQ shape:     {iq.shape}")
    print(f"ADP shape:    {adp.shape}")

    if pgs.ndim == 2:
        print(f"PGS.T shape:  {pgs.T.shape}")

        if pgs.T.shape == iq.shape:
            print("-> PGS.T has the same shape as IQ.")

        if pgs.shape == iq.shape:
            print("-> PGS already has the same shape as IQ.")

    print("\nPossible orientation operations will be shown in the figure.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def show_array(ax, array, title, mask=False):
    """Display one 2D array."""
    if mask:
        ax.imshow(array, interpolation="nearest")
    else:
        ax.imshow(array, cmap="gray", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def get_orientation_variants(array):
    """Return common orientation possibilities for a transposed array."""
    transposed = array.T

    return {
        "Original": array,
        "Transpose": transposed,
        "Transpose + vertical flip": np.flipud(transposed),
        "Transpose + horizontal flip": np.fliplr(transposed),
        "Transpose + both flips": np.flipud(np.fliplr(transposed)),
    }


def visualize_match(match):
    """
    Visualize one complete PGS + IQ + ADP sample.

    The first figure compares the raw arrays and possible PGS orientations.
    The second figure overlays PGS grain boundaries on IQ for every PGS
    orientation that has the same shape as IQ.
    """
    pgs_name, pgs = choose_pgs_array(match["pgs"])
    iq = np.load(match["iq"])
    adp = np.load(match["adp"])

    print_shapes(match)

    variants = get_orientation_variants(pgs)

    # ---------------------------------------------------------------
    # Figure 1: raw arrays + orientation candidates
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

    show_array(
        axes[0], pgs,
        f"PGS original\n{pgs.shape}",
        mask=True,
    )

    show_array(
        axes[1], iq,
        f"Martensite IQ\n{iq.shape}",
    )

    show_array(
        axes[2], adp,
        f"Martensite ADP\n{adp.shape}",
    )

    orientation_names = [
        "Transpose",
        "Transpose + vertical flip",
        "Transpose + horizontal flip",
    ]

    for ax, name in zip(axes[3:], orientation_names):
        candidate = variants[name]
        show_array(
            ax,
            candidate,
            f"PGS {name}\n{candidate.shape}",
            mask=True,
        )

    fig.suptitle(
        f"PGS / SDA inspection: {' / '.join(match['key'])}",
        fontsize=16,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure1_path = OUTPUT_DIR / "selected_sample_raw_and_orientations.png"
    fig.savefig(figure1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure1_path}")

    # ---------------------------------------------------------------
    # Figure 2: PGS grain boundaries over IQ
    # ---------------------------------------------------------------

    if pgs.ndim != 2 or iq.ndim != 2:
        return

    matching_variants = {
        name: array
        for name, array in variants.items()
        if array.shape == iq.shape
    }

    if not matching_variants:
        print("No PGS orientation has the same shape as IQ.")
        return

    fig, axes = plt.subplots(
        1,
        len(matching_variants),
        figsize=(6 * len(matching_variants), 6),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, (name, candidate) in zip(axes, matching_variants.items()):
        ax.imshow(iq, cmap="gray")

        # Grain boundaries are locations where neighboring labels differ.
        boundary = np.zeros(candidate.shape, dtype=bool)
        boundary[:-1, :] |= candidate[:-1, :] != candidate[1:, :]
        boundary[:, :-1] |= candidate[:, :-1] != candidate[:, 1:]

        ax.imshow(
            np.ma.masked_where(~boundary, boundary),
            alpha=0.8,
            interpolation="nearest",
        )

        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"PGS grain boundaries over martensite IQ\n"
        f"{' / '.join(match['key'])}",
        fontsize=16,
    )

    figure2_path = OUTPUT_DIR / "selected_sample_pgs_boundaries_over_iq.png"
    fig.savefig(figure2_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure2_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PGS_DIR.exists():
        raise FileNotFoundError(
            f"PGS directory not found: {PGS_DIR.resolve()}"
        )

    if not SDA_DIR.exists():
        raise FileNotFoundError(
            f"SDA directory not found: {SDA_DIR.resolve()}"
        )

    matches = find_matches()

    if not matches:
        raise FileNotFoundError(
            f"No .mat files found under {PGS_DIR.resolve()}"
        )

    print(f"\nFigures will be saved to: {OUTPUT_DIR.resolve()}")
    print_matches(matches)

    print("\n" + "=" * 90)
    print("PGS MATLAB DATA")
    print("=" * 90)

    for match in matches:
        print_mat_information(match["pgs"])

    # Choose exactly ONE complete sample to visualize.
    selected = None

    if VISUALIZE_SAMPLE is not None:
        for match in matches:
            if match["key"] == VISUALIZE_SAMPLE:
                selected = match
                break
    else:
        for match in matches:
            if (
                match["sda"] is not None
                and match["iq"] is not None
                and match["adp"] is not None
            ):
                selected = match
                break

    if selected is None:
        print("\nNo complete PGS + IQ + ADP sample was found.")
        return

    visualize_match(selected)


if __name__ == "__main__":
    main()