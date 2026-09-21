from pathlib import Path
import numpy as np
import scipy.io

# This script prints the matches of PGs and SDA and their dimensions

PGS_ROOT = Path("datasets/raw/PGs")
SDA_ROOT = Path("datasets/raw/SDA")



def find_sda_files(pgs_path):
    """
    Find the IQ and ADP files corresponding to a PGS file.

    Allows PGS folders to contain extra subfolders such as:
        PGs/4/midt/x100/
        PGs/4/midt/x200/

    while SDA may be:
        SDA/4/midt/x100_processed/
        SDA/4/midt/x200_processed/
    """

    # Get relative path inside PGs
    rel = pgs_path.relative_to(PGS_ROOT)

    # Remove filename
    pgs_dir = rel.parent

    # Try to find matching SDA directories recursively
    sample_root = SDA_ROOT / pgs_dir.parts[0]

    if not sample_root.exists():
        return None, None

    iq_files = list(sample_root.rglob("*iq_sdan.npy"))
    adp_files = list(sample_root.rglob("*adp_1Dsig.npy"))

    # Use the PGS parent folders to narrow the search
    for iq in iq_files:
        for adp in adp_files:

            # Check that IQ and ADP belong to the same processed folder
            if iq.parent != adp.parent:
                continue

            # Check whether the SDA path contains the relevant
            # sample/location folders from the PGS path
            iq_parts = iq.relative_to(SDA_ROOT).parts

            # Match the first folders, e.g.:
            # PGs/1/midt/... -> SDA/1/midt/...
            if len(pgs_dir.parts) >= 2:
                if iq_parts[0] != pgs_dir.parts[0]:
                    continue

                if iq_parts[1] != pgs_dir.parts[1]:
                    continue

            # If PGS has an x100/x200 folder, require it to match
            if len(pgs_dir.parts) >= 3:
                pgs_extra = pgs_dir.parts[2]

                if pgs_extra in ["x100", "x200"]:
                    if pgs_extra not in iq_parts:
                        continue

            return iq, adp

    return None, None


def main():

    pgs_files = sorted(PGS_ROOT.rglob("*.mat"))

    print("=" * 80)
    print("PGS ↔ SDA MATCH CHECK")
    print("=" * 80)

    print(f"\nFound {len(pgs_files)} PGS .mat files\n")

    matches = 0
    pgs_dim_list = []

    for pgs_path in pgs_files:

        print("-" * 80)
        print(f"PGS:")
        print(f"  {pgs_path}")

        # Load PGS
        try:
            data = scipy.io.loadmat(pgs_path)

            if "props" not in data:
                print("  ERROR: 'props' variable not found")
                continue

            pgs = data["props"]

        except Exception as e:
            print(f"  ERROR loading PGS: {e}")
            continue

        # Find IQ and ADP
        iq_path, adp_path = find_sda_files(pgs_path)

        if iq_path is None or adp_path is None:
            print("\n  SDA MATCH: NOT FOUND")
            continue

        matches += 1

        print("\nSDA MATCH:")
        print(f"  IQ : {iq_path}")
        print(f"  ADP: {adp_path}")

        # Load IQ and ADP
        try:
            iq = np.load(iq_path)
            adp = np.load(adp_path)
        except Exception as e:
            print(f"\n  ERROR loading SDA files: {e}")
            continue

        pgs_dim_list.append(pgs.T.shape)

        # Print dimensions
        print("\nDIMENSIONS:")
        print(f"  PGS     : {pgs.shape}")
        print(f"  PGS.T   : {pgs.T.shape}")
        print(f"  IQ      : {iq.shape}")
        print(f"  ADP     : {adp.shape}")

        # Check orientations
        print("\nSHAPE CHECK:")

        print(f"  PGS == IQ     : {pgs.shape == iq.shape}")
        print(f"  PGS.T == IQ   : {pgs.T.shape == iq.shape}")

        print(f"  PGS == ADP    : {pgs.shape == adp.shape}")
        print(f"  PGS.T == ADP  : {pgs.T.shape == adp.shape}")

        print(f"  IQ == ADP      : {iq.shape == adp.shape}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"PGS files found  : {len(pgs_files)}")
    print(f"Matches found    : {matches}")
    print(f"Dimensions       : {pgs_dim_list}")
    print(f"Average dimension: ({round(sum([pgs_dim_list[i][0] for i in range(len(pgs_dim_list))])/len(pgs_dim_list))}, {round(sum([pgs_dim_list[i][1] for i in range(len(pgs_dim_list))])/len(pgs_dim_list))})")
    print("=" * 80)


if __name__ == "__main__":
    main()