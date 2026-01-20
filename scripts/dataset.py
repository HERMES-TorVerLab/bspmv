# Standard imports
import tarfile
from pathlib import Path

# Custom imports
from matrix_macros import *


# Directories
TAR_DIR = Path("../matrix/tar")
OUT_DIR = Path("../matrix")


def check_matrices_have_tar(tar_dir, matrix_files):
    tar_dir = Path(tar_dir)

    if not tar_dir.is_dir():
        raise NotADirectoryError(tar_dir)

    tar_basenames = set()
    for p in tar_dir.iterdir():
        if p.suffix == ".gz" and p.stem.endswith(".tar"):
            tar_basenames.add(Path(p.stem).stem)   # removes .tar.gz
        elif p.suffix == ".tar":
            tar_basenames.add(p.stem)

    matrix_basenames = {Path(m).stem for m in matrix_files}

    present = sorted(matrix_basenames & tar_basenames)
    missing = sorted(matrix_basenames - tar_basenames)

    return present, missing




def extract_matrices_from_tar(
    tar_dir: str | Path,
    out_dir: str | Path,
    matrix_files: list[str] | set[str],
) -> None:
    """
    Extract selected files from .tar.gz archives, flattening internal paths.

    Parameters
    ----------
    tar_dir : str or Path
        Directory containing .tar.gz files
    out_dir : str or Path
        Output directory where files are written directly
    matrix_files : list or set of str
        Filenames to extract
    """
    tar_dir = Path(tar_dir)
    out_dir = Path(out_dir)
    matrix_files = set(matrix_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    for tar_path in tar_dir.glob("*.tar.gz"):
        print(f"Processing {tar_path.name}")
        if tar_path.name == "AGATHA_2015.tar.gz":
            continue

        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                filename = Path(member.name).name  # flatten path

                if filename not in matrix_files:
                    continue

                out_path = out_dir / filename
                if out_path.exists():
                    print(f"  Skipping {filename} (already exists)")
                    continue

                with tar.extractfile(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

                print(f"  Extracted {filename}")




def unpack_matrices():
    extract_matrices_from_tar(tar_dir=TAR_DIR, out_dir=OUT_DIR, matrix_files=MATRIX_FILES)


if __name__ == "__main__":
    unpack_matrices()
    present, missing = check_matrices_have_tar(
        TAR_DIR,
        MATRIX_FILES
    )

    print(f"\nFound {len(present)} / {len(MATRIX_FILES)} matrices")

    if missing:
        print("Missing matrices:")
        for m in missing:
            print(f"  - {m}.tar.gz")
    else:
        print("All matrices are present")