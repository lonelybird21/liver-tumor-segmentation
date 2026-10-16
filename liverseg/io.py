"""Image I/O for CT: NIfTI / NRRD / MHA, DICOM series folders, single DICOM files,
and (with caveats) ordinary pictures such as PNG/JPG screenshots of a CT slice.

All volumes are reoriented to a canonical LPS orientation before they reach a
model, so the network always sees anatomy the same way regardless of how the
scanner or converter stored it. Predictions are written back on the exact
original voxel grid.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

CANONICAL_ORIENTATION = "LPS"
PICTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VOLUME_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".dcm")


class PictureInputWarning(UserWarning):
    """Raised when segmenting a plain picture (no Hounsfield units, no spacing)."""


def is_picture(path: str | Path) -> bool:
    return Path(path).suffix.lower() in PICTURE_SUFFIXES


def _read_dicom_dir(directory: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    candidates = [directory] + [p for p in directory.rglob("*") if p.is_dir()]
    best: Optional[Tuple[int, Sequence[str]]] = None
    for d in candidates:
        for sid in reader.GetGDCMSeriesIDs(str(d)) or []:
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if best is None or len(files) > best[0]:
                best = (len(files), files)
    if best is None:
        raise FileNotFoundError(f"No DICOM series found under {directory}")
    reader.SetFileNames(best[1])
    reader.MetaDataDictionaryArrayUpdateOn()
    return reader.Execute()


def read_picture(
    path: str | Path,
    window_level: float = 50.0,
    window_width: float = 400.0,
    pixel_spacing: float = 0.8,
) -> sitk.Image:
    """Read a PNG/JPG CT slice as a 1-slice volume.

    A picture stores display grey levels, not Hounsfield units, so we invert the
    display window that was (assumed to be) used to render it:
        HU = (level - width/2) + grey/255 * width
    Results are only as good as that assumption. Pixel spacing is also unknown
    unless supplied, so areas/volumes from pictures are approximate.
    """
    warnings.warn(
        "Segmenting a picture: Hounsfield units are reconstructed from the assumed "
        f"display window (L={window_level}, W={window_width}) and pixel spacing "
        f"{pixel_spacing} mm is assumed. Use the original DICOM for reliable results.",
        PictureInputWarning,
        stacklevel=2,
    )
    img = sitk.ReadImage(str(path))
    if img.GetNumberOfComponentsPerPixel() > 1:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = arr[..., :3].mean(axis=-1)
    else:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if arr.ndim == 3:  # e.g. a 1-frame TIFF
        arr = arr[0]
    lo = window_level - window_width / 2.0
    hu = lo + arr / 255.0 * window_width
    out = sitk.GetImageFromArray(hu[None].astype(np.float32))
    out.SetSpacing((pixel_spacing, pixel_spacing, 1.0))
    return out


def read_image(path: str | Path, **picture_kwargs) -> sitk.Image:
    """Read any supported CT input as a 3D SimpleITK image (a single slice has z=1)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        img = _read_dicom_dir(p)
    elif is_picture(p):
        img = read_picture(p, **picture_kwargs)
    else:
        img = sitk.ReadImage(str(p))
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    if img.GetDimension() == 2:
        img = sitk.JoinSeries(img)
    if img.GetDimension() != 3:
        raise ValueError(f"Expected a 2D or 3D image, got {img.GetDimension()}D: {p}")
    return sitk.Cast(img, sitk.sitkFloat32)


def read_label(path: str | Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    if img.GetDimension() == 2:
        img = sitk.JoinSeries(img)
    return sitk.Cast(img, sitk.sitkUInt8)


def orient(img: sitk.Image, code: str = CANONICAL_ORIENTATION) -> sitk.Image:
    return sitk.DICOMOrient(img, code)


def to_array(img: sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Return the voxel array in (z, y, x) order and the spacing in the same order."""
    arr = sitk.GetArrayFromImage(img)
    spacing = tuple(float(s) for s in img.GetSpacing()[::-1])
    return arr, spacing


def label_to_original_grid(label_zyx: np.ndarray, oriented_ref: sitk.Image, original_ref: sitk.Image) -> sitk.Image:
    """Map a label array defined on the reoriented grid back to the original grid.

    The two grids describe the same voxels (only axis order/sign differ), so a
    nearest-neighbour resample is an exact permutation, not an interpolation.
    """
    lab = sitk.GetImageFromArray(label_zyx.astype(np.uint8))
    lab.CopyInformation(oriented_ref)
    return sitk.Resample(lab, original_ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def resample_label_to(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(label, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def write_image(img: sitk.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def case_name(path: str | Path) -> str:
    p = Path(path)
    name = p.name
    for suf in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".dcm", *PICTURE_SUFFIXES):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name
