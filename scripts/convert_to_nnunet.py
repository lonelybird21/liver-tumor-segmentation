"""Convert LiTS (volume-N / segmentation-N) to an nnU-Net v2 raw dataset.

nnU-Net is the strongest widely-used baseline for this task. Train it on the
same patients and score both models with `liverseg-evaluate --pred-dir` so the
comparison is apples to apples. Pass `--exclude-test data/preprocessed/<exp>/splits.json`
to keep our held-out test patients out of nnU-Net training as well.

    python scripts/convert_to_nnunet.py --lits data/raw --out $nnUNet_raw/Dataset017_LiTS \
        --exclude-test data/preprocessed/3d_fullres/splits.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-test", help="splits.json whose 'test' cases go to imagesTs instead of imagesTr")
    a = ap.parse_args()
    src, out = Path(a.lits), Path(a.out)
    test = set(json.loads(Path(a.exclude_test).read_text())["test"]) if a.exclude_test else set()
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n_train = 0
    for img in sorted(src.rglob("volume-*.nii*")):
        cid = re.sub(r"\.nii(\.gz)?$", "", img.name).replace("volume-", "")
        seg = next(iter(src.rglob(f"segmentation-{cid}.nii*")), None)
        if seg is None:
            continue
        name = f"LiTS_{int(cid):03d}"
        split = "Ts" if f"case_{cid}" in test else "Tr"
        ext = ".nii.gz" if img.name.endswith(".gz") else ".nii"
        shutil.copy(img, out / f"images{split}" / f"{name}_0000{ext}")
        shutil.copy(seg, out / f"labels{split}" / f"{name}{ext}")
        n_train += split == "Tr"
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": [1, 2], "tumor": 2},
        "regions_class_order": [1, 2],
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }, indent=2))
    print(f"nnU-Net dataset at {out}: {n_train} training cases, {len(test)} held out")
    print("NOTE: if your LiTS files are .nii (not .nii.gz), set file_ending to .nii in dataset.json")


if __name__ == "__main__":
    main()
