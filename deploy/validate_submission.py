#!/usr/bin/env python3
"""Package and validate one PNG result for every unlabeled test image.

The archive format checked here is the locally documented task convention:
one top-level ``result/`` directory containing same-stem PNG masks. This does
not claim to reproduce an unpublished competition-side evaluator.
"""
import argparse
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def expected_outputs(input_dir: Path) -> dict[str, Path]:
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no test images found under {input_dir}")
    mapping = {}
    for image in images:
        rel = image.relative_to(input_dir).with_suffix(".png").as_posix()
        if rel in mapping:
            raise ValueError(f"duplicate output filename after .png conversion: {rel}")
        mapping[rel] = image
    return mapping


def check_mask(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as im:
        im.load()
        if im.format != "PNG":
            raise ValueError(f"not PNG: {path}")
        if im.mode != "L":
            raise ValueError(f"expected 8-bit single-channel L mode, got {im.mode}: {path}")
        if im.size != expected_size:
            raise ValueError(f"size {im.size} != expected {expected_size}: {path}")
        values = np.asarray(im)
        actual = set(np.unique(values).tolist())
        if not actual <= {0, 1}:
            raise ValueError(f"mask has values outside 0/1: {path}: {sorted(actual)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True, help="result.zip path")
    parser.add_argument("--package", action="store_true", help="create/replace result.zip")
    args = parser.parse_args()

    if args.zip.name != "result.zip":
        raise ValueError(f"submission archive must be named result.zip, got {args.zip.name}")
    expected = expected_outputs(args.input_dir)
    if len(expected) != 300:
        raise ValueError(f"expected exactly 300 evaluation inputs, found {len(expected)}")
    actual_files = {
        p.relative_to(args.pred_dir).as_posix(): p
        for p in args.pred_dir.rglob("*") if p.is_file()
    }
    expected_names = set(expected)
    if set(actual_files) != expected_names:
        missing = sorted(expected_names - set(actual_files))
        extra = sorted(set(actual_files) - expected_names)
        raise ValueError(f"prediction file set mismatch; missing={missing[:10]}, extra={extra[:10]}")

    for rel, source in expected.items():
        with Image.open(source) as im:
            size = im.size
        if size != (672, 378):
            raise ValueError(f"official input size must be 672x378, got {size}: {source}")
        check_mask(actual_files[rel], size)

    if args.package:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for rel in sorted(expected):
                zf.write(actual_files[rel], (PurePosixPath("result") / rel).as_posix())

    with zipfile.ZipFile(args.zip, "r") as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        want = {(PurePosixPath("result") / rel).as_posix() for rel in expected}
        got = set(names)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(f"ZIP member set mismatch; missing={missing[:10]}, extra={extra[:10]}")
        for rel, source in expected.items():
            member = (PurePosixPath("result") / rel).as_posix()
            with Image.open(zf.open(member)) as im:
                im.load()
                if im.format != "PNG" or im.mode != "L":
                    raise ValueError(f"ZIP entry is not an L-mode PNG: {member}")
                with Image.open(source) as source_im:
                    if im.size != source_im.size:
                        raise ValueError(f"ZIP entry size mismatch: {member}")
                values = np.asarray(im)
                unique = set(np.unique(values).tolist())
                if not unique <= {0, 1}:
                    raise ValueError(f"ZIP entry has values outside 0/1: {member}")

    max_bytes = 100 * 1024 * 1024
    if args.zip.stat().st_size > max_bytes:
        raise ValueError(f"ZIP exceeds the 100 MB upload limit: {args.zip.stat().st_size} bytes")

    print(f"PASS: {len(expected)} predictions; dimensions and 0/1 values match inputs")
    print(f"PASS: {args.zip.name} has exactly {len(expected)} PNG files under result/; CRC verified; no extra members")
    print(f"zip_bytes={args.zip.stat().st_size}")


if __name__ == "__main__":
    main()
