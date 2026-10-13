import argparse
import hashlib
import json
import random
import re
import zipfile

from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


CLASSES = ["NORMAL", "PNEUMONIA"]


def get_patient_id(filename):
    stem = Path(filename).stem

    match = re.match(r"(person\d+)_", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match = re.match(
        r"((?:NORMAL2-)?IM-\d+)",
        stem,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    raise ValueError(
        f"Cannot identify patient ID from filename: {filename}"
    )


def prepare_data(zip_path, output):
    output = Path(output)

    if (output / "manifest.json").exists():
        raise ValueError(
            "Data is already prepared. Skip this step."
        )

    output.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = {}
    excluded = Counter()

    with zipfile.ZipFile(zip_path) as archive:
        entries = []
        roots = Counter()

        for info in archive.infolist():
            path = PurePosixPath(info.filename)

            if "__MACOSX" in path.parts:
                continue

            if path.name.startswith("."):
                continue

            if path.suffix.lower() not in (
                ".jpg", ".jpeg", ".png"
            ):
                continue

            if len(path.parts) < 3:
                continue

            split = path.parts[-3]
            label = path.parts[-2]

            if split not in ("train", "val", "test"):
                continue

            if label not in CLASSES:
                continue

            root = "/".join(path.parts[:-3])

            roots[root] += 1
            entries.append(
                (root, info, split, label, path.name)
            )

        if not roots:
            raise ValueError(
                "No supported chest X-ray dataset found."
            )

        # Choose one dataset copy.
        chosen_root = sorted(
            roots,
            key=lambda root: (
                -roots[root],
                len(root),
                root,
            ),
        )[0]

        selected = [
            entry
            for entry in entries
            if entry[0] == chosen_root
        ]

        # Keep test images first when removing duplicates.
        selected.sort(
            key=lambda entry: (
                entry[2] != "test",
                entry[1].filename,
            )
        )

        for _, info, split, label, filename in selected:
            if split == "val":
                excluded["original_validation"] += 1
                continue

            content = archive.read(info)
            image_hash = hashlib.sha256(content).hexdigest()

            if image_hash in seen_hashes:
                if seen_hashes[image_hash] != label:
                    raise ValueError(
                        "An identical image has conflicting labels."
                    )

                excluded["exact_duplicates"] += 1
                continue

            seen_hashes[image_hash] = label

            relative_path = (
                Path("images") / split / label / filename
            )

            destination = output / relative_path
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            destination.write_bytes(content)

            records.append(
                {
                    "path": relative_path.as_posix(),
                    "label": CLASSES.index(label),
                    "source_split": split,
                    "patient": get_patient_id(filename),
                    "sha256": image_hash,
                }
            )

    test_patients = {
        record["patient"]
        for record in records
        if record["source_split"] == "test"
    }

    clean_records = []

    for record in records:
        if (
            record["source_split"] == "train"
            and record["patient"] in test_patients
        ):
            excluded["training_patient_overlap"] += 1
        else:
            clean_records.append(record)

    patient_groups = defaultdict(list)

    for record in clean_records:
        if record["source_split"] == "train":
            patient_groups[record["patient"]].append(record)

    rng = random.Random(42)
    validation_patients = set()

    for label in (0, 1):
        candidates = []

        for patient, images in patient_groups.items():
            labels = {image["label"] for image in images}

            if len(labels) != 1:
                raise ValueError(
                    f"Conflicting labels for patient: {patient}"
                )

            if label in labels:
                candidates.append(patient)

        candidates.sort()
        rng.shuffle(candidates)

        image_count = sum(
            len(patient_groups[patient])
            for patient in candidates
        )

        validation_target = round(image_count * 0.20)
        selected_count = 0

        for patient in candidates:
            if selected_count >= validation_target:
                break

            validation_patients.add(patient)
            selected_count += len(patient_groups[patient])

    for record in clean_records:
        if record["source_split"] == "test":
            record["split"] = "test"
        elif record["patient"] in validation_patients:
            record["split"] = "val"
        else:
            record["split"] = "train"

    for split in ("train", "val", "test"):
        labels = {
            record["label"]
            for record in clean_records
            if record["split"] == split
        }

        if labels != {0, 1}:
            raise ValueError(
                f"{split} must contain both classes."
            )

    manifest = {
        "classes": CLASSES,
        "records": clean_records,
        "excluded": dict(excluded),
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nDataset prepared successfully.")

    for split in ("train", "val", "test"):
        counts = Counter(
            CLASSES[record["label"]]
            for record in clean_records
            if record["split"] == split
        )

        print(f"{split.upper()}: {dict(counts)}")

    print("\nExcluded:", dict(excluded))
    print(
        "\nPatient IDs are inferred from filenames, "
        "not verified patient metadata."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--zip", required=True)
    parser.add_argument("--output", default="data")

    args = parser.parse_args()

    prepare_data(args.zip, args.output)