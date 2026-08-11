from __future__ import annotations

from vision_xai.data.splits import stratified_train_val_split

N_PER_CLASS = 30
N_CLASSES = 4


def _labels() -> dict[str, int]:
    return {f"class{c}_{i:03d}": c for c in range(N_CLASSES) for i in range(N_PER_CLASS)}


def test_same_seed_is_identical() -> None:
    first = stratified_train_val_split(_labels(), 0.2, seed=42)
    second = stratified_train_val_split(_labels(), 0.2, seed=42)
    assert first == second


def test_covers_all_ids_exactly() -> None:
    labels = _labels()
    assignment = stratified_train_val_split(labels, 0.2, seed=42)
    assert set(assignment) == set(labels)
    assert set(assignment.values()) == {"train", "val"}


def test_per_class_val_count_is_round_fraction() -> None:
    labels = _labels()
    assignment = stratified_train_val_split(labels, 0.2, seed=42)
    expected_val = round(N_PER_CLASS * 0.2)
    for class_id in range(N_CLASSES):
        val_count = sum(
            1
            for sample_id, split in assignment.items()
            if labels[sample_id] == class_id and split == "val"
        )
        assert val_count == expected_val


def test_different_seed_changes_assignment() -> None:
    first = stratified_train_val_split(_labels(), 0.2, seed=42)
    second = stratified_train_val_split(_labels(), 0.2, seed=43)
    assert first != second


def test_input_order_does_not_matter() -> None:
    labels = _labels()
    reversed_labels = dict(reversed(list(labels.items())))
    assert stratified_train_val_split(labels, 0.2, seed=42) == stratified_train_val_split(
        reversed_labels, 0.2, seed=42
    )


def test_single_sample_class_stays_in_train() -> None:
    labels = {"only_one": 0}
    assert stratified_train_val_split(labels, 0.5, seed=42) == {"only_one": "train"}
