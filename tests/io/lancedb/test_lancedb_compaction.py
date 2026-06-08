from __future__ import annotations

from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pytest
from lance import Blob

from daft_lance import compact_files
from daft_lance.lance_compaction import compact_files_internal


def create_dataset_with_fragments(path: Path, fragment_data: list[pd.DataFrame]):
    """Create a Lance dataset with multiple fragments by appending batches."""
    assert len(fragment_data) >= 1, "fragment_data must contain at least one DataFrame"
    first_df = fragment_data[0]
    lance.write_dataset(pa.Table.from_pandas(first_df), path, max_rows_per_file=len(first_df))

    for df in fragment_data[1:]:
        lance.write_dataset(pa.Table.from_pandas(df), path, mode="append", max_rows_per_file=len(df))

    return lance.dataset(str(path))


def test_basic_compaction(tmp_path: Path):
    """Basic flow: compact two fragments into one; row count unchanged and fragment count reduced."""
    dataset_path = tmp_path / "test_dataset_for_compaction"
    fragment1 = pd.DataFrame({"id": range(1, 11), "value": [f"row_{i}" for i in range(1, 11)]})
    fragment2 = pd.DataFrame({"id": range(11, 21), "value": [f"row_{i}" for i in range(11, 21)]})

    dataset = create_dataset_with_fragments(dataset_path, [fragment1, fragment2])
    assert len(dataset.get_fragments()) == 2, "Expected 2 fragments initially"
    assert dataset.count_rows() == 20, "Expected 20 rows initially"
    original_schema = dataset.schema

    metrics = compact_files(
        uri=str(dataset_path),
        compaction_options={
            "target_rows_per_fragment": 100,
            "num_threads": 1,
        },
    )
    assert metrics is not None, "Compaction should produce metrics"
    assert getattr(metrics, "fragments_removed", None) == 2, "Should remove 2 fragments"
    assert getattr(metrics, "fragments_added", None) == 1, "Should add 1 fragment"

    dataset = lance.dataset(str(dataset_path))
    fragments = dataset.get_fragments()
    assert len(fragments) == 1, "Expected 1 fragment after compaction"
    assert fragments[0].count_rows() == 20, "Single fragment should contain 20 rows after compaction"
    assert dataset.count_rows() == 20, "Total row count should remain 20"
    assert dataset.schema == original_schema, "Compaction should not alter dataset schema"


def test_deletion_compaction(tmp_path: Path):
    """Deletion materialization: compaction merges fragments and preserves post-deletion row count."""
    dataset_path = tmp_path / "test_dataset_for_deletion_compaction"
    fragment1 = pd.DataFrame({"id": range(1, 11), "value": [f"row_{i}" for i in range(1, 11)]})
    fragment2 = pd.DataFrame({"id": range(11, 21), "value": [f"row_{i}" for i in range(11, 21)]})

    dataset = create_dataset_with_fragments(dataset_path, [fragment1, fragment2])
    assert len(dataset.get_fragments()) == 2
    assert dataset.count_rows() == 20

    dataset.delete("id <= 9")
    dataset = lance.dataset(str(dataset_path))
    assert len(dataset.get_fragments()) == 2, "Delete marks should not immediately change fragment count"
    assert dataset.count_rows() == 11, "Expected 11 rows after deletion"

    metrics = compact_files(
        uri=str(dataset_path),
        compaction_options={
            "materialize_deletions": True,
            "materialize_deletions_threshold": 0.5,
            "target_rows_per_fragment": 100,
            "num_threads": 1,
        },
    )
    assert metrics is not None
    assert getattr(metrics, "fragments_removed", None) == 2
    assert getattr(metrics, "fragments_added", None) == 1

    dataset = lance.dataset(str(dataset_path))
    fragments = dataset.get_fragments()
    assert len(fragments) == 1
    assert fragments[0].count_rows() == 11
    assert dataset.count_rows() == 11


def test_idempotent_repeated_compaction(tmp_path: Path):
    """Idempotency: repeat compaction on already compacted dataset should return None and have no side effects."""
    dataset_path = tmp_path / "test_idempotent_compaction"
    df1 = pd.DataFrame({"id": range(0, 5), "v": [f"a{i}" for i in range(5)]})
    df2 = pd.DataFrame({"id": range(5, 10), "v": [f"b{i}" for i in range(5)]})

    dataset = create_dataset_with_fragments(dataset_path, [df1, df2])
    metrics1 = compact_files(uri=str(dataset_path), compaction_options={"target_rows_per_fragment": 100})
    assert metrics1 is not None

    metrics2 = compact_files(uri=str(dataset_path), compaction_options={"target_rows_per_fragment": 100})
    assert metrics2 is None, "Second compaction should return None"

    dataset = lance.dataset(str(dataset_path))
    assert len(dataset.get_fragments()) == 1
    assert dataset.count_rows() == 10


def test_invalid_compaction_options_key(tmp_path: Path):
    """Unknown compaction option key should raise ValueError."""
    dataset_path = tmp_path / "test_invalid_options"
    df1 = pd.DataFrame({"id": [0, 1], "v": ["x", "y"]})
    df2 = pd.DataFrame({"id": [2, 3], "v": ["z", "w"]})
    create_dataset_with_fragments(dataset_path, [df1, df2])

    with pytest.raises(ValueError):
        compact_files(uri=str(dataset_path), compaction_options={"nonexistent_option": True})


def _blob_v2_table(
    ids: list[int],
    labels: list[str],
    blobs: list[object],
    *,
    schema_metadata: dict[bytes, bytes] | None = None,
    label_metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    arrays = [
        pa.array(ids, type=pa.int64()),
        pa.array(labels, type=pa.string()),
        lance.blob_array(blobs),
    ]
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("label", pa.string(), nullable=False, metadata=label_metadata),
            pa.field("blob", arrays[2].type, nullable=False),
        ],
        metadata=schema_metadata,
    )
    return pa.Table.from_arrays(arrays, schema=schema)


def _read_blob_bytes_by_id(uri: str) -> dict[int, tuple[str, bytes]]:
    ds = lance.dataset(uri)
    rows = ds.to_table(columns=["id", "label"], with_row_id=True).to_pylist()
    row_ids = [row["_rowid"] for row in rows]
    blobs = ds.take_blobs("blob", row_ids)
    out: dict[int, tuple[str, bytes]] = {}
    for row, blob in zip(rows, blobs, strict=True):
        with blob as f:
            out[row["id"]] = (row["label"], f.read())
    return out


def test_blob_v2_compaction_preserves_blob_bytes(tmp_path: Path):
    """Blob V2 compaction rewrites visible rows while preserving blob bytes."""
    dataset_path = tmp_path / "test_blob_v2_compaction"
    external_path = tmp_path / "external.bin"
    external_path.write_bytes(b"external-" * 1024)
    external_uri = f"file://{external_path}"

    expected = {
        1: ("inline", b"tiny-inline"),
        2: ("packed", b"x" * 100_000),
        3: ("dedicated", b"y" * 5_000_000),
        4: ("external", external_path.read_bytes()),
        5: ("external-slice", external_path.read_bytes()[128:512]),
    }
    lance.write_dataset(
        _blob_v2_table(
            [1, 2],
            ["inline", "packed"],
            [expected[1][1], expected[2][1]],
        ),
        dataset_path,
        data_storage_version="2.2",
        max_rows_per_file=2,
        allow_external_blob_outside_bases=True,
    )
    lance.write_dataset(
        _blob_v2_table(
            [3, 4, 5],
            ["dedicated", "external", "external-slice"],
            [
                expected[3][1],
                external_uri,
                Blob.from_uri(external_uri, position=128, size=384),
            ],
        ),
        dataset_path,
        mode="append",
        data_storage_version="2.2",
        max_rows_per_file=3,
        allow_external_blob_outside_bases=True,
    )

    ds = lance.dataset(str(dataset_path))
    assert len(ds.get_fragments()) == 2
    assert _read_blob_bytes_by_id(str(dataset_path)) == expected

    metrics = compact_files(
        uri=str(dataset_path),
        compaction_options={
            "target_rows_per_fragment": 100,
            "num_threads": 1,
        },
    )

    assert metrics is not None
    assert getattr(metrics, "fragments_removed", None) == 2
    assert getattr(metrics, "fragments_added", None) == 1
    assert getattr(metrics, "files_removed", None) == 2
    assert getattr(metrics, "files_added", None) == 1
    ds = lance.dataset(str(dataset_path))
    assert len(ds.get_fragments()) == 1
    external_path.unlink()
    assert _read_blob_bytes_by_id(str(dataset_path)) == expected


def test_blob_v2_compaction_materializes_single_fragment_deletions(tmp_path: Path):
    """Blob V2 fallback should materialize deletions even when fragment count is already one."""
    dataset_path = tmp_path / "test_blob_v2_single_fragment_deletion"
    lance.write_dataset(
        _blob_v2_table([1, 2, 3], ["one", "two", "three"], [b"1", b"2", b"3"]),
        dataset_path,
        data_storage_version="2.2",
        max_rows_per_file=10,
    )
    ds = lance.dataset(str(dataset_path))
    ds.delete("id = 2")
    ds = lance.dataset(str(dataset_path))
    assert len(ds.get_fragments()) == 1
    assert ds.get_fragments()[0].num_deletions == 1
    assert _read_blob_bytes_by_id(str(dataset_path)) == {1: ("one", b"1"), 3: ("three", b"3")}

    metrics = compact_files(
        uri=str(dataset_path),
        compaction_options={
            "materialize_deletions": True,
            "target_rows_per_fragment": 100,
            "num_threads": 1,
        },
    )

    assert metrics is not None
    ds = lance.dataset(str(dataset_path))
    assert len(ds.get_fragments()) == 1
    assert ds.get_fragments()[0].num_deletions == 0
    assert _read_blob_bytes_by_id(str(dataset_path)) == {1: ("one", b"1"), 3: ("three", b"3")}


def test_blob_v2_compaction_rejects_stale_versions(tmp_path: Path):
    """Fallback overwrite must not compact a non-latest snapshot into latest."""
    dataset_path = tmp_path / "test_blob_v2_stale_version_compaction"
    ds = lance.write_dataset(
        _blob_v2_table([1], ["one"], [b"1"]),
        dataset_path,
        data_storage_version="2.2",
        max_rows_per_file=1,
    )
    stale_version = ds.version
    lance.write_dataset(
        _blob_v2_table([2], ["two"], [b"2"]),
        dataset_path,
        mode="append",
        data_storage_version="2.2",
        max_rows_per_file=1,
    )

    with pytest.raises(ValueError, match="latest dataset version"):
        compact_files(
            uri=str(dataset_path), version=stale_version, compaction_options={"target_rows_per_fragment": 100}
        )

    assert _read_blob_bytes_by_id(str(dataset_path)) == {1: ("one", b"1"), 2: ("two", b"2")}


def test_blob_v2_compaction_validates_options(tmp_path: Path):
    """Unknown compaction options should be rejected on the Blob V2 path too."""
    dataset_path = tmp_path / "test_blob_v2_invalid_options"
    lance.write_dataset(
        _blob_v2_table([1], ["one"], [b"1"]),
        dataset_path,
        data_storage_version="2.2",
        max_rows_per_file=1,
    )
    lance.write_dataset(
        _blob_v2_table([2], ["two"], [b"2"]),
        dataset_path,
        mode="append",
        data_storage_version="2.2",
        max_rows_per_file=1,
    )

    with pytest.raises(ValueError, match="Invalid compaction options"):
        compact_files(uri=str(dataset_path), compaction_options={"nonexistent_option": True})


def test_blob_v2_compaction_preserves_schema_metadata(tmp_path: Path):
    """Fallback should keep non-blob field metadata and nullability while rebuilding the table."""
    dataset_path = tmp_path / "test_blob_v2_schema_metadata"
    lance.write_dataset(
        _blob_v2_table(
            [1],
            ["one"],
            [b"1"],
            schema_metadata={b"schema-key": b"schema-value"},
            label_metadata={b"field-key": b"field-value"},
        ),
        dataset_path,
        data_storage_version="2.2",
        max_rows_per_file=1,
    )
    lance.write_dataset(
        _blob_v2_table(
            [2],
            ["two"],
            [b"2"],
            schema_metadata={b"schema-key": b"schema-value"},
            label_metadata={b"field-key": b"field-value"},
        ),
        dataset_path,
        mode="append",
        data_storage_version="2.2",
        max_rows_per_file=1,
    )

    before_schema = lance.dataset(str(dataset_path)).schema
    compact_files(uri=str(dataset_path), compaction_options={"target_rows_per_fragment": 100})
    after_schema = lance.dataset(str(dataset_path)).schema

    assert after_schema == before_schema
    assert after_schema.metadata == {b"schema-key": b"schema-value"}
    assert after_schema.field("label").metadata == {b"field-key": b"field-value"}
    assert after_schema.field("label").nullable is False


def test_non_blob_compaction_still_reaches_planning(monkeypatch):
    """The Blob V2 fallback should not intercept ordinary Lance datasets."""

    class FakeDataset:
        schema = pa.schema([("id", pa.int64())])

    class EmptyPlan:
        tasks = []

        def num_tasks(self):
            return 0

    planned = False

    def plan(*args, **kwargs):
        nonlocal planned
        planned = True
        return EmptyPlan()

    monkeypatch.setattr("daft_lance.lance_compaction.Compaction.plan", plan)

    assert compact_files_internal(FakeDataset()) is None  # type: ignore[arg-type]
    assert planned is True


def test_compaction_with_partition_num(tmp_path: Path):
    """Compaction using partition_num should succeed or gracefully skip when unsupported."""
    dataset_path = tmp_path / "test_compaction_partition_num"
    data = pa.table({"a": range(800), "b": range(800)})
    dataset = lance.write_dataset(data, dataset_path, max_rows_per_file=200)
    pre_fragments = dataset.get_fragments()
    assert len(pre_fragments) == 4

    pre_rows = dataset.count_rows()
    metrics = compact_files(
        uri=str(dataset_path),
        compaction_options={
            "target_rows_per_fragment": 400,
            "num_threads": 1,
        },
        partition_num=2,
        concurrency=2,
    )
    assert metrics is not None, "Compaction should produce metrics"
    dataset = lance.dataset(str(dataset_path))
    post_fragments = len(dataset.get_fragments())
    post_rows = dataset.count_rows()
    assert post_fragments == 2, "Fragment count should be reduced after compaction"
    assert post_rows == pre_rows, "Row count should remain unchanged after compaction"
