import hashlib
import numpy as np
import pytest
from scope.config import MANIFEST, PDES, ASSETS, base_config, load_config
from scope.io import sha256
from scope.data import verify_source, arrow_batches, build_cache, verify_cache


def test_pinned_manifest_and_geometry_identity():
    assert len(PDES) == 5
    assert MANIFEST["revision"] == "3ededc4f2d8a1592a52a7a865cb903dab7647820"
    for pde in PDES:
        for split in ("train", "test"):
            prefix = pde + ("_hf/" if split == "train" else "_test_hf/")
            assert any(f["path"] == prefix + "state.json" for f in MANIFEST["datasets"][pde]["files"])
    source = MANIFEST["datasets"]["ns-bounded"]
    for split in ("train", "test"):
        path = ASSETS / f"{split}-geometry.npy"
        assert sha256(path) == source["audited_splits"][split]["geometry_sha256"]
        assert np.load(path, allow_pickle=False).shape == (source[split + "_records"], 3)


def test_source_checksums(tmp_path):
    content = b"public-source-fixture"
    path = tmp_path / "data.json"
    path.write_bytes(content)
    verify_source(path, {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    verify_source(path, {"bytes": len(content), "git_blob_sha1": git_hash})
    with pytest.raises(ValueError, match="checksum"):
        verify_source(path, {"bytes": len(content), "sha256": "0" * 64})


def test_arrow_cache_conversion_and_corruption(tmp_path, monkeypatch):
    import json
    import pyarrow as pa

    fields = np.random.default_rng(1).normal(size=(2, 2, 128, 128))
    source = tmp_path / "source"
    folder = tmp_path / "poisson"
    source.mkdir()
    folder.mkdir()
    table = pa.table({"id": [0, 1], "data": fields.tolist()})
    with pa.OSFile(str(source / "data.arrow"), "wb") as stream:
        with pa.ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    (source / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data.arrow"}]}))
    expected = tmp_path / "expected.npy"
    np.save(expected, fields.astype(np.float32), allow_pickle=False)
    original = MANIFEST["datasets"]["poisson"].copy()
    original.update(train_records=2, train_cache_sha256=sha256(expected))
    monkeypatch.setitem(MANIFEST["datasets"], "poisson", original)
    receipt = build_cache("poisson", "train", source, folder)
    assert receipt["records"] == 2
    assert verify_cache("poisson", "train", tmp_path) == receipt
    np.testing.assert_array_equal(np.load(folder / "train-f32.npy"), fields.astype(np.float32))
    with (folder / "train-f32.npy").open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(b"bad!")
    with pytest.raises(ValueError, match="checksum"):
        verify_cache("poisson", "train", tmp_path)
