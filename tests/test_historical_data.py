"""Historical dataset download tests (mocked network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from trencher.cli import main
from trencher.data.checksums import md5_file, parse_sha256sums, sha256_file
from trencher.data.download import download_http
from trencher.data.historical import download_historical
from trencher.data.inventory import inventory_file
from trencher.data.zenodo import download_zenodo_record


def test_cli_download_historical_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["data", "download-historical", "--help"])
    assert result.exit_code == 0
    assert "download-historical" in result.output


def test_download_http_resumable_and_checksum(tmp_path: Path, monkeypatch) -> None:
    payload = b"hello-trencher-historical"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, content: bytes, status: int = 200) -> None:
            self.status_code = status
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, chunk_size=1024):
            yield self._content

    class FakeSession:
        def get(self, url, stream=False, headers=None, timeout=120):
            calls["n"] += 1
            headers = headers or {}
            if "Range" in headers:
                # Resume second half.
                return FakeResponse(payload[len(payload) // 2 :], status=206)
            # First call writes only half then "fails" by leaving partial via caller logic.
            return FakeResponse(payload[: len(payload) // 2], status=200)

    dest = tmp_path / "file.bin"
    # Seed a partial file for resume path.
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(payload[: len(payload) // 2])
    result = download_http(
        "https://example.test/file.bin",
        dest,
        expected_size=len(payload),
        checksum=("sha256", digest),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert result["status"] == "downloaded"
    assert dest.read_bytes() == payload
    # Second call should skip verified.
    result2 = download_http(
        "https://example.test/file.bin",
        dest,
        expected_size=len(payload),
        checksum=("sha256", digest),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert result2["status"] == "skipped_verified"


def test_inventory_parquet_and_csv(tmp_path: Path) -> None:
    table = pa.table(
        {
            "mint": ["a", "b"],
            "wallet": ["w1", "w2"],
            "block_time": [1, 2],
        }
    )
    parquet_path = tmp_path / "tokens.parquet"
    pq.write_table(table, parquet_path)
    inv = inventory_file(parquet_path, relative="tokens.parquet")
    assert inv["openable"] is True
    assert inv["row_count"] == 2
    assert "mint" in inv["join_keys"]
    assert "wallet" in inv["join_keys"]

    csv_path = tmp_path / "launches.csv"
    csv_path.write_text("mint,wallet\nm1,w1\nm2,w2\n", encoding="utf-8")
    csv_inv = inventory_file(csv_path, relative="launches.csv")
    assert csv_inv["row_count"] == 2
    assert csv_inv["join_keys"] == ["mint", "wallet"]


def test_parse_sha256sums() -> None:
    text = "abc123 def.bin\n# comment\ndeadbeef *other.bin\n"
    mapping = parse_sha256sums(text)
    assert mapping["def.bin"] == "abc123"
    assert mapping["other.bin"] == "deadbeef"


def test_download_historical_mocked(tmp_path: Path, monkeypatch) -> None:
    external = tmp_path / "external"
    manifests = tmp_path / "manifests"

    # Fake HF downloads for pumpfun + solarchive.
    def fake_hf_download(**kwargs):
        dest_dir: Path = kwargs["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        if kwargs.get("light_only"):
            readme = dest_dir / "README.md"
            readme.write_text("# solarchive\n", encoding="utf-8")
            schema = dest_dir / "schemas" / "transactions.json"
            schema.parent.mkdir(parents=True, exist_ok=True)
            schema.write_text('{"type":"object"}', encoding="utf-8")
            index = dest_dir / "index.json"
            index.write_text('{"partitions":[]}', encoding="utf-8")
            files = [readme, schema, index]
        else:
            # Minimal parquet corpus stand-in.
            table = pa.table({"mint": ["x"], "wallet": ["y"], "block_time": [10]})
            tokens = dest_dir / "tokens.parquet"
            pq.write_table(table, tokens)
            issues = dest_dir / "KNOWN_ISSUES.md"
            issues.write_text("none\n", encoding="utf-8")
            files = [tokens, issues]
        out = []
        for path in files:
            rel = path.relative_to(dest_dir).as_posix()
            out.append(
                {
                    "path": rel,
                    "url": f"https://example/{rel}",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "publisher_checksum": {"alg": "sha256", "value": sha256_file(path)},
                    "verified": True,
                    "status": "downloaded",
                }
            )
        return out

    monkeypatch.setattr("trencher.data.historical.download_hf_dataset", fake_hf_download)
    monkeypatch.setattr("trencher.data.historical.list_hf_files", lambda *a, **k: [])

    def fake_zenodo(**kwargs):
        dest_dir: Path = kwargs["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        source_id = kwargs["source_id"]
        if source_id == "red_pump_2026":
            launches = dest_dir / "red_pump_2026_v1_launches.jsonl.gz"
            import gzip

            with gzip.open(launches, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"mint": "m1", "wallet": "w1"}) + "\n")
            readme = dest_dir / "README.md"
            readme.write_text("red-pump\n", encoding="utf-8")
            schema = dest_dir / "SCHEMA.md"
            schema.write_text("schema\n", encoding="utf-8")
            license_f = dest_dir / "LICENSE"
            license_f.write_text("cc-by-4.0\n", encoding="utf-8")
            citation = dest_dir / "CITATION.cff"
            citation.write_text("title: red-pump\n", encoding="utf-8")
            sums = dest_dir / "SHA256SUMS"
            lines = []
            files = [launches, readme, schema, license_f, citation]
            for path in files:
                lines.append(f"{sha256_file(path)}  {path.name}")
            sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
            records = []
            for path in [*files, sums]:
                records.append(
                    {
                        "path": path.name,
                        "url": f"https://zenodo.example/{path.name}",
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "publisher_checksum": {"alg": "sha256", "value": sha256_file(path)},
                        "verified": True,
                        "status": "downloaded",
                    }
                )
            meta = {
                "title": "RED-PUMP",
                "version": "1.2",
                "doi": "10.5281/zenodo.21383616",
                "license": "cc-by-4.0",
                "record_id": "21383616",
                "url": "https://zenodo.org/records/21383616",
            }
            return meta, records

        # cohort zip
        zip_path = dest_dir / "RED-COHORT-2026-v1_1_1.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("README.md", "cohort\n")
            zf.writestr("sniper_cohorts.jsonl", json.dumps({"wallet": "w"}) + "\n")
        extract_root = dest_dir / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
        (extract_root / f".extracted_{zip_path.stem}").write_text("ok\n", encoding="utf-8")
        records = [
            {
                "path": zip_path.name,
                "url": "https://zenodo.example/cohort.zip",
                "size": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
                "publisher_checksum": {"alg": "md5", "value": md5_file(zip_path)},
                "verified": True,
                "status": "downloaded",
            }
        ]
        for extracted in sorted(extract_root.rglob("*")):
            if extracted.is_file() and not extracted.name.startswith(".extracted_"):
                rel = extracted.relative_to(dest_dir).as_posix()
                records.append(
                    {
                        "path": rel,
                        "url": None,
                        "size": extracted.stat().st_size,
                        "sha256": sha256_file(extracted),
                        "publisher_checksum": None,
                        "verified": False,
                        "status": "extracted",
                    }
                )
        meta = {
            "title": "RED-COHORT",
            "version": "1.1.1",
            "doi": "10.5281/zenodo.21765387",
            "license": "cc-by-4.0",
            "record_id": "21765387",
            "url": "https://zenodo.org/records/21765387",
        }
        return meta, records

    monkeypatch.setattr("trencher.data.historical.download_zenodo_record", fake_zenodo)

    result = download_historical(
        external_root=external,
        manifest_root=manifests,
        force=False,
    )
    assert result.ok, result.errors
    assert (manifests / "registry.json").exists()
    assert (manifests / "inventory.json").exists()
    assert (manifests / "pumpfun_memecoin_corpus" / "provenance.json").exists()
    assert (external / "red_cohort_2026" / "RED-COHORT-2026-v1_1_1.zip").exists()
    assert (external / "red_cohort_2026" / "extracted" / "README.md").exists()
    registry = json.loads((manifests / "registry.json").read_text())
    assert registry["ok"] is True
    assert len(registry["datasets"]) == 4


def test_zenodo_download_uses_http_and_sha256sums(tmp_path: Path, monkeypatch) -> None:
    payload = b'{"mint":"abc"}\n'
    dest = tmp_path / "zenodo"
    dest.mkdir()

    content = {
        "launches.jsonl": payload,
    }
    sha = __import__("hashlib").sha256(payload).hexdigest()
    sums = f"{sha}  launches.jsonl\n".encode()
    content["SHA256SUMS"] = sums

    record = {
        "id": 21383616,
        "doi": "10.5281/zenodo.21383616",
        "metadata": {
            "title": "RED-PUMP",
            "version": "1.2",
            "doi": "10.5281/zenodo.21383616",
            "license": {"id": "cc-by-4.0"},
        },
        "files": [
            {
                "key": "launches.jsonl",
                "size": len(payload),
                "checksum": f"md5:{__import__('hashlib').md5(payload).hexdigest()}",
                "links": {"content": "https://zenodo.test/launches.jsonl"},
            },
            {
                "key": "SHA256SUMS",
                "size": len(sums),
                "checksum": f"md5:{__import__('hashlib').md5(sums).hexdigest()}",
                "links": {"content": "https://zenodo.test/SHA256SUMS"},
            },
        ],
    }

    monkeypatch.setattr(
        "trencher.data.zenodo.fetch_zenodo_record",
        lambda record_id, session=None: record,
    )

    def fake_download_http(url, dest_path, **kwargs):
        name = Path(dest_path).name
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(content[name])
        return {
            "path": Path(dest_path),
            "status": "downloaded",
            "size": len(content[name]),
            "sha256": __import__("hashlib").sha256(content[name]).hexdigest(),
            "checksum_ok": True,
        }

    monkeypatch.setattr("trencher.data.zenodo.download_http", fake_download_http)
    meta, files = download_zenodo_record(
        source_id="red_pump_2026",
        record_id="21383616",
        dest_dir=dest,
    )
    assert meta["version"] == "1.2"
    assert any(f["path"] == "launches.jsonl" and f["verified"] for f in files)
    assert (dest / "launches.jsonl").read_bytes() == payload
