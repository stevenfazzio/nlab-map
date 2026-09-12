"""Shared helpers for the pipeline stages."""

import os
import stat
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path("data")
RAW = DATA / "raw"


def write_parquet_safely(df, output_path):
    """Write df so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path)
        if output_path.exists():
            os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_path, 0o666 & ~umask)
        meta = pq.read_metadata(tmp_path)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        assert set(meta.schema.names) >= set(df.columns), (
            f"columns missing on disk: {set(df.columns) - set(meta.schema.names)}"
        )
        os.replace(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
    print(f"wrote {output_path} ({meta.num_rows} rows, {len(df.columns)} cols)")


def validate_stage_output(df, stage_name, expected_columns):
    missing = set(expected_columns) - set(df.columns)
    assert not missing, f"{stage_name}: missing columns {missing}"
    assert len(df) > 0, f"{stage_name}: output is empty"
    print(f"{stage_name}: {len(df)} rows, {len(df.columns)} columns")
