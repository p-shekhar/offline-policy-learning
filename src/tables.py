from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame
