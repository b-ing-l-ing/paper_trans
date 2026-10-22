"""输出层 —— 数据集转 DataFrame / CSV。

records → pandas DataFrame → CSV，交给下游 CTGAN 训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import Schema


def to_dataframe(
    records: list[dict[str, Any]],
    schema: Schema,
    new_fields: list | None = None,
) -> pd.DataFrame:
    """把 records 转成 DataFrame。

    列顺序：schema 字段（按字段库顺序）→ 新字段（按出现顺序追加）。
    缺失值由 pandas 自动置为 NaN。
    """
    columns = schema.field_names()
    if new_fields:
        for nf in new_fields:
            if nf.name not in columns:
                columns.append(nf.name)
    return pd.DataFrame(records, columns=columns)


def save_csv(df: pd.DataFrame, path: str | Path) -> str:
    """DataFrame 存 CSV。

    用 UTF-8-SIG 编码（带 BOM），Excel 打开中文不乱码。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
