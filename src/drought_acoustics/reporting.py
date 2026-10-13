"""Self-contained HTML run report (figures embedded, no server needed).

Each training run writes ``report.html`` next to its metrics so a supervisor,
agronomist or marker can review the result by double-clicking one file.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--bg:#fcfcfb;--accent:#2a78d6}
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg);
max-width:1000px;margin:0 auto;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);
padding-bottom:4px}.sub{color:var(--ink2);margin:0 0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--ink2);font-size:12px}
table{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%;overflow-x:auto;display:block}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:8px 0;background:#fff}
code{background:#f1f0ec;padding:1px 4px;border-radius:4px}
"""


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _kpi(value: float, label: str, ci=None) -> str:
    ci_txt = f"<span>95% CI {ci[0]:.3f}-{ci[1]:.3f}</span><br>" if ci else ""
    return f'<div class="kpi"><b>{value:.3f}</b>{ci_txt}<span>{html.escape(label)}</span></div>'


def _table(df: pd.DataFrame) -> str:
    df = df.copy()
    for col in ("n", "clips", "plants", "label"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df.to_html(float_format=lambda v: f"{v:.3f}", border=0)


def write_report(run_dir: Path, metrics: dict, figures: list[Path],
                 tables: dict[str, pd.DataFrame], info: dict) -> Path:
    t = metrics["test"]
    ci = metrics.get("test_ci", {})
    kpis = "".join([
        _kpi(t["macro_f1"], "Macro F1 (test)", ci.get("macro_f1")),
        _kpi(t["roc_auc"], "ROC-AUC (test)", ci.get("roc_auc")),
        _kpi(t["balanced_accuracy"], "Balanced accuracy", ci.get("balanced_accuracy")),
        _kpi(t["recall_drought"], "Drought recall (sensitivity)"),
        _kpi(metrics["plant_level"]["roc_auc"], "Plant-level ROC-AUC"),
    ])
    body = [
        f"<h1>Run report: {html.escape(info['experiment'])}</h1>",
        f"<p class='sub'>Model <code>{html.escape(info['model'])}</code> | dataset "
        f"<code>{info['dataset_fingerprint']}</code> | config <code>{info['config_hash']}</code> | "
        f"git <code>{info['environment'].get('git_commit')}</code> | {info['environment']['timestamp']}</p>",
        f"<div class='kpis'>{kpis}</div>",
    ]
    for name, df in tables.items():
        body.append(f"<h2>{html.escape(name)}</h2>{_table(df)}")
    body.append("<h2>Figures</h2>")
    body += [_img(f) for f in figures if f.exists()]
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(info['experiment'])} report</title><style>{CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")
    out = run_dir / "report.html"
    out.write_text(page)
    return out
