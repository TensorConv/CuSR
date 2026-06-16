"""make_report.py — render a structured report.json into a self-contained report.html.

Usage:
    uv run python scripts/make_report.py results/<run>/report.json

Plots referenced in report.json are embedded as base64 (the HTML is standalone,
safe to commit / open anywhere). Schema (all fields optional except title):

{
  "title": str, "date": str, "question": str, "setup": str,
  "env":     {k: v},                       # git/gpu/driver/clock/...
  "params":  {section: {k: v}},            # nested; sweep/kernel/synth/...
  "findings":[{"claim": str, "evidence": str}],
  "tables":  [{"name": str, "columns": [str], "rows": [[...]]}],
  "plots":   [{"file": str, "caption": str}],   # file relative to report.json dir
  "caveats": [str],
  "reproduce":[str]                        # shell commands
}
"""
import base64, json, sys, html
from pathlib import Path

CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;
margin:2rem auto;padding:0 1.2rem;color:#1a1a1a}
h1{border-bottom:3px solid #2c6e49;padding-bottom:.3rem}
h2{margin-top:2rem;color:#2c6e49;border-bottom:1px solid #ddd;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.6rem 0;font-size:14px}
th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}th{background:#f0f5f2;text-align:center}
td:first-child,th:first-child{text-align:left}
code,pre{background:#f5f5f5;border-radius:4px}code{padding:1px 5px}
pre{padding:.8rem 1rem;overflow-x:auto;border:1px solid #e0e0e0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 1rem;font-size:13.5px}
.kv div:nth-child(odd){color:#555;font-family:monospace}
.finding{background:#f7faf8;border-left:3px solid #2c6e49;padding:.5rem .9rem;margin:.5rem 0}
.finding b{color:#2c6e49} .caveat{color:#8a5a00}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:.4rem 0}
.cap{font-size:13px;color:#666;margin-bottom:1rem}
small{color:#888}
"""

def esc(x): return html.escape(str(x))

def kv_block(d):
    out = ['<div class="kv">']
    for k, v in d.items():
        out.append(f"<div>{esc(k)}</div><div>{esc(v)}</div>")
    out.append("</div>")
    return "".join(out)

def main():
    jpath = Path(sys.argv[1])
    R = json.loads(jpath.read_text())
    base = jpath.parent
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(R.get('title','report'))}</title>",
         f"<style>{CSS}</style></head><body>"]
    H.append(f"<h1>{esc(R.get('title','(untitled)'))}</h1>")
    if R.get("date"): H.append(f"<small>{esc(R['date'])}</small>")
    if R.get("question"): H.append(f"<p><b>Question.</b> {esc(R['question'])}</p>")
    if R.get("setup"): H.append(f"<p><b>Setup.</b> {esc(R['setup'])}</p>")

    if R.get("env"):
        H.append("<h2>Environment</h2>" + kv_block(R["env"]))

    if R.get("params"):
        H.append("<h2>Parameters</h2>")
        for section, d in R["params"].items():
            H.append(f"<h3 style='margin:.6rem 0 .2rem'>{esc(section)}</h3>")
            H.append(kv_block(d) if isinstance(d, dict) else f"<p>{esc(d)}</p>")

    if R.get("findings"):
        H.append("<h2>Findings</h2>")
        for f in R["findings"]:
            H.append(f"<div class='finding'><b>{esc(f.get('claim',''))}</b><br>{esc(f.get('evidence',''))}</div>")

    for t in R.get("tables", []):
        H.append(f"<h2>{esc(t.get('name','table'))}</h2><table>")
        H.append("<tr>" + "".join(f"<th>{esc(c)}</th>" for c in t.get("columns", [])) + "</tr>")
        for row in t.get("rows", []):
            H.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        H.append("</table>")

    for p in R.get("plots", []):
        fp = base / p["file"]
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode()
            H.append(f"<img src='data:image/png;base64,{b64}'>")
            if p.get("caption"): H.append(f"<div class='cap'>{esc(p['caption'])}</div>")

    if R.get("caveats"):
        H.append("<h2>Caveats</h2><ul>")
        H += [f"<li class='caveat'>{esc(c)}</li>" for c in R["caveats"]]
        H.append("</ul>")

    if R.get("reproduce"):
        H.append("<h2>Reproduce</h2><pre>" + "\n".join(esc(c) for c in R["reproduce"]) + "</pre>")

    H.append("</body></html>")
    out = base / "report.html"
    out.write_text("".join(H))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
