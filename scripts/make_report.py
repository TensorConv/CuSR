"""make_report.py — render a structured report.json into a self-contained report.html.

Usage:
    uv run python scripts/make_report.py results/<run>/report.json

Plots referenced in report.json are embedded as base64 (the HTML is standalone,
safe to commit / open anywhere). Schema (all fields optional except title):

{
  "title": str, "date": str,
  "question": str|{"zh","en"}, "setup": str|{"zh","en"},
  "env":     {k: v},                       # git/gpu/driver/clock/...  (English)
  "params":  {section: {k: v}},            # nested; sweep/kernel/synth/...  (English)
  "findings":[{"claim": str|{"zh","en"}, "evidence": str|{"zh","en"}}],
  "tables":  [{"name": str, "columns": [str], "rows": [[...]]}],   # English
  "plots":   [{"file": str, "caption": str|{"zh","en"}}],  # file rel. to report.json dir
  "caveats": [str|{"zh","en"}],
  "reproduce":[str]                        # shell commands (English, copy-paste)
}

Bilingual content: prose fields may be a plain string (rendered as-is,
backward-compatible) OR a {"zh": "...", "en": "..."} object. For an object the
Chinese is the PRIMARY line and the English a SECONDARY, muted, smaller line.
Structural section labels are always bilingual (中文 / English).

Theme: system-adaptive via `color-scheme: light dark` + CSS custom properties,
overridden in `@media (prefers-color-scheme: dark)`. body/surfaces/tables/code
always set an explicit bg AND fg so the report never inherits the host editor's
background with mismatched text. A small fixed toggle lets the reader force a mode.
"""
import base64, json, sys, html
from pathlib import Path

CSS = """
:root{
  color-scheme: light dark;
  --bg:#ffffff; --surface:#f6f8fa; --text:#1f2328; --muted:#57606a;
  --accent:#0969da; --border:#d0d7de; --caveat:#9a6700;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d1117; --surface:#161b22; --text:#e6edf3; --muted:#9198a1;
    --accent:#4493f8; --border:#30363d; --caveat:#d29922;
  }
}
/* Manual override: data-theme on <html> wins over the media query. */
html[data-theme="light"]{
  --bg:#ffffff; --surface:#f6f8fa; --text:#1f2328; --muted:#57606a;
  --accent:#0969da; --border:#d0d7de; --caveat:#9a6700;
}
html[data-theme="dark"]{
  --bg:#0d1117; --surface:#161b22; --text:#e6edf3; --muted:#9198a1;
  --accent:#4493f8; --border:#30363d; --caveat:#d29922;
}
html{background:var(--bg)}
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,
"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
max-width:980px;margin:2rem auto;padding:0 1.2rem;
background:var(--bg);color:var(--text)}
h1{border-bottom:3px solid var(--accent);padding-bottom:.3rem;color:var(--text)}
h2{margin-top:2rem;color:var(--accent);border-bottom:1px solid var(--border);padding-bottom:.2rem}
h3{color:var(--text)}
a{color:var(--accent)}
table{border-collapse:collapse;margin:.6rem 0;font-size:14px;background:var(--bg)}
th,td{border:1px solid var(--border);padding:4px 10px;text-align:right;color:var(--text)}
th{background:var(--surface);text-align:center}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even) td{background:var(--surface)}
code,pre{background:var(--surface);color:var(--text);border-radius:4px}
code{padding:1px 5px}
pre{padding:.8rem 1rem;overflow-x:auto;border:1px solid var(--border)}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 1rem;font-size:13.5px}
.kv div:nth-child(odd){color:var(--muted);font-family:monospace}
.kv div:nth-child(even){color:var(--text)}
.finding{background:var(--surface);border-left:3px solid var(--accent);
padding:.5rem .9rem;margin:.5rem 0;color:var(--text)}
.finding b{color:var(--accent)}
.caveat{color:var(--caveat)}
img{max-width:100%;border:1px solid var(--border);border-radius:4px;margin:.4rem 0}
.cap{font-size:13px;color:var(--muted);margin-bottom:1rem}
small{color:var(--muted)}
/* Secondary (English) line under a primary (Chinese) line. */
.en{display:block;color:var(--muted);font-size:.9em;margin-top:.1rem}
.finding .en{font-weight:normal}
/* Fixed manual light/dark toggle. */
#theme-toggle{position:fixed;top:.7rem;right:.7rem;z-index:99;
background:var(--surface);color:var(--text);border:1px solid var(--border);
border-radius:6px;padding:.3rem .6rem;font:13px/1 inherit;cursor:pointer}
#theme-toggle:hover{border-color:var(--accent)}
"""

# Tiny script: cycle auto -> light -> dark, persist in localStorage.
TOGGLE_JS = """
(function(){
  var KEY='cusr-report-theme', root=document.documentElement;
  var order=['auto','light','dark'], label={auto:'主题跟随系统 / Theme: auto',
    light:'亮色 / Light',dark:'暗色 / Dark'};
  function apply(m){ if(m==='auto'){root.removeAttribute('data-theme');}
    else{root.setAttribute('data-theme',m);} var b=document.getElementById('theme-toggle');
    if(b)b.textContent=label[m]; }
  var cur=localStorage.getItem(KEY)||'auto'; apply(cur);
  document.addEventListener('DOMContentLoaded',function(){
    var b=document.getElementById('theme-toggle'); if(!b)return; apply(cur);
    b.addEventListener('click',function(){
      cur=order[(order.indexOf(cur)+1)%order.length];
      localStorage.setItem(KEY,cur); apply(cur);
    });
  });
})();
"""


def esc(x):
    return html.escape(str(x))


def bilingual(v, en_class="en"):
    """Render a prose value that is either a plain string or {"zh","en"}.

    Plain string -> escaped as-is (backward compatible).
    {"zh","en"}  -> Chinese primary line + English secondary (muted) line.
    Partial dicts (only zh or only en) degrade gracefully.
    """
    if isinstance(v, dict):
        zh = v.get("zh")
        en = v.get("en")
        parts = []
        if zh is not None:
            parts.append(esc(zh))
        if en is not None:
            parts.append(f"<span class='{en_class}'>{esc(en)}</span>")
        return "".join(parts) if parts else ""
    return esc(v)


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
    H = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{esc(R.get('title','report'))}</title>",
        f"<style>{CSS}</style>",
        f"<script>{TOGGLE_JS}</script></head><body>",
        "<button id='theme-toggle' type='button'>主题跟随系统 / Theme: auto</button>",
    ]
    H.append(f"<h1>{esc(R.get('title','(untitled)'))}</h1>")
    if R.get("date"):
        H.append(f"<small>{esc(R['date'])}</small>")
    if R.get("question"):
        H.append(f"<p><b>问题 / Question.</b> {bilingual(R['question'])}</p>")
    if R.get("setup"):
        H.append(f"<p><b>实验设置 / Setup.</b> {bilingual(R['setup'])}</p>")

    if R.get("env"):
        H.append("<h2>环境 / Environment</h2>" + kv_block(R["env"]))

    if R.get("params"):
        H.append("<h2>参数 / Parameters</h2>")
        for section, d in R["params"].items():
            H.append(f"<h3 style='margin:.6rem 0 .2rem'>{esc(section)}</h3>")
            H.append(kv_block(d) if isinstance(d, dict) else f"<p>{esc(d)}</p>")

    if R.get("findings"):
        H.append("<h2>发现 / Findings</h2>")
        for f in R["findings"]:
            claim = bilingual(f.get("claim", ""))
            evidence = bilingual(f.get("evidence", ""))
            H.append(
                "<div class='finding'><b>" + claim + "</b><br>"
                "<span style='color:var(--muted);font-size:.85em'>证据 / evidence:</span> "
                + evidence + "</div>"
            )

    for t in R.get("tables", []):
        H.append(f"<h2>{esc(t.get('name','table'))}</h2><table>")
        H.append("<tr>" + "".join(f"<th>{esc(c)}</th>" for c in t.get("columns", [])) + "</tr>")
        for row in t.get("rows", []):
            H.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        H.append("</table>")

    plots = R.get("plots", [])
    if plots:
        H.append("<h2>图 / Plots</h2>")
    for p in plots:
        fp = base / p["file"]
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode()
            H.append(f"<img src='data:image/png;base64,{b64}'>")
            if p.get("caption"):
                H.append(f"<div class='cap'>{bilingual(p['caption'])}</div>")

    if R.get("caveats"):
        H.append("<h2>注意事项 / Caveats</h2><ul>")
        H += [f"<li class='caveat'>{bilingual(c)}</li>" for c in R["caveats"]]
        H.append("</ul>")

    if R.get("reproduce"):
        H.append("<h2>复现 / Reproduce</h2><pre>"
                 + "\n".join(esc(c) for c in R["reproduce"]) + "</pre>")

    H.append("</body></html>")
    out = base / "report.html"
    out.write_text("".join(H))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
