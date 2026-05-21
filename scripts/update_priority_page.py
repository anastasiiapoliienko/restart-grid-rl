"""After priority training+eval+trace finish, update the /priority/ page
with real numbers and the actual trace JSON, then drop the PREVIEW badge.
"""
import json
import re
import statistics
from pathlib import Path

REPO = Path("/Users/anastasiiapoliienko/my-project/restart-grid-rl")
DEPLOY = Path("/Users/anastasiiapoliienko/my-project/restart-grid-deploy")

EVAL_TXT = REPO / "results/priority_eval.txt"
TRACE_JSON = REPO / "results/trace_priority.json"
PAGE = DEPLOY / "priority/index.html"


def parse_eval(text: str):
    """Returns dict like {greedy: {...}, sequential: {...}, priority: [{...}, {...}, {...}]}."""
    out = {"greedy": None, "sequential": None, "priority": []}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("-") or line.lower().startswith("policy") or line.startswith("Held-out"):
            continue
        # Match policy name first
        is_greedy = line.startswith("greedy")
        is_seq    = line.startswith("sequential")
        is_prio   = line.startswith("priority")
        if not (is_greedy or is_seq or is_prio):
            continue
        # Split fields by whitespace, ignoring leading name part
        parts = re.split(r"\s{2,}", line)
        # Header columns: policy, trip%, restore%, hosp%, min→hosp, min→crit, steps, peak A
        # But split might fold name + number cells; safer:
        # Take last 7 numeric/dash tokens
        toks = line.split()
        # Find first token that's a number — that's where the metrics start
        for i, tok in enumerate(toks):
            if re.match(r"^[\d.]+$", tok) or tok == "—":
                start = i
                break
        else:
            continue
        cells = toks[start:]
        if len(cells) < 7:
            continue
        def num(c):
            try: return float(c)
            except Exception: return None
        rec = {
            "trip":     num(cells[0]),
            "restore":  num(cells[1]),
            "hosp_pct": num(cells[2]),
            "hosp_min": num(cells[3]),
            "crit_min": num(cells[4]),
            "steps":    num(cells[5]),
            "peak":     num(cells[6]),
        }
        if is_greedy:    out["greedy"]    = rec
        elif is_seq:     out["sequential"] = rec
        elif is_prio:    out["priority"].append(rec)
    return out


def fmt(v, suffix="", default="—"):
    if v is None: return default
    return f"{v:.1f}{suffix}"


def aggregate_priority(prio_rows):
    keys = ["trip", "restore", "hosp_pct", "hosp_min", "crit_min", "steps", "peak"]
    out = {}
    for k in keys:
        vals = [r[k] for r in prio_rows if r[k] is not None]
        out[k] = statistics.mean(vals) if vals else None
        out[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    text = EVAL_TXT.read_text()
    parsed = parse_eval(text)
    g, s, plist = parsed["greedy"], parsed["sequential"], parsed["priority"]
    if not plist:
        print("no priority rows parsed; aborting")
        return 1
    p = aggregate_priority(plist)

    # Speedup vs sequential on minutes-to-hospital
    if s and s["hosp_min"] and p["hosp_min"]:
        speedup = s["hosp_min"] / p["hosp_min"]
        speedup_str = f"{speedup:.1f}×"
    else:
        speedup_str = "—"

    html = PAGE.read_text()

    # ── Replace hero numbers (idempotent: matches any current value) ──
    # Card 1: minutes-to-hospital
    html = re.sub(
        r'(<div class="v">)[^<]+(</div><div class="k">[^<]*hospital online)',
        lambda m: m.group(1) + fmt(p["hosp_min"]) + m.group(2),
        html, count=1,
    )
    # Card 2: minutes-to-critical
    html = re.sub(
        r'(<div class="v">)[^<]+(</div><div class="k">[^<]*every critical)',
        lambda m: m.group(1) + fmt(p["crit_min"]) + m.group(2),
        html, count=1,
    )
    # Card 3: speedup
    html = re.sub(
        r'(<div class="v g">)[^<]+(</div><div class="k">[^<]*faster to hospital)',
        lambda m: m.group(1) + speedup_str + m.group(2),
        html, count=1,
    )

    # ── Replace table cells (idempotent: matches data-pending OR data-filled) ──
    def cell_html(pending, val_str):
        return f'class="num" data-filled="{pending}">{val_str}'
    repls = {
        "g.hosp":  fmt(g["hosp_min"]) + " min" if g and g.get("hosp_min") else "—",
        "g.crit":  fmt(g["crit_min"]) + " min" if g and g.get("crit_min") else "—",
        "g.steps": fmt(g["steps"])             if g and g.get("steps") else "—",
        "g.trip":  fmt(g["trip"], "%")         if g else "—",
        "s.hosp":  fmt(s["hosp_min"]) + " min" if s and s.get("hosp_min") else "—",
        "s.crit":  fmt(s["crit_min"]) + " min" if s and s.get("crit_min") else "—",
        "s.steps": fmt(s["steps"])             if s and s.get("steps") else "—",
        "s.trip":  fmt(s["trip"], "%")         if s else "—",
        "p.hosp":  (fmt(p["hosp_min"]) + " ± " + fmt(p["hosp_min_std"]) + " min") if p["hosp_min"] is not None else "—",
        "p.crit":  (fmt(p["crit_min"]) + " ± " + fmt(p["crit_min_std"]) + " min") if p["crit_min"] is not None else "—",
        "p.steps": (fmt(p["steps"]) + " ± " + fmt(p["steps_std"])) if p["steps"] is not None else "—",
        "p.trip":  fmt(p["trip"], "%"),
    }
    for k, v in repls.items():
        # idempotent: match either data-pending="…">— or data-filled="…">any-value
        html = re.sub(
            rf'class="num" data-(?:pending|filled)="{re.escape(k)}">[^<]+',
            cell_html(k, v),
            html,
        )

    # ── Inline the trace JSON (placeholder substitution; first run only) ──
    if TRACE_JSON.exists():
        trace = TRACE_JSON.read_text().strip()
        html = html.replace("__PRIORITY_TRACE_JSON__", trace, 1)
    else:
        print("trace JSON missing; animation will remain placeholder")

    # ── Re-inline the scenarios bundle (idempotent: replaces whatever is there) ──
    BUNDLE = DEPLOY / "priority/traces.json"
    if BUNDLE.exists():
        bundle = BUNDLE.read_text().strip()
        new_html, n = re.subn(
            r'(<script id="scenarios-data" type="application/json">)[^<]*(</script>)',
            lambda m: m.group(1) + bundle + m.group(2),
            html, count=1, flags=re.DOTALL,
        )
        if n:
            html = new_html
            print(f"  re-inlined scenarios bundle ({len(bundle)} bytes)")
        else:
            print("  scenarios-data block not found (probably new layout)")

    # ── Remove the PREVIEW badge ──
    html = re.sub(
        r'\n\s*<div style="display:inline-flex;align-items:center;gap:10px;background:var\(--paper\);border:1px solid var\(--line\);padding:6px 14px;[^>]*">\s*\n\s*<span[^>]*>PREVIEW</span>.*?</div>\n',
        "\n",
        html, count=1, flags=re.DOTALL,
    )

    PAGE.write_text(html)
    print("updated:", PAGE)
    print(f"  hospital: {fmt(p['hosp_min'])} min (priority) vs {fmt(s['hosp_min']) if s else '—'} (seq-d8) → speedup {speedup_str}")
    print(f"  critical: {fmt(p['crit_min'])} min")
    print(f"  steps: {fmt(p['steps'])} (priority)")
    print(f"  trip rate: {fmt(p['trip'], '%')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
