from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

BOT_TOKEN = os.environ["TG_TOKEN"]
CHAT_ID   = os.environ["TG_CHAT"]
LOG_DIR   = sys.argv[1]
TITLE     = sys.argv[2] if len(sys.argv) > 2 else "SFT"
TRAIN_LOG = "/run_out/train.log"

TREND_W       = 10
POLL_INTERVAL = 60
def _tg(method, **kwargs):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        r = requests.post(url, timeout=30, **kwargs)
        return r.json()
    except Exception as e:
        print(f"[tg] {method} failed: {e}", flush=True)
        return {}

def send_text(text: str):
    _tg("sendMessage", json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"})

def send_photo(buf: io.BytesIO, caption: str = ""):
    buf.seek(0)
    _tg("sendPhoto", data={"chat_id": CHAT_ID, "caption": caption},
        files={"photo": ("plot.png", buf, "image/png")})
def max_steps() -> int | None:
    states = sorted(Path(LOG_DIR).glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if states:
        try:
            return int(json.loads(states[-1].read_text())["max_steps"])
        except Exception:
            pass
    try:
        with open(TRAIN_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            tail = f.read().decode("utf-8", "replace")
        totals = [int(t) for t in re.findall(r"\|\s*\d+/(\d+) \[", tail) if int(t) >= 50]
        return totals[-1] if totals else None
    except OSError:
        return None
def read_tb(log_dir: str) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    tb_dir = str(sorted(event_files, key=lambda p: p.stat().st_mtime)[-1].parent)
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {"steps": [e.step for e in events], "values": [e.value for e in events]}
    return data
def _rolling(vals: list, w: int = TREND_W):
    if not vals:
        return None, None
    window = vals[-w:]
    return float(np.mean(window)), float(np.std(window))

def _trend(vals: list, w: int = TREND_W):
    if len(vals) < w * 2:
        return None
    return float(np.mean(vals[-w:]) - np.mean(vals[-w * 2:-w]))

def smooth(vals: list, w: int = 20) -> list:
    arr = np.array(vals, dtype=float)
    cs = np.cumsum(arr)
    cs[w:] = cs[w:] - cs[:-w]
    counts = np.minimum(np.arange(1, len(arr) + 1), w)
    return list(cs / counts)
def analyze(data: dict) -> dict:
    info: dict = {}

    for key in ("train/loss", "train/learning_rate"):
        if key in data and data[key]["steps"]:
            info["step"] = data[key]["steps"][-1]
            break

    if "train/learning_rate" in data:
        lr_vals = data["train/learning_rate"]["values"]
        if lr_vals:
            info["lr"] = lr_vals[-1]
            info["warmup_done"] = lr_vals[-1] >= max(lr_vals) * 0.99 and len(lr_vals) > 5

    for key in ("train/loss", "eval/loss", "train/grad_norm"):
        short = key.split("/")[-1]
        if key in data and data[key]["values"]:
            vals = data[key]["values"]
            info[short] = vals[-1]
            m, s = _rolling(vals)
            if m is not None:
                info[f"{short}_avg"], info[f"{short}_sigma"] = m, s
            info[f"{short}_trend"] = _trend(vals)

    return info
def _plot_ax(ax, data: dict, tag: str, label: str, color: str, yrange=None):
    ax.set_facecolor("#1f2937")
    if tag not in data or not data[tag]["values"]:
        ax.set_visible(False)
        return
    steps  = data[tag]["steps"]
    values = data[tag]["values"]
    if tag == "eval/loss" or len(values) < 15:
        plot_steps = steps[:]
        plot_vals  = values[:]
        if len(plot_vals) >= 2 and plot_vals[0] > plot_vals[1] * 2:
            plot_steps = plot_steps[1:]
            plot_vals  = plot_vals[1:]
        ax.plot(plot_steps, plot_vals, color=color, linewidth=1.8, marker="o", markersize=6)
        if plot_vals:
            vmin, vmax = min(plot_vals), max(plot_vals)
            margin = max((vmax - vmin) * 0.4, 0.03)
            ax.set_ylim(vmin - margin, vmax + margin)
    else:
        sm = smooth(values)
        ax.plot(steps, values, color=color, linewidth=0.5, alpha=0.25)
        ax.plot(steps, sm,     color=color, linewidth=2.0)
        if not yrange and len(values) > 50:
            p5  = float(np.percentile(values, 5))
            p95 = float(np.percentile(values, 95))
            if max(values) > p95 * 1.5:
                margin = max((p95 - p5) * 0.3, 0.05)
                ax.set_ylim(p5 - margin, p95 + margin)
    if yrange:
        ax.set_ylim(yrange)
    last = values[-1]
    ax.axhline(last, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(0.97, 0.93, f"{last:.4g}", transform=ax.transAxes,
            ha="right", va="top", color="#fbbf24", fontsize=12, fontweight="bold")
    ax.set_title(label, color="white", fontsize=13, pad=6)
    ax.tick_params(colors="#9ca3af", labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#374151")


def make_plot(data: dict, step: int, total: int | None) -> io.BytesIO | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#111827")
    fig.suptitle(f"{TITLE} — step {step}/{total or '?'}", color="white", fontsize=15, y=1.005)

    _plot_ax(axes[0][0], data, "train/loss",       "Train Loss  [↓]",      "#4fc3f7")
    _plot_ax(axes[0][1], data, "eval/loss",        "Eval Loss  [↓]",       "#f97316")
    _plot_ax(axes[1][0], data, "train/learning_rate", "Learning Rate",      "#66bb6a")
    _plot_ax(axes[1][1], data, "train/grad_norm",  "Grad Norm  [стабильно]", "#a78bfa")

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf
def format_step_line(data: dict, step: int, total: int | None, elapsed_h: float) -> str:
    def at(key):
        d = data.get(key, {})
        steps, vals = d.get("steps", []), d.get("values", [])
        if step in steps:
            return vals[steps.index(step)]
        return vals[-1] if vals else None

    loss = at("train/loss")
    gn   = at("train/grad_norm")
    lr   = at("train/learning_rate")

    gn_icon = "🔴" if gn is not None and (not np.isfinite(gn) or gn > 10) else ""

    if total:
        eta_h = (total - step) * (elapsed_h / max(step, 1))
        parts = [f"<code>#{step:>4}/{total}</code> ({step / total * 100:.0f}%, ETA ~{eta_h:.1f}h)"]
    else:
        parts = [f"<code>#{step:>4}</code>"]
    if loss is not None:
        parts.append(f"loss <code>{loss:.4f}</code>")
    if lr is not None:
        parts.append(f"lr <code>{lr:.1e}</code>")
    if gn is not None:
        parts.append(f"{gn_icon}gn <code>{gn:.3f}</code>")
    return "  ".join(parts)


def format_eval_line(data: dict) -> str:
    ev = data.get("eval/loss", {})
    steps, vals = ev.get("steps", []), ev.get("values", [])
    if not vals:
        return ""
    cur_step, cur_val = steps[-1], vals[-1]

    if len(vals) >= 2:
        prev_val = vals[-2]
        delta = cur_val - prev_val
        arrow = "⬇️" if delta < -0.01 else ("⬆️" if delta > 0.01 else "➡️")
        delta_str = f"  {arrow} <code>{delta:+.4f}</code> от step {steps[-2]}"
    else:
        delta_str = ""

    best = min(vals)
    best_str = f"\nлучший <code>{best:.4f}</code> @ step {steps[vals.index(best)]}"

    plateau = ""
    if len(vals) >= 4:
        deltas = [abs(vals[i] - vals[i-1]) for i in range(-3, 0)]
        if all(d < 0.01 for d in deltas):
            plateau = "\n⚠️ <b>Плато</b> — eval не двигается 3 точки подряд"

    return f"📊 <b>Eval @ step {cur_step}</b>  loss = <code>{cur_val:.4f}</code>{delta_str}{best_str}{plateau}"
def main():
    print(f"[tg-monitor] started, watching {LOG_DIR}", flush=True)
    last_sent_step = -1
    last_eval_sent = 0
    start_time = time.time()

    while True:
        try:
            time.sleep(POLL_INTERVAL)

            try:
                data = read_tb(LOG_DIR)
            except Exception as e:
                print(f"[tg-monitor] TB read error: {e}", flush=True)
                continue

            if not data:
                continue

            info = analyze(data)
            current_step = info.get("step", 0)
            total = max_steps()
            elapsed_h = (time.time() - start_time) / 3600

            if isinstance(current_step, int) and current_step > last_sent_step:
                if "train/loss" in data:
                    new_steps = [s for s in data["train/loss"]["steps"] if s > last_sent_step]
                    for s in new_steps:
                        send_text(format_step_line(data, s, total, elapsed_h))
                        print(f"[tg-monitor] step {s}", flush=True)
                        time.sleep(0.5)
                    last_sent_step = current_step

            ev_steps = data.get("eval/loss", {}).get("steps", [])
            new_evals = [s for s in ev_steps if s > last_eval_sent and s > 0]
            for es in new_evals:
                eval_line = format_eval_line(data)
                if eval_line:
                    send_text(eval_line)
                buf = make_plot(data, current_step if isinstance(current_step, int) else es, total)
                if buf:
                    send_photo(buf, caption=f"{TITLE} eval @ step {es}")
                last_eval_sent = es
                time.sleep(0.5)

        except Exception as e:
            print(f"[tg-monitor] CRASH: {e}", flush=True)
            send_text(f"⚠️ Monitor error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
