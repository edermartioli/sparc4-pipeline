"""
sparc4_plot_raw_lc.py
=====================
SPARC4 raw photometry light curve monitor — ZeroMQ subscriber.

Plots Δmag = −2.5 log10(flux / flux₀) vs local time for the target
and each comparison star, one panel per channel.

Target and comparison assignments are read from channel 1's database
(the authoritative source for cross-channel consistent ref_ids).

Usage
-----
::

    python sparc4_plot_raw_lc.py --db /data/SPARC4/reduced
    python sparc4_plot_raw_lc.py --host 192.168.1.10 --db /data/SPARC4/reduced

Options
-------
--host HOST         Publisher host                  [default: localhost]
--port PORT         Publisher ZeroMQ port           [default: 5556]
--channels LIST     Comma-separated channel list    [default: 1,2,3,4]
--db DIR            Directory with monitor_chN.db   [default: none]
--history N         Maximum data points per source  [default: 300]
--interval SEC      Plot refresh interval (s)       [default: 2.0]
--utc-offset HOURS  Local UTC offset (e.g. -3)     [default: -3]

Author
------
Eder Martioli <martioli@lna.br>
Laboratório Nacional de Astrofísica — LNA/MCTI
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as _mticker

try:
    import zmq
except ImportError:
    raise SystemExit("pyzmq is required:  pip install pyzmq")

try:
    from sparc4_astro_utils import (compute_sun_events, add_twilight_lines,
                                     night_xlim, date_from_jd,
                                     OPD_LON, OPD_LAT, OPD_ALT)
    _HAS_ASTRO = True
except ImportError:
    _HAS_ASTRO = False

CH_COLOR     = {1: "darkblue", 2: "darkgreen", 3: "darkorange", 4: "darkred"}
CH_BAND      = {1: "g",        2: "r",         3: "i",          4: "z"}
TARGET_COLOR = "green"
COMP_COLORS  = ["red", "darkorange", "purple", "saddlebrown", "crimson", "teal"]


def _jd_to_hours(jd_arr: np.ndarray, utc_offset_h: float) -> np.ndarray:
    if len(jd_arr) == 0: return np.array([])
    return ((jd_arr + 0.5 + utc_offset_h / 24.0) % 1.0) * 24.0

def _format_time_axis(ax) -> None:
    def _fmt(x, pos):
        x = x % 24.0; h = int(x); m = int(round((x-h)*60)) % 60
        return f"{h:02d}:{m:02d}"
    ax.xaxis.set_major_formatter(_mticker.FuncFormatter(_fmt))
    ax.xaxis.set_major_locator(_mticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(_mticker.MultipleLocator(1/12))
    ax.tick_params(axis="x", which="minor", length=3)

def _sync_xlim(axes_list: list) -> None:
    """Sync x-axis limits across panels that have real time-series data.

    Ignores empty panels and reference lines (axhline) so their default
    xlim does not corrupt the shared scale.  Adds a 5-minute margin.
    """
    mins, maxs = [], []
    for ax in axes_list:
        # Only count lines that have actual time-series data (more than 2 pts
        # and not a flat reference line added by axhline)
        data_lines = [
            line for line in ax.get_lines()
            if len(line.get_xdata()) > 2
            and not (line.get_linestyle() in (':', '--', 'dotted', 'dashed')
                     and len(set(line.get_ydata())) == 1)
        ]
        if not data_lines:
            continue
        xl = ax.get_xlim()
        if xl[1] > xl[0]:
            mins.append(xl[0]); maxs.append(xl[1])
    if not mins:
        return
    margin = 5 / 60  # 5 minutes in decimal hours
    for ax in axes_list:
        ax.set_xlim(min(mins) - margin, max(maxs) + margin)

class _Buf:
    """Circular flux buffer; computes Δmag relative to first measurement."""
    def __init__(self, maxlen: int) -> None:
        self.jd:    Deque[float] = deque(maxlen=maxlen)
        self.flux:  Deque[float] = deque(maxlen=maxlen)
        self.flux0: Optional[float] = None

    def add(self, jd: float, flux: float) -> bool:
        if not (np.isfinite(flux) and flux > 0): return False
        if self.flux0 is None: self.flux0 = flux
        self.jd.append(jd); self.flux.append(flux)
        return True

    def dmag(self) -> np.ndarray:
        if self.flux0 is None or len(self.flux) == 0: return np.array([])
        with np.errstate(invalid="ignore", divide="ignore"):
            dm = -2.5 * np.log10(np.array(self.flux) / self.flux0)
        dm[~np.isfinite(dm)] = np.nan
        return dm

    def jd_arr(self) -> np.ndarray: return np.array(self.jd)
    def __len__(self) -> int: return len(self.jd)


def _load_channel_data(db_path: str, history: int) -> tuple:
    """Load selection and flux history from one channel's own DB.

    In POLAR mode, beam-0 and beam-1 fluxes for the same star are summed so
    the returned data always has one entry per star keyed by the beam-0 ref_id.

    Returns (target_id, comp_ids, beam_map, {ref_id: (jd_arr, flux_arr)}).
    """
    if not os.path.isfile(db_path):
        return None, [], {}, {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        sel_rows = conn.execute(
            "SELECT role, ref_id, beam FROM selection ORDER BY rank"
        ).fetchall()
        target_id, comp_ids, beam_map = None, [], {}
        for row in sel_rows:
            rid  = row["ref_id"]
            beam = row["beam"]
            if beam is not None:
                beam_map[rid] = beam
            if row["role"] == "target": target_id = rid
            else: comp_ids.append(rid)

        is_polar = bool(beam_map)
        all_ids  = ([target_id] if target_id is not None else []) + comp_ids

        if is_polar:
            b0_ids = [r for r in all_ids if beam_map.get(r) == 0]
            b1_ids = [r for r in all_ids if beam_map.get(r) == 1]
            b1_for_b0 = {b0: b1_ids[i] for i, b0 in enumerate(b0_ids)
                         if i < len(b1_ids)}
            data = {}
            for b0 in b0_ids:
                b1 = b1_for_b0.get(b0)
                pts0 = conn.execute(
                    "SELECT f.jd, s.flux FROM sources s "
                    "JOIN frames f ON f.id = s.frame_id "
                    "WHERE s.ref_id = ? AND f.jd IS NOT NULL "
                    "  AND f.bad = 0 AND s.flux IS NOT NULL AND s.flux > 0 "
                    "ORDER BY f.jd", (b0,)
                ).fetchall()[-history:]
                if not pts0: continue
                jds0 = np.array([r["jd"]   for r in pts0])
                fl0  = np.array([r["flux"] for r in pts0])
                if b1 is not None:
                    pts1 = conn.execute(
                        "SELECT f.jd, s.flux FROM sources s "
                        "JOIN frames f ON f.id = s.frame_id "
                        "WHERE s.ref_id = ? AND f.jd IS NOT NULL "
                        "  AND f.bad = 0 AND s.flux IS NOT NULL AND s.flux > 0 "
                        "ORDER BY f.jd", (b1,)
                    ).fetchall()[-history:]
                    jds1 = np.array([r["jd"]   for r in pts1])
                    fl1  = np.array([r["flux"] for r in pts1])
                    summed_jds, summed_fl = [], []
                    for jd, f0 in zip(jds0, fl0):
                        if len(jds1):
                            idx = int(np.argmin(np.abs(jds1 - jd)))
                            if abs(jds1[idx] - jd) < 1/86400:
                                summed_jds.append(jd)
                                summed_fl.append(f0 + fl1[idx])
                    data[b0] = (np.array(summed_jds), np.array(summed_fl))                                if summed_jds else (jds0, fl0)
                else:
                    data[b0] = (jds0, fl0)
            target_id = b0_ids[0] if b0_ids else target_id
            comp_ids  = b0_ids[1:]
        else:
            data = {}
            for rid in all_ids:
                pts = conn.execute(
                    "SELECT f.jd, s.flux FROM sources s "
                    "JOIN frames f ON f.id = s.frame_id "
                    "WHERE s.ref_id = ? AND f.jd IS NOT NULL "
                    "  AND f.bad = 0 AND s.flux IS NOT NULL AND s.flux > 0 "
                    "ORDER BY f.jd", (rid,)
                ).fetchall()[-history:]
                if pts:
                    data[rid] = (np.array([r["jd"]   for r in pts]),
                                 np.array([r["flux"] for r in pts]))
        conn.close()
        n = sum(len(v[0]) for v in data.values())
        mode = "POLAR" if is_polar else "PHOT"
        print(f"  {os.path.basename(db_path)} [{mode}]: "
              f"target={target_id}  comps={comp_ids}  pts={n}")
        return target_id, comp_ids, beam_map, data
    except Exception as exc:
        print(f"  [warn] {db_path}: {exc}")
        return None, [], {}, {}

def run(host: str, port: int, channels: List[int],
        db_dir: Optional[str], history: int, interval: float,
        utc_offset: float,
        twilight: bool = False,
        night_xlim_flag: bool = False,
        obs_lon: float = OPD_LON,
        obs_lat: float = OPD_LAT,
        obs_alt: float = OPD_ALT,
        obs_date: str = "") -> None:

    ref_channel = channels[0]

    # Per-channel selections — each channel has its own ref_ids
    per_ch_target: Dict[int, Optional[int]] = {ch: None for ch in channels}
    per_ch_comps:  Dict[int, List[int]]     = {ch: []   for ch in channels}

    bufs: Dict[int, Dict[int, _Buf]] = {ch: {} for ch in channels}

    def _ensure(ch: int, rid: int) -> None:
        if rid not in bufs[ch]:
            bufs[ch][rid] = _Buf(history)

    if db_dir:
        print("[raw_lc] Loading historical data...")
        for ch in channels:
            path = os.path.join(db_dir, f"monitor_ch{ch}.db")
            t, comps, _bm, data = _load_channel_data(path, history)
            per_ch_target[ch] = t
            per_ch_comps[ch]  = comps
            for rid, (jds, fluxes) in data.items():
                _ensure(ch, rid)
                for jd, fl in zip(jds, fluxes):
                    bufs[ch][rid].add(jd, fl)

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    for ch in channels:
        sock.setsockopt_string(zmq.SUBSCRIBE, f"sparc4.ch{ch}")
    sock.setsockopt(zmq.RCVTIMEO, int(interval * 1000))
    print(f"[raw_lc] Subscribed  tcp://{host}:{port}  ch={channels}")

    nch     = len(channels)
    panel_h = min(3.0, 9.0 / nch)
    fig, axs = plt.subplots(nch, 1, figsize=(13, panel_h*nch), squeeze=False)
    axes: Dict[int, plt.Axes] = {ch: axs[i][0] for i,ch in enumerate(channels)}

    fig.patch.set_facecolor("white")
    fig.suptitle("SPARC4  —  Raw Light Curves  (Δmag from first frame)",
                 fontsize=14, fontweight="bold", color="black", y=0.99)
    fig.canvas.manager.set_window_title("SPARC4 — Raw LC")

    for idx, ch in enumerate(channels):
        ax = axes[ch]
        ax.set_facecolor("white")
        ax.set_ylabel("Δ mag", fontsize=12)
        ax.invert_yaxis()
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.grid(True, color="lightgray", ls="--", lw=0.6)
        for spine in ax.spines.values(): spine.set_edgecolor("gray")
        ax.tick_params(colors="black", labelsize=11)
        ax.set_title(f"{CH_BAND[ch]}  (ch{ch})", fontsize=12,
                     fontweight="bold", color=CH_COLOR[ch], loc="left")
        _format_time_axis(ax)
        if idx == nch - 1:
            ax.set_xlabel(f"Local time  (UTC{utc_offset:+.0f}h)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    lines: Dict[int, Dict[int, plt.Line2D]] = {ch: {} for ch in channels}
    last_draw = 0.0
    new_data  = False

    plt.ion()
    fig.show()

    def _redraw() -> None:
        for ch in channels:
            tid  = per_ch_target[ch]
            cids = per_ch_comps[ch]
            ordered = ([tid] if tid is not None else []) + \
                      [r for r in cids if r != tid]
            ax = axes[ch]
            if not ordered:
                ax.set_title(f"{CH_BAND[ch]}  (ch{ch})  — waiting for selection",
                             fontsize=11, fontweight="bold",
                             color=CH_COLOR[ch], loc="left")
                continue

            latest = []
            for rid in ordered:
                if rid not in bufs[ch]: continue
                buf = bufs[ch][rid]
                dm  = buf.dmag()
                jds = buf.jd_arr()
                if len(dm) == 0: continue
                ht = _jd_to_hours(jds, utc_offset)

                if rid == tid:
                    col, marker, lw, ms, mfc = TARGET_COLOR, "o", 2.0, 7, TARGET_COLOR
                    lbl = f"Target  (src {rid})"
                else:
                    ci = cids.index(rid)
                    col = COMP_COLORS[ci % len(COMP_COLORS)]
                    marker, lw, ms, mfc = "s", 1.5, 5, "white"
                    lbl = f"C{ci+1}  (src {rid})"

                if rid not in lines[ch]:
                    l, = ax.plot(ht, dm, marker=marker, ls="-", color=col,
                                 markerfacecolor=mfc, markeredgecolor=col,
                                 lw=lw, ms=ms, zorder=3, label=lbl)
                    lines[ch][rid] = l
                else:
                    lines[ch][rid].set_data(ht, dm)

                if len(dm) and np.isfinite(dm[-1]):
                    pfx = "T" if rid == tid else f"C{cids.index(rid)+1}"
                    latest.append(f"{pfx}={dm[-1]:+.3f}")

            ax.relim(); ax.autoscale_view()
            handles, lbls = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, lbls, loc="upper left", fontsize=10,
                          framealpha=0.9, edgecolor="gray")
            suffix = ("  |  " + "  ".join(latest)) if latest else ""
            ax.set_title(f"{CH_BAND[ch]}  (ch{ch}){suffix}",
                         fontsize=11, fontweight="bold",
                         color=CH_COLOR[ch], loc="left")

        _sync_xlim(list(axes.values()))
        if _HAS_ASTRO and twilight and sun_evts is not None:
            for _ax in axes.values():
                for _l in list(_ax.get_lines()):
                    if getattr(_l, "_is_twilight", False):
                        _l.remove()
            add_twilight_lines(list(axes.values()), sun_evts,
                               legend_ax_index=0)
            for _ax in axes.values():
                for _l in _ax.get_lines():
                    if _l.get_label() in ("Sunset","Sunrise",
                        "Civil twil.","Nautical twil.","Astron. twil."):
                        _l._is_twilight = True
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # ── Connect xlim_changed callbacks for interactive pan/zoom sync ──────
    # When the user pans or zooms any panel, all other panels follow.
    # The _syncing guard prevents infinite callback recursion.
    _syncing = [False]

    def _on_xlim_changed(ax_changed):
        if _syncing[0]:
            return
        _syncing[0] = True
        try:
            xl = ax_changed.get_xlim()
            for ax in axes.values():
                if ax is not ax_changed:
                    ax.set_xlim(xl)
            fig.canvas.draw_idle()
        finally:
            _syncing[0] = False

    for _ax in axes.values():
        _ax.callbacks.connect("xlim_changed", _on_xlim_changed)

    # ── Sun events / twilight ─────────────────────────────────────────────
    sun_evts = None
    if _HAS_ASTRO and (twilight or night_xlim_flag):
        _jd0 = next(
            (jd for ch_bufs in bufs.values()
             for buf in ch_bufs.values()
             for jd in (list(buf.jd) if hasattr(buf, "jd") else list(buf))
             if np.isfinite(jd)), None)
        _date = obs_date or (date_from_jd(_jd0, utc_offset) if _jd0 else "")
        if not _date:
            from datetime import date as _d
            _date = _d.today().isoformat()
        sun_evts = compute_sun_events(
            _date, obs_lon, obs_lat, obs_alt, utc_offset)
        if night_xlim_flag and sun_evts is not None:
            xlo, xhi = night_xlim(sun_evts)
            if xlo is not None:
                for _ax in axes.values():
                    _ax.set_xlim(xlo, xhi)
                print(f"[raw_lc] Night x-lim: {xlo:.3f} – {xhi:.3f} h")

    _redraw()

    try:
        while True:
            try:
                parts = sock.recv_multipart()
                msg   = json.loads(parts[1].decode())
            except zmq.Again:
                if new_data and time.time() - last_draw >= interval:
                    _redraw(); last_draw = time.time(); new_data = False
                time.sleep(0.05); fig.canvas.flush_events(); continue

            ch = msg.get("channel")
            if ch not in channels: continue
            jd      = msg.get("jd")
            sources = msg.get("sources", [])
            if jd is None or not sources: continue

            # Update this channel's own selection from its message
            msg_target = msg.get("target_id")
            msg_comps  = msg.get("comp_ids") or []
            if msg_target is not None:
                if (per_ch_target[ch] != msg_target
                        or per_ch_comps[ch] != list(msg_comps)):
                    per_ch_target[ch] = msg_target
                    per_ch_comps[ch]  = list(msg_comps)
                    print(f"[raw_lc] ch{ch}: "
                          f"target={msg_target}  comps={msg_comps}")

            tid  = per_ch_target[ch]
            cids = per_ch_comps[ch]

            # Build flux dict; in POLAR mode sum beam-0 + beam-1 per star.
            inst_mode  = msg.get("inst_mode", "PHOT")
            is_polar   = (inst_mode == "POLAR")
            flux_by_ref: dict = {}   # logical ref_id -> total flux
            if is_polar:
                # Group by logical star: beam-0 ref_id is canonical
                b0_flux = {s["ref_id"]: s["flux"] for s in sources
                           if s.get("beam") == 0 and s.get("flux") is not None}
                b1_flux = {s["ref_id"]: s["flux"] for s in sources
                           if s.get("beam") == 1 and s.get("flux") is not None}
                # Sum by insertion order (beam-0 and beam-1 arrive in pairs)
                b0_ids = list(b0_flux.keys())
                b1_ids = list(b1_flux.keys())
                for i, r0 in enumerate(b0_ids):
                    f0 = b0_flux[r0]
                    f1 = b1_flux.get(b1_ids[i]) if i < len(b1_ids) else 0.0
                    flux_by_ref[r0] = (f0 or 0) + (f1 or 0)
            else:
                flux_by_ref = {s["ref_id"]: s["flux"] for s in sources
                               if s.get("flux") is not None}
            for rid, fl in flux_by_ref.items():
                if fl is not None and np.isfinite(fl) and fl > 0:
                    _ensure(ch, rid)
                    bufs[ch][rid].add(jd, float(fl))
            new_data = True

            if tid is not None:
                ht = _jd_to_hours(np.array([jd]), utc_offset)[0]
                track = [tid] + [r for r in cids if r != tid]
                pstr = "  ".join(
                    [f"T={flux_by_ref.get(track[0],0):.0f}"] +
                    [f"C{i}={flux_by_ref.get(r,0):.0f}"
                     for i,r in enumerate(track[1:],1)]
                )
                print(f"  ch{ch} ({CH_BAND[ch]})  {ht%24:.3f}h  {pstr}")
            else:
                print(f"  ch{ch} ({CH_BAND[ch]})  jd={jd:.6f}  "
                      f"{len(flux_by_ref)} sources stored (no selection yet)")

            if time.time() - last_draw >= interval:
                _redraw(); last_draw = time.time(); new_data = False
            time.sleep(0.01); fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[raw_lc] Stopped.")
    finally:
        plt.ioff(); plt.show(block=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="SPARC4 raw Δmag light curves — ZeroMQ subscriber",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int,   default=5556)
    p.add_argument("--channels",   default="1,2,3,4")
    p.add_argument("--db",         default="")
    p.add_argument("--history",    type=int,   default=300)
    p.add_argument("--interval",   type=float, default=2.0)
    p.add_argument("--utc-offset", type=float, default=-3.0)
    g = p.add_argument_group("Night / twilight options")
    g.add_argument("--twilight", action="store_true",
                   help="Draw sunset/sunrise and twilight vertical lines")
    g.add_argument("--night-xlim", action="store_true",
                   help="Fix x-axis from sunset to sunrise of the current night")
    g.add_argument("--obs-lon",   type=float, default=OPD_LON,
                   help="Observatory longitude deg E  [default: OPD]")
    g.add_argument("--obs-lat",   type=float, default=OPD_LAT,
                   help="Observatory latitude deg N   [default: OPD]")
    g.add_argument("--obs-alt",   type=float, default=OPD_ALT,
                   help="Observatory altitude m       [default: OPD]")
    g.add_argument("--obs-date",  default="",
                   help="ISO date of the night evening (default: infer from data)")
    opts = p.parse_args()
    channels = [int(c) for c in opts.channels.split(",")]
    run(opts.host, opts.port, channels,
        opts.db or None, opts.history, opts.interval, opts.utc_offset,
        twilight=opts.twilight, night_xlim_flag=opts.night_xlim,
        obs_lon=opts.obs_lon, obs_lat=opts.obs_lat,
        obs_alt=opts.obs_alt, obs_date=opts.obs_date)


if __name__ == "__main__":
    main()
