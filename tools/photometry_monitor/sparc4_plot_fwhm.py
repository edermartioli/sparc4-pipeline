"""
sparc4_plot_fwhm.py
===================
SPARC4 real-time FWHM monitor — ZeroMQ subscriber.

Displays median FWHM in arcsec vs local time for all four channels.
Loads historical data from the SQLite databases on startup, then
appends new measurements as they arrive from the publisher.

All channel panels share the same time axis so seeing trends
can be compared directly across bands.

Usage
-----
::

    python sparc4_plot_fwhm.py --db /data/SPARC4/reduced
    python sparc4_plot_fwhm.py --host 192.168.1.10 --db /data/SPARC4/reduced

Options
-------
--host HOST         Publisher host                  [default: localhost]
--port PORT         Publisher ZeroMQ port           [default: 5556]
--channels LIST     Comma-separated channel list    [default: 1,2,3,4]
--db DIR            Directory with monitor_chN.db   [default: none]
--history N         Maximum data points to display  [default: 300]
--interval SEC      Plot refresh interval (s)       [default: 2.0]
--platescale PX     Arcsec/pixel override           [default: from msg]
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
from typing import Dict, Deque, List, Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

CH_COLOR = {1: "darkblue", 2: "darkgreen", 3: "darkorange", 4: "darkred"}
CH_BAND  = {1: "g",        2: "r",         3: "i",          4: "z"}


# ─────────────────────────────────────────────────────────────────────────────
# Time utilities
# ─────────────────────────────────────────────────────────────────────────────

def _jd_to_hours(jd_arr: np.ndarray, utc_offset_h: float) -> np.ndarray:
    """Convert JD array to local decimal hours (0–24)."""
    if len(jd_arr) == 0:
        return np.array([])
    frac = (jd_arr + 0.5 + utc_offset_h / 24.0) % 1.0
    return frac * 24.0

def _format_time_axis(ax) -> None:
    """Format x-axis ticks as HH:MM."""
    def _fmt(x, pos):
        x = x % 24.0
        h = int(x); m = int(round((x - h) * 60)) % 60
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

# ─────────────────────────────────────────────────────────────────────────────
# DB loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_db(db_path: str, platescale: float, history: int):
    """Return (hours_list, fwhm_arcsec_list, fwhm_err_list, nsrc_list)."""
    if not os.path.isfile(db_path):
        return [], [], [], []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT jd, fwhm, fwhm_err, n_sources FROM frames "
            "WHERE jd IS NOT NULL AND fwhm IS NOT NULL AND bad = 0 "
            "ORDER BY jd"
        ).fetchall()
        conn.close()
        rows = rows[-history:]
        jds   = [r["jd"]        for r in rows]
        fwhms = [r["fwhm"] * platescale        for r in rows]
        ferrs = [(r["fwhm_err"] or 0) * platescale for r in rows]
        nsrcs = [r["n_sources"] for r in rows]
        print(f"  {os.path.basename(db_path)}: {len(jds)} points loaded")
        return jds, fwhms, ferrs, nsrcs
    except Exception as exc:
        print(f"  [warn] {db_path}: {exc}")
        return [], [], [], []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(host: str, port: int, channels: List[int], db_dir: Optional[str],
        history: int, interval: float, platescale_override: Optional[float],
        utc_offset: float,
        twilight: bool = False,
        night_xlim_flag: bool = False,
        obs_lon: float = OPD_LON,
        obs_lat: float = OPD_LAT,
        obs_alt: float = OPD_ALT,
        obs_date: str = "") -> None:

    ps_default = platescale_override or 0.335

    buf_jd:   Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}
    buf_fwhm: Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}
    buf_ferr: Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}
    buf_nsrc: Dict[int, Deque] = {ch: deque(maxlen=history) for ch in channels}

    if db_dir:
        print("[fwhm] Loading historical data...")
        for ch in channels:
            path = os.path.join(db_dir, f"monitor_ch{ch}.db")
            jds, fwhms, ferrs, nsrcs = _load_db(path, ps_default, history)
            buf_jd[ch].extend(jds);   buf_fwhm[ch].extend(fwhms)
            buf_ferr[ch].extend(ferrs); buf_nsrc[ch].extend(nsrcs)

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    for ch in channels:
        sock.setsockopt_string(zmq.SUBSCRIBE, f"sparc4.ch{ch}")
    sock.setsockopt(zmq.RCVTIMEO, int(interval * 1000))
    print(f"[fwhm] Subscribed  tcp://{host}:{port}  ch={channels}")

    # Figure: main FWHM panel + slim N-sources strip
    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    fig.canvas.manager.set_window_title("SPARC4 — FWHM Monitor")
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1],
                           hspace=0.10, left=0.09, right=0.97,
                           top=0.92, bottom=0.11)

    ax_fwhm = fig.add_subplot(gs[0])
    ax_nsrc = fig.add_subplot(gs[1], sharex=ax_fwhm)

    for ax in (ax_fwhm, ax_nsrc):
        ax.set_facecolor("white")
        ax.grid(True, color="lightgray", ls="--", lw=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors="black", labelsize=11)

    ax_fwhm.set_ylabel('FWHM  (arcsec)', fontsize=12)
    ax_fwhm.axhline(1.0, color="gray",       lw=1.0, ls=":", label='1.0"')
    ax_fwhm.axhline(2.0, color="darkorange", lw=1.0, ls=":", label='2.0"')
    ax_fwhm.set_ylim(0, 4)
    plt.setp(ax_fwhm.get_xticklabels(), visible=False)

    ax_nsrc.set_ylabel("N src", fontsize=11)
    ax_nsrc.set_xlabel(f"Local time  (UTC{utc_offset:+.0f}h)", fontsize=12)
    ax_nsrc.yaxis.set_major_locator(_mticker.MaxNLocator(integer=True, nbins=4))
    _format_time_axis(ax_nsrc)

    fig.suptitle("SPARC4  —  Seeing / FWHM Monitor",
                 fontsize=14, fontweight="bold", color="black")

    lines_fwhm: dict = {}
    fills_fwhm: dict = {}
    lines_nsrc: dict = {}

    last_draw = 0.0
    new_data  = False

    plt.ion()
    fig.show()

    def _redraw():
        for ch in channels:
            if not buf_jd[ch]:
                continue
            jds  = np.array(buf_jd[ch])
            fwhm = np.array(buf_fwhm[ch])
            ferr = np.nan_to_num(np.array(buf_ferr[ch]))
            nsrc = np.array(buf_nsrc[ch])
            ht   = _jd_to_hours(jds, utc_offset)
            col  = CH_COLOR[ch]
            lbl  = f"{CH_BAND[ch]} (ch{ch})"

            valid = np.isfinite(fwhm)
            if not valid.any():
                continue

            if ch not in lines_fwhm:
                lines_fwhm[ch], = ax_fwhm.plot(
                    ht[valid], fwhm[valid], "o-", color=col,
                    lw=1.8, ms=5, label=lbl, zorder=3)
                fills_fwhm[ch] = ax_fwhm.fill_between(
                    ht[valid],
                    (fwhm - ferr)[valid], (fwhm + ferr)[valid],
                    color=col, alpha=0.15, zorder=2)
                lines_nsrc[ch], = ax_nsrc.plot(
                    ht, nsrc, "s-", color=col, alpha=0.85, lw=1.4, ms=4)
            else:
                lines_fwhm[ch].set_data(ht[valid], fwhm[valid])
                fills_fwhm[ch].remove()
                fills_fwhm[ch] = ax_fwhm.fill_between(
                    ht[valid],
                    (fwhm - ferr)[valid], (fwhm + ferr)[valid],
                    color=col, alpha=0.15, zorder=2)
                lines_nsrc[ch].set_data(ht, nsrc)

        ax_fwhm.relim(); ax_fwhm.autoscale_view(scalex=True, scaley=False)
        ax_nsrc.relim(); ax_nsrc.autoscale_view()
        ax_nsrc.set_ylim(bottom=0)

        # Shared x-axis (both panels already share via sharex; sync ensures
        # the limits reflect all data)
        _sync_xlim([ax_fwhm, ax_nsrc])

        if not ax_fwhm.get_legend():
            ax_fwhm.legend(loc="upper left", fontsize=10,
                           ncol=len(channels), framealpha=0.9)

        if buf_jd[channels[0]]:
            last_jd = max(buf_jd[ch][-1] for ch in channels if buf_jd[ch])
            last_ht = _jd_to_hours(np.array([last_jd]), utc_offset)[0]
            ax_fwhm.set_title(
                f"Last: {last_ht%24:.4f} h  (JD {last_jd:.6f})",
                color="gray", fontsize=9, loc="right", pad=4)

        if _HAS_ASTRO and twilight and sun_evts is not None:
            for _ax in [ax_fwhm, ax_nsrc]:
                for _l in list(_ax.get_lines()):
                    if getattr(_l, "_is_twilight", False):
                        _l.remove()
            add_twilight_lines([ax_fwhm, ax_nsrc], sun_evts,
                               legend_ax_index=0)
            for _ax in [ax_fwhm, ax_nsrc]:
                for _l in _ax.get_lines():
                    if _l.get_label() in ("Sunset","Sunrise",
                        "Civil twil.","Nautical twil.","Astron. twil."):
                        _l._is_twilight = True
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # ── Sun events / twilight ─────────────────────────────────────────────
    sun_evts = None
    if _HAS_ASTRO and (twilight or night_xlim_flag):
        _jd0 = next(
            (jd for dq in buf_jd.values() for jd in dq if np.isfinite(jd)),
            None)
        _date = obs_date or (date_from_jd(_jd0, utc_offset) if _jd0 else "")
        if not _date:
            from datetime import date as _d
            _date = _d.today().isoformat()
        sun_evts = compute_sun_events(
            _date, obs_lon, obs_lat, obs_alt, utc_offset)
        if night_xlim_flag and sun_evts is not None:
            xlo, xhi = night_xlim(sun_evts)
            if xlo is not None:
                ax_fwhm.set_xlim(xlo, xhi)
                print(f"[fwhm] Night x-lim: {xlo:.3f} – {xhi:.3f} h")


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
            if ch not in channels:
                continue
            jd      = msg.get("jd")
            fwhm_as = msg.get("fwhm_arcsec")
            ferr    = msg.get("fwhm_err_pix")
            nsrc    = msg.get("n_sources", 0)
            ps      = platescale_override or msg.get("platescale", ps_default)

            if jd is None or fwhm_as is None:
                continue

            ferr_as = (ferr * ps) if ferr is not None else 0.0
            buf_jd[ch].append(jd)
            buf_fwhm[ch].append(fwhm_as)
            buf_ferr[ch].append(ferr_as)
            buf_nsrc[ch].append(nsrc)
            new_data = True

            ht = _jd_to_hours(np.array([jd]), utc_offset)[0]
            print(f"  ch{ch} ({CH_BAND[ch]})  "
                  f"{ht%24:.4f}h  FWHM={fwhm_as:.2f}\"  N={nsrc}")

            if time.time() - last_draw >= interval:
                _redraw(); last_draw = time.time(); new_data = False

            time.sleep(0.01)
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n[fwhm] Stopped.")
    finally:
        plt.ioff(); plt.show(block=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="SPARC4 FWHM monitor — ZeroMQ subscriber",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int,   default=5556)
    p.add_argument("--channels",   default="1,2,3,4")
    p.add_argument("--db",         default="")
    p.add_argument("--history",    type=int,   default=300)
    p.add_argument("--interval",   type=float, default=2.0)
    p.add_argument("--platescale", type=float, default=None)
    p.add_argument("--utc-offset", type=float, default=-3.0,
                   help="Local time UTC offset in hours (e.g. -3 for Brazil)")
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
                   help="ISO date of the night's evening (default: infer from data)")
    opts = p.parse_args()
    channels = [int(c) for c in opts.channels.split(",")]
    run(opts.host, opts.port, channels,
        opts.db or None, opts.history, opts.interval,
        opts.platescale, opts.utc_offset,
        twilight=opts.twilight,
        night_xlim_flag=opts.night_xlim,
        obs_lon=opts.obs_lon, obs_lat=opts.obs_lat,
        obs_alt=opts.obs_alt, obs_date=opts.obs_date)


if __name__ == "__main__":
    main()
