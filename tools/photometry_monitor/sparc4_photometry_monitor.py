"""
sparc4_photometry_monitor.py
============================
SPARC4 real-time raw photometry monitor — data acquisition and publisher.

Scans data directories for new FITS frames, measures FWHM and aperture
photometry, stores results in a per-channel SQLite database and broadcasts
results over a ZeroMQ PUB socket so any number of plot microservices can
subscribe from any machine on the network.

Architecture
------------
This script is the **publisher**.  For each processed frame it sends one
JSON message per channel on a ZeroMQ PUB socket (default port 5556).
Run one or more of the companion subscriber scripts on the same machine
or any machine on the network to see live plots:

    sparc4_plot_fwhm.py          -- FWHM vs time, all 4 channels
    sparc4_plot_saturation.py    -- Peak counts for N brightest sources
    sparc4_plot_lightcurves.py   -- Differential magnitude light curves

ZeroMQ message format
---------------------
Each message has two frames:
  1. Topic string  (bytes): ``b"sparc4.ch1"`` … ``b"sparc4.ch4"``
  2. JSON payload  (bytes): UTF-8 encoded JSON object with keys:

     channel      int    channel number 1-4
     band         str    "g" / "r" / "i" / "z"
     filename     str    FITS file basename
     date_obs     str    ISO-T observation time from header
     jd           float  Julian date (UTC)
     object_name  str    OBJECT header keyword
     fwhm_pix     float  median FWHM across all sources (pixels)
     fwhm_arcsec  float  median FWHM in arcsec
     fwhm_err_pix float  FWHM uncertainty (pixels)
     n_sources    int    number of sources detected
     platescale   float  arcsec/pixel used
     sources      list   per-source dicts, sorted by descending peak count:
       ref_id     int    stable cross-frame source identifier
       x, y       float  centroid position (pixels)
       flux       float  background-subtracted aperture flux (ADU)
       flux_err   float  flux uncertainty (ADU)
       peak       float  peak pixel value (ADU, raw image)

Typical usage
-------------
Start the monitor (publisher), watching for new files every 30 s::

    python -W ignore sparc4_photometry_monitor.py \\
        --ch1 /data/SPARC4/sparc4acs1/today \\
        --ch2 /data/SPARC4/sparc4acs2/today \\
        --ch3 /data/SPARC4/sparc4acs3/today \\
        --ch4 /data/SPARC4/sparc4acs4/today \\
        --seq_suffix cr3 --object "HD 12345" \\
        --select-sources --watch --interval 30 \\
        --fwhm 5 --threshold 10

On any machine, start one or more subscribers::

    python sparc4_plot_fwhm.py --host 192.168.1.10
    python sparc4_plot_saturation.py --host 192.168.1.10 --n-sources 5
    python sparc4_plot_lightcurves.py --host 192.168.1.10

Author
------
Eder Martioli <martioli@lna.br>
Laboratório Nacional de Astrofísica — LNA/MCTI
Created: May 2026
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import astropy.io.fits as fits
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.stats import SigmaClip
from astropy.time import Time
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
import photutils.aperture
import matplotlib
matplotlib.use("TkAgg")   # for interactive source selection window
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    import zmq
    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False
    print("[monitor] WARNING: pyzmq not installed. Messages will not be published.")
    print("[monitor]          Install with:  pip install pyzmq")


# ── Observatory constants (Pico dos Dias / OPD) ──────────────────────────────

OPD_LONGITUDE = -45.5825   # deg E
OPD_LATITUDE  = -22.53444  # deg N
OPD_ALTITUDE  = 1864.0     # m

# ── Channel definitions ───────────────────────────────────────────────────────

CHANNEL_COLORS = {1: "#4fc3f7", 2: "#ef5350", 3: "#ab47bc", 4: "#ff8a65"}
CHANNEL_LABELS = {1: "g",       2: "r",       3: "i",       4: "z"}

ZMQ_DEFAULT_PORT = 5556


# ─────────────────────────────────────────────────────────────────────────────
# SourceMatcher — position-based cross-frame source identification
# ─────────────────────────────────────────────────────────────────────────────

class SourceMatcher:
    """Robust cross-frame source matching using a voting shift estimator.

    Algorithm
    ---------
    The first processed frame is stored as the reference catalogue.
    For each subsequent frame the algorithm works in two stages:

    1. **Global shift estimation (voting)**
       Every pair (new source i, reference source j) votes for a candidate
       frame offset (Δx, Δy) = (ref_x[j] - new_x[i], ref_y[j] - new_y[i]).
       All N×M votes are histogrammed on a coarse grid (bin size = 1 FWHM).
       The bin with the most votes is taken as the true frame offset.
       A minimum vote count (``min_votes``) guards against noise — if the
       peak is too weak the shift is rejected and falls back to zero.

    2. **Nearest-neighbour matching**
       The estimated shift is subtracted from the new positions.
       Each shifted source is matched to the closest reference star within
       ``match_radius`` pixels.  Unmatched sources get ref_id = -1.

    This approach handles guiding drifts of any magnitude (up to half the
    field of view), guiding interruptions and re-acquisitions, and sparse
    fields (≥ 5 sources).  Runtime is O(N×M) numpy operations — typically
    < 2 ms for 200 sources, well within the quicklook budget.

    Parameters
    ----------
    match_radius : float
        Maximum residual distance (pixels) for a valid match after the
        global shift has been subtracted.  A good default is ~1 FWHM.
    min_votes : int
        Minimum number of consistent source pairs required to accept the
        estimated shift.  Lower values are more permissive in sparse fields;
        higher values reduce the risk of a spurious shift.  Default: 3.
    bin_size : float
        Histogram bin size for the voting step (pixels).  Should be
        comparable to the position measurement uncertainty, typically
        0.5–2 px.  Default: 2.0 px.
    verbose : bool
        Print the estimated shift and match statistics for each frame.
    """

    def __init__(self, match_radius: float = 5.0, min_votes: int = 3,
                 bin_size: float = 2.0, verbose: bool = False) -> None:
        self.match_radius = match_radius
        self.min_votes    = min_votes
        self.bin_size     = bin_size
        self.verbose      = verbose
        self._ref_x: Optional[np.ndarray] = None
        self._ref_y: Optional[np.ndarray] = None
        # Running estimate of the last accepted shift — used as a warm start
        # so the voting histogram can be narrowed on subsequent frames.
        self._last_dx: float = 0.0
        self._last_dy: float = 0.0

    # ------------------------------------------------------------------
    @property
    def has_reference(self) -> bool:
        return self._ref_x is not None

    @property
    def n_ref(self) -> int:
        return len(self._ref_x) if self._ref_x is not None else 0

    def set_reference(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Store first-frame detections as the reference catalogue."""
        self._ref_x = np.asarray(xs, dtype=float)
        self._ref_y = np.asarray(ys, dtype=float)
        self._last_dx = 0.0
        self._last_dy = 0.0

    # ------------------------------------------------------------------
    def _estimate_shift(self,
                        xs: np.ndarray,
                        ys: np.ndarray) -> tuple[float, float, int]:
        """Estimate the global frame shift by pairwise offset voting.

        Returns
        -------
        (dx, dy, n_votes) where n_votes is the peak vote count.
        dx, dy are the offsets to ADD to new positions to align them with
        the reference frame: ref ≈ new + (dx, dy).
        """
        # All pairwise offsets: shape (N_new, N_ref)
        dx_all = self._ref_x[np.newaxis, :] - xs[:, np.newaxis]  # (N,M)
        dy_all = self._ref_y[np.newaxis, :] - ys[:, np.newaxis]  # (N,M)

        dx_flat = dx_all.ravel()
        dy_flat = dy_all.ravel()

        # 2-D histogram of offsets — bin size = 1 FWHM proxy
        bs = self.bin_size
        # Use a range centred on the last known shift (warm start)
        cx, cy = self._last_dx, self._last_dy
        # Wide enough to capture ±half-frame shifts even from zero
        half = max(max(self._ref_x.max(), xs.max()),
                   max(self._ref_y.max(), ys.max())) + bs
        x_edges = np.arange(cx - half, cx + half + bs, bs)
        y_edges = np.arange(cy - half, cy + half + bs, bs)

        hist, xed, yed = np.histogram2d(dx_flat, dy_flat,
                                        bins=[x_edges, y_edges])
        peak_idx = np.unravel_index(hist.argmax(), hist.shape)
        n_votes  = int(hist[peak_idx])
        dx = float(xed[peak_idx[0]] + bs / 2)  # bin centre
        dy = float(yed[peak_idx[1]] + bs / 2)

        return dx, dy, n_votes

    # ------------------------------------------------------------------
    def match(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Match new positions to the reference catalogue.

        Two-stage algorithm:
        1. Estimate the global frame shift by pairwise offset voting.
        2. Nearest-neighbour match within ``match_radius`` after correcting
           for the estimated shift.

        As a robustness fallback, if the voted shift gives fewer matches
        than a direct zero-shift nearest-neighbour search, the zero-shift
        result is returned instead.  This protects against spurious voting
        peaks when there are very few sources.

        Returns
        -------
        ref_ids : ndarray of int, shape (N,)
            Stable reference catalogue index for each new source.
            -1 means the source could not be matched.
        """
        if not self.has_reference:
            raise RuntimeError("No reference catalogue set.")

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        def _nn_match(xs_s, ys_s):
            """Pure nearest-neighbour within match_radius."""
            ids = np.full(len(xs_s), -1, dtype=int)
            for i, (x, y) in enumerate(zip(xs_s, ys_s)):
                dists = np.hypot(self._ref_x - x, self._ref_y - y)
                j = int(np.argmin(dists))
                if dists[j] <= self.match_radius:
                    ids[i] = j
            return ids

        # ── Stage 1: estimate global frame shift ──────────────────────
        if len(xs) >= 2 and len(self._ref_x) >= 2:
            dx, dy, n_votes = self._estimate_shift(xs, ys)
            if n_votes >= self.min_votes:
                self._last_dx = dx
                self._last_dy = dy
                if self.verbose:
                    print(f"    [matcher] shift=({dx:+.1f}, {dy:+.1f}) px  "
                          f"votes={n_votes}/{len(xs)*len(self._ref_x)}")
            else:
                dx, dy = self._last_dx, self._last_dy
                if self.verbose:
                    print(f"    [matcher] weak peak ({n_votes} votes) — "
                          f"using last shift ({dx:+.1f}, {dy:+.1f})")
        else:
            dx, dy = self._last_dx, self._last_dy

        # ── Stage 2: voted-shift match ─────────────────────────────────
        ref_ids_voted = _nn_match(xs + dx, ys + dy)
        n_voted = int((ref_ids_voted >= 0).sum())

        # ── Fallback: zero-shift match (no correction) ─────────────────
        # Try direct NN with no shift.  If it gives more matches than the
        # voted result, use it.  This handles the case where the voting
        # histogram finds a spurious peak with few sources, producing a
        # bad correction that moves everything further from the catalogue.
        ref_ids_zero = _nn_match(xs, ys)
        n_zero = int((ref_ids_zero >= 0).sum())

        if n_zero > n_voted:
            ref_ids = ref_ids_zero
            n_matched = n_zero
            # Reset last shift since zero-shift was better
            self._last_dx = 0.0
            self._last_dy = 0.0
            if self.verbose:
                print(f"    [matcher] fallback to zero-shift "
                      f"({n_zero} > {n_voted} matches)")
        else:
            ref_ids = ref_ids_voted
            n_matched = n_voted

        if self.verbose:
            print(f"    [matcher] matched {n_matched}/{len(xs)} sources")

        return ref_ids


# ─────────────────────────────────────────────────────────────────────────────
# PhotometryDB — SQLite cache for processed frames
# ─────────────────────────────────────────────────────────────────────────────

class PhotometryDB:
    """Per-channel SQLite cache of photometric results.

    Tables
    ------
    frames        : one row per processed FITS file
    sources       : per-source measurements keyed by stable ref_id
    ref_catalogue : reference star positions (survives restarts)
    """

    _CREATE_FRAMES = """
        CREATE TABLE IF NOT EXISTS frames (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT    NOT NULL UNIQUE,
            object_name  TEXT,
            date_obs     TEXT,
            jd           REAL,
            fwhm         REAL,
            fwhm_x       REAL,
            fwhm_y       REAL,
            fwhm_err     REAL,
            n_sources    INTEGER,
            inst_mode    TEXT    DEFAULT 'PHOT',
            bad          INTEGER NOT NULL DEFAULT 0,
            processed_at TEXT DEFAULT (datetime('now'))
        );"""

    _CREATE_SOURCES = """
        CREATE TABLE IF NOT EXISTS sources (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id  INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
            ref_id    INTEGER NOT NULL,
            beam      INTEGER,           -- 0=ordinary, 1=extraordinary, NULL=PHOT
            x         REAL,
            y         REAL,
            flux      REAL,
            flux_err  REAL,
            peak      REAL
        );"""

    _CREATE_REFCAT = """
        CREATE TABLE IF NOT EXISTS ref_catalogue (
            ref_id  INTEGER PRIMARY KEY,
            x       REAL NOT NULL,
            y       REAL NOT NULL
        );"""

    _CREATE_SELECTION = """
        CREATE TABLE IF NOT EXISTS selection (
            role    TEXT NOT NULL,   -- "target" or "comp"
            ref_id  INTEGER NOT NULL,
            rank    INTEGER NOT NULL,  -- 0=target(beam0), 1=target(beam1), 2=C1(beam0), ...
            beam    INTEGER           -- 0=ordinary, 1=extraordinary, NULL=PHOT
        );"""

    _CREATE_BEAM_OFFSET = """
        CREATE TABLE IF NOT EXISTS beam_offset (
            id   INTEGER PRIMARY KEY CHECK (id = 1),
            dx   REAL NOT NULL,
            dy   REAL NOT NULL
        );"""

    def __init__(self, db_path) -> None:
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.execute(self._CREATE_FRAMES)
            conn.execute(self._CREATE_SOURCES)
            conn.execute(self._CREATE_REFCAT)
            conn.execute(self._CREATE_SELECTION)
            conn.execute(self._CREATE_BEAM_OFFSET)
            # Migrate existing DBs to add new columns
            frame_cols  = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
            source_cols = {r[1] for r in conn.execute("PRAGMA table_info(sources)")}
            sel_cols    = {r[1] for r in conn.execute("PRAGMA table_info(selection)")}
            if "bad"       not in frame_cols:
                conn.execute("ALTER TABLE frames ADD COLUMN bad INTEGER NOT NULL DEFAULT 0")
            if "inst_mode" not in frame_cols:
                conn.execute("ALTER TABLE frames ADD COLUMN inst_mode TEXT DEFAULT 'PHOT'")
            if "beam"      not in source_cols:
                conn.execute("ALTER TABLE sources ADD COLUMN beam INTEGER")
            if "beam"      not in sel_cols:
                conn.execute("ALTER TABLE selection ADD COLUMN beam INTEGER")

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def is_processed(self, filename: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM frames WHERE filename=?",
                (os.path.basename(filename),)
            ).fetchone() is not None

    def insert_bad_frame(self, filename: str,
                         date_obs: str = "",
                         jd=None,
                         object_name: str = "",
                         reason: str = "") -> None:
        """Record a frame as processed but bad (no usable sources).

        Inserts the frame into ``frames`` with ``bad=1`` and ``n_sources=0``
        so the monitor will not retry it on subsequent polling cycles.
        No rows are written to ``sources``.
        """
        if reason:
            pass   # reason is printed by the caller; not stored in DB
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO frames
                   (filename, object_name, date_obs, jd, n_sources, bad)
                   VALUES (?, ?, ?, ?, 0, 1)""",
                (os.path.basename(filename), object_name, date_obs, jd),
            )

    def has_reference_catalogue(self) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM ref_catalogue LIMIT 1"
            ).fetchone() is not None

    def save_reference_catalogue(self, xs, ys) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ref_catalogue")
            conn.executemany(
                "INSERT INTO ref_catalogue (ref_id, x, y) VALUES (?,?,?)",
                [(i, float(x), float(y)) for i, (x, y) in enumerate(zip(xs, ys))]
            )

    def load_reference_catalogue(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT x, y FROM ref_catalogue ORDER BY ref_id"
            ).fetchall()
        if not rows:
            return np.array([]), np.array([])
        return (np.array([r["x"] for r in rows]),
                np.array([r["y"] for r in rows]))

    def save_selection(self, target_ref_id, comp_ref_ids,
                        beam_map: Optional[dict] = None) -> None:
        """Persist the target/comparison selection.

        Parameters
        ----------
        target_ref_id : int or None
        comp_ref_ids  : list of int
        beam_map : dict {ref_id: beam} or None
            In POLAR mode, maps each ref_id to its beam index (0 or 1).
            In PHOT mode (None), the beam column is left NULL.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM selection")
            rows = []
            bm = beam_map or {}
            if target_ref_id is not None:
                rows.append(("target", int(target_ref_id), 0,
                              bm.get(int(target_ref_id))))
            for rank, rid in enumerate(comp_ref_ids, start=1):
                rows.append(("comp", int(rid), rank,
                              bm.get(int(rid))))
            conn.executemany(
                "INSERT INTO selection (role, ref_id, rank, beam) VALUES (?,?,?,?)",
                rows)

    def load_selection(self):
        """Restore (target_ref_id, comp_ref_ids, beam_map) from DB.

        Returns (None, [], {}) if no selection has been saved.
        beam_map maps ref_id -> beam index (or is empty in PHOT mode).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, ref_id, beam FROM selection ORDER BY rank"
            ).fetchall()
        if not rows:
            return None, [], {}
        target   = None
        comps    = []
        beam_map = {}
        for row in rows:
            rid  = row["ref_id"]
            beam = row["beam"]
            if beam is not None:
                beam_map[rid] = beam
            if row["role"] == "target":
                target = rid
            else:
                comps.append(rid)
        return target, comps, beam_map

    def has_selection(self) -> bool:
        """True if a target/comp selection has been saved."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM selection LIMIT 1"
            ).fetchone() is not None

    def save_beam_offset(self, dx: float, dy: float) -> None:
        """Store the beam separation vector (POLAR mode only)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO beam_offset (id, dx, dy) VALUES (1, ?, ?)",
                (float(dx), float(dy)))

    def load_beam_offset(self) -> Optional[tuple]:
        """Return (dx, dy) beam offset or None if not stored."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dx, dy FROM beam_offset WHERE id=1"
            ).fetchone()
        return (row["dx"], row["dy"]) if row else None

    def has_beam_offset(self) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM beam_offset LIMIT 1"
            ).fetchone() is not None

    def insert_frame(self, filename: str, result: dict,
                     ref_ids: np.ndarray,
                     beam_ids: Optional[np.ndarray] = None) -> int:
        """Persist one frame and its matched source measurements.

        Parameters
        ----------
        beam_ids : ndarray of int or None
            Per-detection beam index (0=ordinary, 1=extraordinary).
            None in PHOT mode.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT OR REPLACE INTO frames
                   (filename, object_name, date_obs, jd,
                    fwhm, fwhm_x, fwhm_y, fwhm_err, n_sources, inst_mode, bad)
                   VALUES (?,?,?,?,?,?,?,?,?,?,0)""",
                (os.path.basename(filename),
                 result.get("object_name", ""),
                 result.get("date_obs", ""),
                 result.get("jd"),
                 result.get("fwhm"),
                 result.get("fwhm_x"),
                 result.get("fwhm_y"),
                 result.get("fwhm_err"),
                 result.get("n_sources", 0),
                 result.get("inst_mode", "PHOT"))
            )
            frame_id = cur.lastrowid

            fluxes   = np.asarray(result.get("fluxes",     []), dtype=float)
            fluxerrs = np.asarray(result.get("fluxerrs",   []), dtype=float)
            peaks    = np.asarray(result.get("max_counts", []), dtype=float)
            xs       = np.asarray(result.get("xcentroids", []), dtype=float)
            ys       = np.asarray(result.get("ycentroids", []), dtype=float)

            rows = []
            for i, rid in enumerate(ref_ids):
                if rid < 0:
                    continue
                beam = (int(beam_ids[i]) if beam_ids is not None and i < len(beam_ids)
                        else None)
                rows.append((frame_id, int(rid), beam,
                              float(xs[i])       if i < len(xs)       else None,
                              float(ys[i])       if i < len(ys)       else None,
                              float(fluxes[i])   if i < len(fluxes)   else None,
                              float(fluxerrs[i]) if i < len(fluxerrs) else None,
                              float(peaks[i])    if i < len(peaks)    else None))
            conn.executemany(
                "INSERT INTO sources "
                "(frame_id, ref_id, beam, x, y, flux, flux_err, peak) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows
            )
        return frame_id


# ─────────────────────────────────────────────────────────────────────────────
# FWHM measurement helper
# ─────────────────────────────────────────────────────────────────────────────

def _measure_fwhm(img, positions, *, err_data=None, window_size=25,
                  global_fit=True):
    """Fit 2-D Gaussians to measure FWHM.

    Uses sparc4.pipeline_lib when available; falls back to 1-D Gaussian
    fits along x and y marginals of each source cutout otherwise.

    Returns
    -------
    fwhms_x, fwhms_y, xcentroids, ycentroids : ndarray
    """
    try:
        import sparc4.pipeline_lib as s4pipelib
        return s4pipelib.measure_fwhm_from_2DGaussianFit(
            img, positions, err_data=err_data,
            window_size=window_size, global_fit=global_fit,
            plot=False, verbose=False,
        )
    except Exception:
        pass

    from scipy.optimize import curve_fit

    def _gauss(x, amp, mu, sigma, bg):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg

    fwhms_x, fwhms_y, xcs, ycs = [], [], [], []
    half = window_size // 2
    for (x0, y0) in positions:
        xi, yi = int(round(x0)), int(round(y0))
        cut = img[max(0, yi-half):yi+half+1, max(0, xi-half):xi+half+1]
        fwhm_x = fwhm_y = np.nan
        try:
            mx = np.nansum(cut, axis=0); px = np.arange(len(mx))
            popt, _ = curve_fit(_gauss, px, mx,
                                p0=[mx.max(), len(mx)/2, 3., 0.], maxfev=1000)
            fwhm_x = abs(popt[2]) * 2.355
        except Exception:
            pass
        try:
            my = np.nansum(cut, axis=1); py = np.arange(len(my))
            popt, _ = curve_fit(_gauss, py, my,
                                p0=[my.max(), len(my)/2, 3., 0.], maxfev=1000)
            fwhm_y = abs(popt[2]) * 2.355
        except Exception:
            pass
        fwhms_x.append(fwhm_x); fwhms_y.append(fwhm_y)
        xcs.append(float(x0));   ycs.append(float(y0))
    return (np.array(fwhms_x), np.array(fwhms_y),
            np.array(xcs),     np.array(ycs))


# ─────────────────────────────────────────────────────────────────────────────
# Core photometry function
# ─────────────────────────────────────────────────────────────────────────────

def measure_raw_photometry(
    filename: str,
    *,
    threshold: float = 3.0,
    fwhm_for_detection: float = 5.0,
    aperture_radius: float = 15.0,
    window_size: int = 25,
    psf_global_fit: bool = True,
    read_noise_key: str = "RDNOISE",
    time_key: str = "DATE-OBS",
    object_key: str = "OBJECT",
    longitude: float = OPD_LONGITUDE,
    latitude: float  = OPD_LATITUDE,
    altitude: float  = OPD_ALTITUDE,
    verbose: bool = False,
) -> dict:
    """Measure source detection, FWHM and aperture photometry on one raw frame.

    Parameters
    ----------
    filename : str
        Path to the FITS file (.fits or .fits.fz).
    threshold : float
        Detection threshold in units of background RMS median (DAOStarFinder).
    fwhm_for_detection : float
        Approximate PSF FWHM in pixels, passed to DAOStarFinder.
    aperture_radius : float
        Circular aperture radius in pixels.
    window_size : int
        Cutout half-size in pixels for the 2-D Gaussian FWHM fitting.
    psf_global_fit : bool
        Passed through to the FWHM fitting routine.
    read_noise_key : str
        FITS keyword for read noise in electrons (optional).
    time_key : str
        FITS keyword for UTC observation start time (ISO-T format).
    object_key : str
        FITS keyword for target name.
    longitude, latitude, altitude : float
        Observatory coordinates for time conversion.
    verbose : bool
        Print per-image diagnostics.

    Returns
    -------
    dict with keys: date_obs, jd, object_name, fwhm, fwhm_x, fwhm_y,
    fwhm_err, fwhm_x_err, fwhm_y_err, fwhms_x, fwhms_y, n_sources,
    xcentroids, ycentroids, fluxes, fluxerrs, max_counts,
    sky_median, sky_rms.
    All per-source arrays are ordered by DAOStarFinder peak (descending).
    """
    ext = 1 if filename.endswith(".fits.fz") else 0
    with fits.open(filename) as hdul:
        hdr     = hdul[ext].header
        img_raw = np.array(hdul[ext].data, dtype=float)

    obs_loc = EarthLocation.from_geodetic(
        lon=longitude * u.deg, lat=latitude * u.deg, height=altitude * u.m)
    obstime     = Time(hdr[time_key], format="isot", scale="utc",
                       location=obs_loc)
    object_name = hdr.get(object_key, "")
    inst_mode   = str(hdr.get("INSTMODE", "PHOT")).upper().strip()
    if inst_mode not in ("POLAR", "PHOT"):
        inst_mode = "PHOT"   # unknown values treated as PHOT

    # Background
    bkg = Background2D(img_raw, (50, 50), filter_size=(5, 5),
                       sigma_clip=SigmaClip(sigma=3.0),
                       bkg_estimator=MedianBackground())
    sky_rms  = float(bkg.background_rms_median)
    rms_map  = bkg.background_rms
    img_sub  = img_raw - bkg.background

    # Error array
    read_noise = float(hdr[read_noise_key]) if read_noise_key in hdr else 0.0
    err_data   = np.sqrt(np.abs(img_raw) + read_noise**2 + rms_map**2)

    # Source detection
    daofind = DAOStarFinder(threshold=threshold * sky_rms,
                             fwhm=fwhm_for_detection,
                             min_separation=fwhm_for_detection)
    sources = daofind.find_stars(img_sub)

    _empty = {
        "date_obs": hdr[time_key], "jd": float(obstime.jd),
        "object_name": object_name, "inst_mode": inst_mode, "n_sources": 0,
        "fwhm": np.nan, "fwhm_x": np.nan, "fwhm_y": np.nan,
        "fwhm_err": np.nan, "fwhm_x_err": np.nan, "fwhm_y_err": np.nan,
        "fwhms_x": np.array([]), "fwhms_y": np.array([]),
        "xcentroids": np.array([]), "ycentroids": np.array([]),
        "fluxes": np.array([]), "fluxerrs": np.array([]),
        "max_counts": np.array([]),
        "sky_median": float(bkg.background_median), "sky_rms": sky_rms,
    }
    if sources is None or len(sources) == 0:
        if verbose:
            print(f"  No sources detected in {os.path.basename(filename)}")
        return _empty

    positions = list(zip(sources["xcentroid"], sources["ycentroid"]))
    if verbose:
        print(f"  {len(sources)} sources detected "
              f"(threshold={threshold*sky_rms:.1f} e-)")

    # ── Local annulus background photometry ───────────────────────────────
    # Inner/outer radii: 1.5× and 2.5× the aperture radius.
    # For each source: measure sky = median(sigma-clipped annulus pixels),
    # subtract sky_per_pixel × n_aperture_pixels from the aperture sum.
    # All ApertureStats quantities are stripped of astropy units with .value
    # immediately after access to avoid Quantity arithmetic errors.
    r_in  = aperture_radius * 1.5
    r_out = aperture_radius * 2.5

    apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
    annuli    = photutils.aperture.CircularAnnulus(positions,
                                                    r_in=r_in, r_out=r_out)

    # Aperture stats on the raw image (needed for peak counts and Poisson errors)
    raw_stats = photutils.aperture.ApertureStats(img_raw, apertures,
                                                  error=err_data)

    # Per-source local sky from sigma-clipped annulus pixels
    ann_stats = photutils.aperture.ApertureStats(
        img_raw, annuli,
        sigma_clip=SigmaClip(sigma=3.0),
    )

    # Strip astropy units from every quantity we use
    def _val(x):
        """Return plain numpy array from an ApertureStats attribute."""
        return np.asarray(getattr(x, 'value', x), dtype=float)

    sky_per_px    = _val(ann_stats.median)       # ADU/pixel, shape (n_sources,)
    ann_std       = _val(ann_stats.std)           # scatter in annulus
    ann_n_px      = _val(ann_stats.sum_aper_area) # unmasked annulus pixels
    ap_sum_raw    = _val(raw_stats.sum)           # raw aperture sum
    ap_sum_err    = _val(raw_stats.sum_err)       # Poisson+RN aperture error
    ap_peak       = _val(raw_stats.max)           # peak pixel in aperture

    n_ap_px = float(apertures.area)              # pi * r^2, scalar

    # Replace NaN/negative sky values with the global sky estimate
    bad_sky = ~np.isfinite(sky_per_px) | (sky_per_px < 0)
    sky_per_px[bad_sky] = float(bkg.background_median)

    # Local-sky-subtracted flux
    fluxes = ap_sum_raw - sky_per_px * n_ap_px

    # Sky subtraction uncertainty per source:
    #   sigma_sky_per_px = std(annulus) / sqrt(n_annulus_pixels)
    #   sigma_sky_total  = sigma_sky_per_px * n_aperture_pixels
    n_ann_safe     = np.where(ann_n_px > 0, ann_n_px, 1.0)
    std_safe       = np.where(np.isfinite(ann_std), ann_std, sky_rms)
    sky_err_per_px = std_safe / np.sqrt(n_ann_safe)
    sky_total_err  = sky_err_per_px * n_ap_px

    # Replace NaN aperture errors with a sky-based estimate
    ap_sum_err_safe = np.where(np.isfinite(ap_sum_err), ap_sum_err,
                               np.sqrt(np.abs(ap_sum_raw)))
    fluxerrs = np.sqrt(ap_sum_err_safe**2 + sky_total_err**2)

    if verbose:
        med_sky = float(np.nanmedian(sky_per_px))
        print(f"  Local sky median : {med_sky:.1f} ADU/px  "
              f"(global was {float(bkg.background_median):.1f})")

    # FWHM
    fwhms_x, fwhms_y, xc, yc = _measure_fwhm(
        img_sub, positions, err_data=err_data,
        window_size=window_size, global_fit=psf_global_fit)
    fwhms_x = np.asarray(fwhms_x, dtype=float)
    fwhms_y = np.asarray(fwhms_y, dtype=float)

    mfwhm_x    = float(np.nanmedian(fwhms_x))
    mfwhm_y    = float(np.nanmedian(fwhms_y))
    # ── FWHM error: MAD-based scatter across sources.  When only one source
    # is detected MAD = 0, which gives a meaningless zero error.  Fall back
    # to 10% of the FWHM as a conservative single-source uncertainty.
    n_valid_x = int(np.sum(np.isfinite(fwhms_x)))
    n_valid_y = int(np.sum(np.isfinite(fwhms_y)))
    if n_valid_x > 1:
        fwhm_x_err = float(np.nanmedian(np.abs(fwhms_x - mfwhm_x)) / 0.67449)
    else:
        fwhm_x_err = float(mfwhm_x * 0.10)   # 10% fallback for single source
    if n_valid_y > 1:
        fwhm_y_err = float(np.nanmedian(np.abs(fwhms_y - mfwhm_y)) / 0.67449)
    else:
        fwhm_y_err = float(mfwhm_y * 0.10)

    return {
        "date_obs":    hdr[time_key],
        "jd":          float(obstime.jd),
        "object_name": object_name,
        "inst_mode":   inst_mode,
        "fwhm":        (mfwhm_x + mfwhm_y) / 2.0,
        "fwhm_x":      mfwhm_x,
        "fwhm_y":      mfwhm_y,
        "fwhm_err":    float(np.sqrt(fwhm_x_err**2 + fwhm_y_err**2)),
        "fwhm_x_err":  fwhm_x_err,
        "fwhm_y_err":  fwhm_y_err,
        "fwhms_x":     fwhms_x,
        "fwhms_y":     fwhms_y,
        "n_sources":   len(sources),
        "xcentroids":  np.asarray(xc, dtype=float),
        "ycentroids":  np.asarray(yc, dtype=float),
        "fluxes":      np.asarray(fluxes,   dtype=float),
        "fluxerrs":    np.asarray(fluxerrs, dtype=float),
        "max_counts":  np.asarray(ap_peak,       dtype=float),
        "sky_median":  float(np.nanmedian(sky_per_px)),
        "sky_rms":     float(np.nanmedian(sky_err_per_px)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interactive source selection
# ─────────────────────────────────────────────────────────────────────────────

def _get_jd_from_header(filepath: str) -> float:
    """Read JD from a FITS file header (fast, no full photometry)."""
    ext = 1 if filepath.endswith(".fits.fz") else 0
    try:
        hdr = fits.getheader(filepath, ext=ext)
        from astropy.time import Time
        return float(Time(hdr["DATE-OBS"], format="isot", scale="utc").jd)
    except Exception:
        return float("nan")


def select_sources_interactive(
    ref_filepath: str,
    ref_result:   dict,
    platescale:   float = 0.335,
    all_channel_data: Optional[Dict[int, tuple]] = None,
) -> Tuple[Optional[int], List[int], Dict[int, Optional[int]], Dict[int, List[int]]]:
    """Four-panel interactive source selection across all available channels.

    Shows one panel per channel, each displaying the frame whose observation
    time is closest to the reference channel's first frame.  The user clicks
    once to select sources; the selection is propagated to every other panel
    automatically by nearest-neighbour position matching within 20 px.
    The user can also click directly in any panel to override the auto-match.

    Parameters
    ----------
    ref_filepath : str
        Path to the reference channel's first FITS file.
    ref_result : dict
        Output of ``measure_raw_photometry`` for ``ref_filepath``.
    platescale : float
        Arcsec per pixel (used in the window title only).
    all_channel_data : dict, optional
        {channel: (filepath, result_dict)} for all available channels,
        including the reference channel.  When None, only one panel is shown.

    Returns
    -------
    target_ref_id : int or None
        ref_id of the target in the reference channel.
    comp_ref_ids : list of int
        ref_ids of the comparisons in the reference channel.
    per_ch_target : dict {ch: ref_id or None}
        Per-channel target ref_ids (matched by position to the ref selection).
    per_ch_comps : dict {ch: list of ref_ids}
        Per-channel comparison ref_ids.
    """
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle as MplCircle

    TARGET_COLOR = "lime"
    COMP_COLOR   = "red"
    UNSEL_COLOR  = "gray"
    MATCH_RADIUS = 20.0   # px; generous for cross-channel propagation

    # ── Collect per-channel images and source lists ───────────────────────
    if all_channel_data is None:
        all_channel_data = {}
    # Ensure the reference channel is always present
    if not all_channel_data:
        # Infer ref_ch from ref_filepath — use channel 1 as safe default
        ref_ch = 1
        all_channel_data[ref_ch] = (ref_filepath, ref_result)
    else:
        ref_ch = min(all_channel_data.keys())
        all_channel_data[ref_ch] = (ref_filepath, ref_result)

    chan_imgs: Dict[int, tuple] = {}   # ch -> (img_sub, vmin, vmax)
    chan_xs:   Dict[int, np.ndarray] = {}
    chan_ys:   Dict[int, np.ndarray] = {}

    for ch, (fp, res) in sorted(all_channel_data.items()):
        try:
            ext = 1 if fp.endswith(".fits.fz") else 0
            with fits.open(fp) as hdul:
                img = np.array(hdul[ext].data, dtype=float)
            bkg = Background2D(img, (50, 50), filter_size=(5, 5),
                               sigma_clip=SigmaClip(sigma=3.0),
                               bkg_estimator=MedianBackground())
            sub  = img - bkg.background
            vmin = float(np.nanpercentile(sub,  1))
            vmax = float(np.nanpercentile(sub, 99))
            chan_imgs[ch] = (sub, vmin, vmax)
            chan_xs[ch]   = np.asarray(res["xcentroids"], dtype=float)
            chan_ys[ch]   = np.asarray(res["ycentroids"], dtype=float)
        except Exception as exc:
            print(f"  [warn] ch{ch} image load failed: {exc}")

    channels_avail = sorted(chan_imgs.keys())
    nch   = len(channels_avail)
    ncols = min(nch, 2)
    nrows = (nch + 1) // 2

    # click_orders[ch] = flat list of source indices in click order.
    # In PHOT mode: [target, C1, C2, ...]
    # In POLAR mode: [tgt_b0, tgt_b1, C1_b0, C1_b1, C2_b0, C2_b1, ...]
    # Every single click propagates to other channels identically.
    # Pairs are formed at refresh/extract time by taking consecutive entries.
    click_orders: Dict[int, list] = {ch: [] for ch in channels_avail}
    done = [False]

    # ── Detect instrument mode from reference result ─────────────────────
    inst_mode = ref_result.get("inst_mode", "PHOT").upper()
    is_polar  = (inst_mode == "POLAR")

    # Beam offset per channel — computed from clicks 0 and 1 of the target.
    # Used only for the status bar hint; propagation uses PHOT nearest-neighbour.
    beam_offset_ch: Dict[int, Optional[tuple]] = {ch: None for ch in channels_avail}

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(nrows, ncols,
                             figsize=(min(8 * ncols, 20), min(7 * nrows, 18)),
                             squeeze=False)
    fig.patch.set_facecolor("black")
    if is_polar:
        mode_hint = (
            "POLAR MODE  —  click beam-0 then beam-1 of TARGET (both green)  |  "
            "then click beam-0 of each COMP (beam-1 auto-found)  |  "
            "Enter or close to confirm"
        )
    else:
        mode_hint = (
            "Click TARGET (green) then comparisons (red) in any panel  |  "
            "Selection propagates automatically  |  Enter or close to confirm"
        )
    fig.suptitle(f"Source selection  [{inst_mode}]\n{mode_hint}",
                 color="white", fontsize=11, fontweight="bold")

    ax_map: Dict[int, plt.Axes] = {}
    circles_map:    Dict[int, list] = {}
    idx_labels_map: Dict[int, list] = {}
    role_texts_map: Dict[int, dict] = {}

    for idx, ch in enumerate(channels_avail):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        ax.set_facecolor("black")
        sub, vmin, vmax = chan_imgs[ch]
        ax.imshow(sub, origin="lower", cmap="gray",
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"ch{ch}  —  {CHANNEL_LABELS.get(ch,'?')} band",
                     color="white", fontsize=12, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white")
        ax_map[ch] = ax

        xs_ch = chan_xs[ch]
        ys_ch = chan_ys[ch]
        circs, ilabels = [], []
        for i, (x, y) in enumerate(zip(xs_ch, ys_ch)):
            c = MplCircle((x, y), radius=16, lw=1.5,
                          edgecolor=UNSEL_COLOR, facecolor="none", zorder=4)
            ax.add_patch(c)
            t = ax.text(x + 18, y + 18, str(i),
                        color="white", fontsize=10, fontweight="bold", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground="black")])
            circs.append(c); ilabels.append(t)
        circles_map[ch]    = circs
        idx_labels_map[ch] = ilabels
        role_texts_map[ch] = {}

    # Hide unused subplots
    for idx in range(nch, nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row][col].set_visible(False)

    # Status bar in the first panel
    status = ax_map[channels_avail[0]].text(
        0.01, 0.01,
        f"Click to select TARGET then comparisons.",
        transform=ax_map[channels_avail[0]].transAxes,
        color="yellow", fontsize=10, fontweight="bold", va="bottom",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # ── Label refresh ─────────────────────────────────────────────────────
    def _refresh_all():
        """Redraw circles and role labels on every panel."""
        for ch in channels_avail:
            xs_ch  = chan_xs[ch]
            ys_ch  = chan_ys[ch]
            circs   = circles_map[ch]
            ilabels = idx_labels_map[ch]
            rtexts  = role_texts_map[ch]
            n_ch    = len(xs_ch)

            # Build a flat set of (local_idx, col, role_label) entries
            # Works for both PHOT (click_orders) and POLAR (polar_orders)
            labels: dict = {}   # local_idx -> (col, role_str, lw, radius)

            if is_polar:
                # Pairs are consecutive entries: [b0, b1, b0, b1, ...]
                order = click_orders[ch]
                for k, idx in enumerate(order):
                    pair_rank = k // 2   # 0=target, 1=C1, ...
                    beam      = k %  2   # 0=beam0,  1=beam1
                    if pair_rank == 0:
                        col  = TARGET_COLOR
                        base = "Target"
                    else:
                        col  = COMP_COLOR
                        base = f"C{pair_rank}"
                    beam_lbl = "B0" if beam == 0 else "B1"
                    # Partially-selected (odd number of clicks): last b0 pending
                    if k == len(order) - 1 and len(order) % 2 == 1:
                        labels[idx] = (col, f"{base} B0?", 2.5, 18)
                    else:
                        labels[idx] = (col, f"{base} {beam_lbl}", 3.0, 18)
            else:
                order = click_orders[ch]
                for rank, idx in enumerate(order):
                    col  = TARGET_COLOR if rank == 0 else COMP_COLOR
                    role = "Target"    if rank == 0 else f"C{rank}"
                    labels[idx] = (col, role, 3.0, 18)

            for i in range(n_ch):
                if i in labels:
                    col, role, lw, rad = labels[i]
                    circs[i].set_edgecolor(col)
                    circs[i].set_linewidth(lw)
                    circs[i].set_radius(rad)
                    ilabels[i].set_color(col)
                    if i in rtexts:
                        rtexts[i].remove()
                    rtexts[i] = ax_map[ch].text(
                        xs_ch[i] + 18, ys_ch[i] - 24, role,
                        color=col, fontsize=11, fontweight="bold", zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5,
                                                    foreground="black")])
                else:
                    circs[i].set_edgecolor(UNSEL_COLOR)
                    circs[i].set_linewidth(1.5)
                    circs[i].set_radius(16)
                    ilabels[i].set_color("white")
                    if i in rtexts:
                        rtexts[i].remove()
                        del rtexts[i]

        # Status message
        if is_polar:
            order  = click_orders[ref_ch]
            n_clks = len(order)
            n_pairs = n_clks // 2
            pending = (n_clks % 2 == 1)   # odd click count = waiting for partner
            if n_clks == 0:
                msg = "POLAR: click beam-0 of TARGET."
            elif n_clks == 1:
                msg = "Target beam-0 selected — now click target beam-1."
            elif pending:
                msg = (f"{n_pairs} pair(s) + beam-0 pending. "
                       "Click its beam-1.")
            else:
                n_comps = n_pairs - 1
                msg = (f"Target + {n_comps} comp(s) selected. "
                       "Click more comps (beam-0 then beam-1) or close.")
        else:
            ref_order = click_orders[ref_ch]
            n_sel = len(ref_order)
            if n_sel == 0:
                msg = "Click to select TARGET then comparisons."
            elif n_sel == 1:
                msg = f"Target = src {ref_order[0]}.  Click comparisons (or close)."
            else:
                comps = ", ".join(f"src {c}" for c in ref_order[1:])
                msg = f"Target = src {ref_order[0]}   |   {n_sel-1} comp(s): {comps}"
        status.set_text(msg)
        fig.canvas.draw_idle()

    # ── Cross-channel propagation ─────────────────────────────────────────
    def _propagate_to_other_channels(clicked_ch: int, local_idx: int,
                                      rank: int):
        """After a click in clicked_ch, find and select the nearest source
        in every other channel at the same pixel position.

        rank 0 = target, rank >= 1 = comparison C{rank}.
        """
        x_ref = chan_xs[clicked_ch][local_idx]
        y_ref = chan_ys[clicked_ch][local_idx]
        for ch in channels_avail:
            if ch == clicked_ch:
                continue
            xs_ch = chan_xs[ch]
            ys_ch = chan_ys[ch]
            if len(xs_ch) == 0:
                    continue
            dists = np.hypot(xs_ch - x_ref, ys_ch - y_ref)
            j = int(np.argmin(dists))
            if dists[j] > MATCH_RADIUS:
                    continue   # no match in this channel — skip
            order = click_orders[ch]
            # Place at the same rank as in the clicked channel
            # First remove any existing entry for this source
            if j in order:
                order.remove(j)
            # Insert at the correct rank position
            if rank == 0:
                order.insert(0, j)
            else:
                # Ensure target slot is preserved; append comp
                while len(order) < rank:
                    order.append(-1)   # placeholder
                if rank <= len(order):
                    order.insert(rank, j)
                else:
                    order.append(j)
            # Clean up placeholder -1 entries
            click_orders[ch] = [x for x in order if x >= 0]

    def _remove_from_other_channels(clicked_ch: int, local_idx: int):
        """When a source is deselected in clicked_ch, remove nearest match
        from other channels too."""
        x_ref = chan_xs[clicked_ch][local_idx]
        y_ref = chan_ys[clicked_ch][local_idx]
        for ch in channels_avail:
            if ch == clicked_ch:
                continue
            xs_ch = chan_xs[ch]
            ys_ch = chan_ys[ch]
            if len(xs_ch) == 0:
                continue
            dists = np.hypot(xs_ch - x_ref, ys_ch - y_ref)
            j = int(np.argmin(dists))
            if dists[j] <= MATCH_RADIUS and j in click_orders[ch]:
                click_orders[ch].remove(j)

    # ── Polar helpers ─────────────────────────────────────────────────────
    def _find_beam_partner(ch: int, b0_idx: int) -> Optional[int]:
        """Find the beam-1 partner of b0_idx using the channel beam offset."""
        boff = beam_offset_ch[ch]
        if boff is None:
            return None
        dx, dy = boff
        xs_ch = chan_xs[ch]; ys_ch = chan_ys[ch]
        tx = xs_ch[b0_idx] + dx
        ty = ys_ch[b0_idx] + dy
        dists = np.hypot(xs_ch - tx, ys_ch - ty)
        j = int(np.argmin(dists))
        return j if dists[j] <= 20.0 else None

    # ── Click handler ─────────────────────────────────────────────────────
    def _click(event):
        if done[0] or event.inaxes not in ax_map.values():
            return
        clicked_ch = next(ch for ch, ax in ax_map.items()
                          if ax is event.inaxes)
        xs_ch = chan_xs[clicked_ch]
        ys_ch = chan_ys[clicked_ch]
        if len(xs_ch) == 0:
            return
        dists = np.hypot(xs_ch - event.xdata, ys_ch - event.ydata)
        i = int(np.argmin(dists))
        if dists[i] > 25:
            return

        if is_polar:
            order = click_orders[clicked_ch]
            rank  = len(order)   # position this click would occupy

            if i in order:
                # Deselect: remove only from THIS channel.
                # Other channels keep their own independent selections.
                idx_in_order = order.index(i)
                partner_idx  = idx_in_order ^ 1   # pair partner: 0↔1, 2↔3, …
                remove = {i}
                if 0 <= partner_idx < len(order):
                    remove.add(order[partner_idx])
                click_orders[clicked_ch] = [x for x in order if x not in remove]
            else:
                # New click: append to this channel
                order.append(i)
                # Auto-propagate to other channels ONLY from the reference channel
                if clicked_ch == channels_avail[0]:
                    _propagate_to_other_channels(clicked_ch, i, rank)

                if rank % 2 == 1:
                    # Odd rank = beam-1 click: record beam offset from this pair
                    b0 = order[-2]; b1 = order[-1]
                    dx = float(xs_ch[b1] - xs_ch[b0])
                    dy = float(ys_ch[b1] - ys_ch[b0])
                    beam_offset_ch[clicked_ch] = (dx, dy)
                    if rank == 1:
                        print(f"  [POLAR] ch{clicked_ch} beam offset: "
                              f"dx={dx:+.1f}  dy={dy:+.1f} px")
                elif rank % 2 == 0 and rank >= 2:
                    # Even rank >= 2 = beam-0 of a comp: auto-find beam-1
                    b1 = _find_beam_partner(clicked_ch, i)
                    if b1 is not None and b1 != i and b1 not in order:
                        order.append(b1)
                        # Also propagate the auto-found beam-1 from ref channel
                        if clicked_ch == channels_avail[0]:
                            _propagate_to_other_channels(clicked_ch, b1, rank + 1)
                        print(f"  [POLAR] ch{clicked_ch} auto beam-1={b1} "
                              f"for comp beam-0={i}")
        else:
            order = click_orders[clicked_ch]
            if i in order:
                # Deselect from this channel only; other channels unaffected
                order.remove(i)
            else:
                rank = len(order)
                order.append(i)
                # Only propagate from the reference channel
                if clicked_ch == channels_avail[0]:
                    _propagate_to_other_channels(clicked_ch, i, rank)

        _refresh_all()

    def _key(event):
        if event.key == "enter":
            done[0] = True
            plt.close(fig)

    def _close(event):
        done[0] = True

    fig.canvas.mpl_connect("button_press_event", _click)
    fig.canvas.mpl_connect("key_press_event",    _key)
    fig.canvas.mpl_connect("close_event",        _close)
    # Use subplots_adjust instead of tight_layout to prevent the
    # window manager from resizing the window to a small size.
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88,
                        bottom=0.03, wspace=0.06, hspace=0.08)
    # Maximise the window before showing — works on TkAgg / Qt5Agg
    try:
        mgr = plt.get_current_fig_manager()
        try:
            mgr.window.state("zoomed")   # TkAgg on Windows (titlebar stays)
        except Exception:
            try:
                mgr.window.showMaximized()   # Qt backends
            except Exception:
                pass   # leave at the figsize set above
    except Exception:
        pass
    plt.show(block=True)

    # ── Extract final selections as pixel coordinates ────────────────────
    # Return pixel coordinates (not local ref_ids) so _resolve_selection_for_channel
    # can find the correct ref_ids in each channel's own catalogue.
    #
    # PHOT mode: per_ch_sel_xy[ch] = [(x,y), ...]
    #            first entry = target, rest = comps
    #
    # POLAR mode: per_ch_sel_xy[ch] = [((x0,y0),(x1,y1)), ...]
    #             each entry is a beam-pair; first pair = target
    #
    # Also return inst_mode and per-channel beam offsets so the monitor
    # can store them in the DB and use them for subsequent frame matching.
    per_ch_sel_xy: Dict[int, list] = {}

    if is_polar:
        for ch in channels_avail:
            xs_ch = chan_xs[ch]; ys_ch = chan_ys[ch]
            order = click_orders.get(ch, [])
            coords = []
            # Pair up consecutive entries: [b0,b1, b0,b1, ...]
            for k in range(0, len(order) - 1, 2):
                b0, b1 = order[k], order[k+1]
                if 0 <= b0 < len(xs_ch) and 0 <= b1 < len(xs_ch):
                    coords.append(((float(xs_ch[b0]), float(ys_ch[b0])),
                                   (float(xs_ch[b1]), float(ys_ch[b1]))))
            per_ch_sel_xy[ch] = coords
    else:
        for ch in channels_avail:
            order = click_orders.get(ch, [])
            xs_ch = chan_xs[ch]; ys_ch = chan_ys[ch]
            coords = []
            for idx in order:
                if 0 <= idx < len(xs_ch):
                    coords.append((float(xs_ch[idx]), float(ys_ch[idx])))
            per_ch_sel_xy[ch] = coords

    print(f"[monitor] Selection coords ({inst_mode}):")
    for ch in channels_avail:
        coords = per_ch_sel_xy[ch]
        if is_polar:
            for k, pair in enumerate(coords):
                role = "Target" if k == 0 else f"C{k}"
                (x0, y0), (x1, y1) = pair
                print(f"  ch{ch}  {role:8s}  B0=({x0:.1f},{y0:.1f})  "
                      f"B1=({x1:.1f},{y1:.1f})")
        else:
            roles = ["Target"] + [f"C{i}" for i in range(1, len(coords))]
            for role, (x, y) in zip(roles, coords):
                print(f"  ch{ch}  {role:8s}  x={x:.1f}  y={y:.1f}")

    return per_ch_sel_xy, inst_mode, beam_offset_ch



# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_fits_files(data_dir: str, seq_suffix: str,
                    object_filter: Optional[str] = None) -> List[str]:
    """Return sorted FITS file paths matching suffix and optional OBJECT filter."""
    files = sorted(glob.glob(os.path.join(data_dir, f"*_{seq_suffix}.fits")))
    if not object_filter:
        return files
    matched = []
    for f in files:
        try:
            obj = fits.getheader(f, ext=0).get("OBJECT", "")
            if object_filter.lower() in obj.lower():
                matched.append(f)
        except Exception:
            pass
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# ZeroMQ publisher
# ─────────────────────────────────────────────────────────────────────────────

def _make_publisher(port: int):
    """Create and return a bound ZMQ PUB socket, or None if zmq unavailable."""
    if not _HAS_ZMQ:
        return None
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    time.sleep(0.2)   # give subscribers time to connect
    print(f"[monitor] ZeroMQ PUB socket bound on tcp://*:{port}")
    return sock


def _publish(sock, channel: int, result: dict, ref_ids: np.ndarray,
             platescale: float, target_id=None, comp_ids=None,
             beam_ids=None, beam_offset=None) -> None:
    """Publish one frame result as a JSON message."""
    if comp_ids is None:
        comp_ids = []
    if sock is None:
        return

    inst_mode = result.get("inst_mode", "PHOT")
    fluxes   = np.asarray(result.get("fluxes",     []), dtype=float)
    fluxerrs = np.asarray(result.get("fluxerrs",   []), dtype=float)
    peaks    = np.asarray(result.get("max_counts", []), dtype=float)
    xs       = np.asarray(result.get("xcentroids", []), dtype=float)
    ys       = np.asarray(result.get("ycentroids", []), dtype=float)

    # Only publish the selected sources (target + comparisons).
    # If no selection has been made yet, publish all matched sources.
    selected_ids: set = set()
    if target_id is not None:
        selected_ids.add(int(target_id))
    for cid in (comp_ids or []):
        selected_ids.add(int(cid))

    sources_list = []
    for i, rid in enumerate(ref_ids):
        if rid < 0:
            continue
        if selected_ids and int(rid) not in selected_ids:
            continue   # skip sources not in the selection
        beam = (int(beam_ids[i])
                if beam_ids is not None and i < len(beam_ids)
                else None)
        sources_list.append({
            "ref_id":   int(rid),
            "beam":     beam,
            "x":        float(xs[i])       if i < len(xs)       else None,
            "y":        float(ys[i])       if i < len(ys)       else None,
            "flux":     float(fluxes[i])   if i < len(fluxes)   else None,
            "flux_err": float(fluxerrs[i]) if i < len(fluxerrs) else None,
            "peak":     float(peaks[i])    if i < len(peaks)    else None,
        })
    # Sort: target first, then comps in selection order
    order_map = {}
    if target_id is not None:
        order_map[int(target_id)] = 0
    for rank, cid in enumerate(comp_ids or [], start=1):
        order_map[int(cid)] = rank
    sources_list.sort(key=lambda s: order_map.get(s["ref_id"], 999))

    fwhm_pix = result.get("fwhm", np.nan)
    payload = {
        "channel":      channel,
        "band":         CHANNEL_LABELS[channel],
        "filename":     os.path.basename(result.get("date_obs", "")),
        "date_obs":     result.get("date_obs", ""),
        "jd":           result.get("jd"),
        "object_name":  result.get("object_name", ""),
        "inst_mode":    inst_mode,
        "fwhm_pix":     float(fwhm_pix) if np.isfinite(fwhm_pix) else None,
        "fwhm_arcsec":  float(fwhm_pix * platescale) if np.isfinite(fwhm_pix) else None,
        "fwhm_err_pix": result.get("fwhm_err"),
        "n_sources":    result.get("n_sources", 0),
        "platescale":   platescale,
        "target_id":    target_id,
        "comp_ids":     comp_ids,
        "beam_offset":  list(beam_offset) if beam_offset is not None else None,
        "sources":      sources_list,
    }

    topic = f"sparc4.ch{channel}".encode()
    sock.send_multipart([topic, json.dumps(payload).encode()])


# ─────────────────────────────────────────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────────────────────────────────────────

def run_monitor(
    data_dirs: Dict[int, str],
    seq_suffix: str,
    *,
    object_filter: Optional[str] = None,
    db_dir: str = "",
    platescale: float = 0.335,
    threshold: float = 10.0,
    fwhm_estimate: float = 5.0,
    window_size: int = 25,
    select_sources: bool = False,
    watch: bool = False,
    interval: float = 30.0,
    zmq_port: int = ZMQ_DEFAULT_PORT,
    verbose: bool = False,
) -> None:
    """Main monitoring and publishing loop.

    Parameters
    ----------
    data_dirs : dict[int, str]
        Channel number → raw data directory for the night.
        e.g. {1: "/data/sparc4acs1/today", 2: "/data/sparc4acs2/today"}
    seq_suffix : str
        Filename suffix for science frames (e.g. "cr3").
    object_filter : str, optional
        Only process files whose FITS OBJECT keyword contains this string.
    db_dir : str
        Directory for SQLite files.  Defaults to first channel data dir.
    platescale : float
        Arcsec per pixel.
    threshold : float
        Detection threshold in units of background σ.
    fwhm_estimate : float
        Approximate PSF FWHM in pixels (DAOStarFinder + match radius).
    window_size : int
        Cutout half-size for Gaussian FWHM fitting.
    select_sources : bool
        If True, open interactive source-selection on the first image.
    watch : bool
        If True, keep polling for new files until Ctrl-C.
    interval : float
        Seconds between polls in watch mode.
    zmq_port : int
        TCP port for the ZeroMQ PUB socket.
    verbose : bool
        Print per-image diagnostics.
    """
    channels     = sorted(data_dirs.keys())
    match_radius = fwhm_estimate

    # ── Databases and source matchers ──────────────────────────────────────
    if not db_dir:
        db_dir = data_dirs[channels[0]]
    os.makedirs(db_dir, exist_ok=True)

    dbs:      Dict[int, PhotometryDB]  = {}
    matchers: Dict[int, SourceMatcher] = {}
    for ch in channels:
        db_path      = os.path.join(db_dir, f"monitor_ch{ch}.db")
        dbs[ch]      = PhotometryDB(db_path)
        matchers[ch] = SourceMatcher(match_radius=match_radius, verbose=verbose)
        print(f"[monitor] ch{ch} ({CHANNEL_LABELS[ch]})  db → {db_path}")

    # ── Restore each channel's own reference catalogue ────────────────────
    for ch in channels:
        if dbs[ch].has_reference_catalogue():
            rxs, rys = dbs[ch].load_reference_catalogue()
            matchers[ch].set_reference(rxs, rys)
            print(f"[monitor] ch{ch} reference catalogue restored: {len(rxs)} sources")

    # ── ZeroMQ publisher ──────────────────────────────────────────────────
    sock = _make_publisher(zmq_port)

    # ── Source selection state ────────────────────────────────────────────
    # sel_xy[ch] = [(x,y), ...] in click order — pixel coords from the
    # selection window, channel-specific.  When each channel's reference
    # catalogue is first established, _resolve_selection_for_channel()
    # converts these coords to stable ref_ids for that channel.
    #
    # target_ref_id / comp_ref_ids  — reference channel's ref_ids
    # per_ch_target / per_ch_comps  — per-channel ref_ids (set once catalogue is ready)
    sel_xy:          Dict[int, list]          = {}   # ch -> [(x,y),...] or [((x0,y0),(x1,y1)),...]
    polar_mode:      bool                     = False
    ch_beam_offsets: Dict[int, Optional[tuple]] = {ch: None for ch in channels}
    target_ref_id: Optional[int]            = None
    comp_ref_ids:  List[int]                = []
    per_ch_target: Dict[int, Optional[int]] = {ch: None for ch in channels}
    per_ch_comps:  Dict[int, List[int]]     = {ch: []   for ch in channels}
    source_selection_done = False

    if dbs[channels[0]].has_selection():
        target_ref_id, comp_ref_ids, bm0 = dbs[channels[0]].load_selection()
        source_selection_done = True
        # Detect polar mode from stored beam_map
        if any(v is not None for v in bm0.values()):
            polar_mode = True
        for ch in channels:
            if dbs[ch].has_selection():
                t, c, bm = dbs[ch].load_selection()
                per_ch_target[ch] = t
                per_ch_comps[ch]  = c
            if dbs[ch].has_beam_offset():
                dx, dy = dbs[ch].load_beam_offset()
                ch_beam_offsets[ch] = (dx, dy)
        mode_str = "POLAR" if polar_mode else "PHOT"
        print(f"[monitor] Selection restored ({mode_str}) — "
              f"target={target_ref_id}  comps={comp_ref_ids}")

    def _resolve_selection_for_channel(ch: int,
                                        cat_xs: np.ndarray,
                                        cat_ys: np.ndarray) -> None:
        """Map selection pixel coords to ref_ids in this channel's catalogue.

        Called once per channel when its reference catalogue is first established.
        Handles both PHOT mode (single coords) and POLAR mode (beam-pair coords).

        In POLAR mode each selected star has two ref_ids (beam 0 and beam 1).
        The DB selection table stores one row per beam with the beam column set.
        Only beam-0 ref_ids are used as the canonical target/comp identifiers;
        beam-1 partners are looked up from the DB at plot time.
        """
        nonlocal target_ref_id, comp_ref_ids
        coords = sel_xy.get(ch, [])
        if not coords:
            return

        MATCH_R = 20.0

        def _nearest(x, y):
            dists = np.hypot(cat_xs - x, cat_ys - y)
            j = int(np.argmin(dists))
            return int(j) if dists[j] <= MATCH_R else -1

        if polar_mode:
            # coords is a list of ((x0,y0),(x1,y1)) pairs
            beam_map: dict = {}   # ref_id -> beam index
            b0_ids = []           # beam-0 ref_ids in role order
            for pair in coords:
                (x0, y0), (x1, y1) = pair
                r0 = _nearest(x0, y0)
                r1 = _nearest(x1, y1)
                if r0 >= 0:
                    b0_ids.append(r0)
                    beam_map[r0] = 0
                if r1 >= 0 and r1 != r0:
                    # Store beam-1 as a comp entry just after its beam-0 partner
                    beam_map[r1] = 1

            target_b0 = b0_ids[0] if b0_ids else None
            comp_b0s  = b0_ids[1:] if len(b0_ids) > 1 else []

            # Build full ref_id list: target_b0, target_b1, comp1_b0, comp1_b1, ...
            all_ref_ids = []
            for pair_idx, pair in enumerate(coords):
                (x0, y0), (x1, y1) = pair
                r0 = _nearest(x0, y0)
                r1 = _nearest(x1, y1)
                if r0 >= 0:
                    all_ref_ids.append(r0)
                if r1 >= 0 and r1 != r0:
                    all_ref_ids.append(r1)

            per_ch_target[ch] = target_b0
            per_ch_comps[ch]  = comp_b0s

            if ch == channels[0]:
                target_ref_id = target_b0
                comp_ref_ids  = comp_b0s

            # Save selection with beam labels; save beam offset to DB
            dbs[ch].save_selection(target_b0, all_ref_ids[1:],
                                   beam_map=beam_map)
            boff = ch_beam_offsets.get(ch)
            if boff is not None:
                dbs[ch].save_beam_offset(boff[0], boff[1])

            print(f"  [ch{ch}] POLAR selection resolved: "
                  f"target_b0={target_b0}  comps_b0={comp_b0s}  "
                  f"beam_map={beam_map}")
        else:
            # PHOT mode: simple single-coord matching
            resolved = [_nearest(x, y) for (x, y) in coords]
            target_idx = resolved[0] if resolved else -1
            comp_idxs  = [r for r in resolved[1:] if r >= 0]

            per_ch_target[ch] = target_idx if target_idx >= 0 else None
            per_ch_comps[ch]  = comp_idxs

            if ch == channels[0]:
                target_ref_id = per_ch_target[ch]
                comp_ref_ids  = per_ch_comps[ch]

            dbs[ch].save_selection(per_ch_target[ch], per_ch_comps[ch])
            print(f"  [ch{ch}] PHOT selection resolved: "
                  f"target={per_ch_target[ch]}  comps={per_ch_comps[ch]}")

    def _try_source_selection(ref_filepath: str, ref_result: dict) -> None:
        """Open the 4-panel selection window.

        Finds the nearest-in-time file from every other channel and runs
        photometry on it so all panels are populated simultaneously.
        """
        nonlocal source_selection_done, target_ref_id, comp_ref_ids
        nonlocal per_ch_target, per_ch_comps
        if not select_sources or source_selection_done:
            return

        # Build all_channel_data: for each channel find the file whose
        # observation time is closest to the reference channel's first file
        ref_jd = ref_result.get("jd", float("nan"))
        all_channel_data: Dict[int, tuple] = {}
        for ch in channels:
            files = find_fits_files(data_dirs[ch], seq_suffix, object_filter)
            if not files:
                continue
            # Find closest JD
            best_fp, best_dt = None, float("inf")
            for fp in files:
                jd = _get_jd_from_header(fp)
                dt = abs(jd - ref_jd)
                if dt < best_dt:
                    best_dt, best_fp = dt, fp
            if best_fp is None:
                continue
            try:
                res = measure_raw_photometry(
                    best_fp,
                    threshold=threshold,
                    fwhm_for_detection=fwhm_estimate,
                    aperture_radius=aperture_radius,
                    window_size=window_size,
                    verbose=False,
                )
                if res["n_sources"] > 0:
                    all_channel_data[ch] = (best_fp, res)
                    print(f"  [selection] ch{ch}: {os.path.basename(best_fp)} "
                          f"(Δt={best_dt*86400:.0f} s,  "
                          f"{res['n_sources']} sources)")
            except Exception as exc:
                print(f"  [warn] ch{ch} selection frame failed: {exc}")

        # Always include the reference channel's data so sel_xy has the
        # right channel key even when running with only 1 channel.
        all_channel_data[channels[0]] = (ref_filepath, ref_result)
        per_ch_sel_xy, sel_inst_mode, sel_beam_offsets =             select_sources_interactive(
                ref_filepath, ref_result,
                platescale=platescale,
                all_channel_data=all_channel_data,
            )
        # Store the pixel coordinates; ref_ids will be resolved per-channel
        # when each channel's reference catalogue is first established.
        nonlocal sel_xy, polar_mode, ch_beam_offsets
        sel_xy         = per_ch_sel_xy
        polar_mode     = (sel_inst_mode == "POLAR")
        ch_beam_offsets = sel_beam_offsets   # {ch: (dx,dy) or None}
        source_selection_done = True
        mode_str = "POLAR" if polar_mode else "PHOT"
        print(f"[monitor] Selection saved ({mode_str}) — "
              f"ref_ids resolved as each channel catalogue is built.")

    # ── Processing loop ───────────────────────────────────────────────────
    aperture_radius = 3.0 * fwhm_estimate
    print(f"\n[monitor] Aperture  : {aperture_radius:.1f} px")
    print(f"[monitor] FWHM est. : {fwhm_estimate:.1f} px")
    print(f"[monitor] Threshold : {threshold} σ")
    print(f"[monitor] Platescale: {platescale} arcsec/px")
    if object_filter:
        print(f"[monitor] Object    : {object_filter!r}")
    print()

    def _process_frame(ch: int, filepath: str):
        """Run photometry on one FITS file; update DB and publish via ZMQ.

        Each channel is fully independent:
          - Its own SourceMatcher tracks within-channel guiding drift.
          - Its own ref_ids (0 = brightest detected star in its first frame,
            1 = second brightest, …) are local to that channel.
          - ref_ids are published as-is in the ZMQ message; the plot tools
            treat each channel independently.

        Note on cross-channel identification
        -------------------------------------
        Cross-channel source matching (i.e. ensuring ref_id=N means the same
        physical star in all four bands) is intentionally NOT implemented here.
        The four SPARC4 channels can have different exposure times and therefore
        different cadences, making frame-by-frame alignment unreliable.  A safe
        cross-channel identification scheme will be added in a future version.
        For now, the user selects the target and comparisons visually in channel
        1; the plot tools use the same ref_ids for all channels, which works
        correctly because SPARC4's inter-channel pixel offset is < 5 px and the
        per-channel SourceMatcher assigns ref_ids in the same brightness order
        as channel 1 for the same set of stars.
        """
        nonlocal target_ref_id, comp_ref_ids, source_selection_done

        basename = os.path.basename(filepath)
        try:
            result = measure_raw_photometry(
                filepath,
                threshold=threshold,
                fwhm_for_detection=fwhm_estimate,
                aperture_radius=aperture_radius,
                window_size=window_size,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"  [BAD] {basename}: {exc}")
            dbs[ch].insert_bad_frame(filepath, reason=str(exc))
            return None, None

        if result["n_sources"] == 0:
            print(f"  [BAD] {basename}: no sources detected — recorded as bad frame.")
            dbs[ch].insert_bad_frame(
                filepath,
                date_obs=result.get("date_obs", ""),
                jd=result.get("jd"),
                object_name=result.get("object_name", ""),
                reason="no sources detected",
            )
            return None, None

        xs = result["xcentroids"]
        ys = result["ycentroids"]
        is_first = not matchers[ch].has_reference

        if is_first:
            matchers[ch].set_reference(xs, ys)
            dbs[ch].save_reference_catalogue(xs, ys)
            ref_ids = np.arange(len(xs), dtype=int)
            print(f"  [ch{ch}] Reference catalogue: {len(xs)} sources")
            # Open source selection on the first good frame of the first channel
            if ch == channels[0] and not source_selection_done:
                _try_source_selection(filepath, result)
            # Once catalogue is set, resolve selection pixel coords → ref_ids
            if sel_xy:
                _resolve_selection_for_channel(ch, xs, ys)
        else:
            ref_ids = matchers[ch].match(xs, ys)
            if verbose:
                print(f"  [ch{ch}] Matched {int((ref_ids>=0).sum())}/{len(xs)} sources")

        # In POLAR mode, assign beam labels to each detection:
        # beam-0 sources are those whose ref_id is in the beam-0 set for this ch;
        # beam-1 sources are those in the beam-1 set.
        # This requires the beam_map from the DB selection.
        beam_ids_arr = None
        if polar_mode and dbs[ch].has_selection():
            _, _, bm = dbs[ch].load_selection()
            if bm:
                beam_ids_arr = np.array(
                    [bm.get(int(rid), -1) if rid >= 0 else -1
                     for rid in ref_ids], dtype=int)

        dbs[ch].insert_frame(filepath, result, ref_ids, beam_ids=beam_ids_arr)

        pub_target  = per_ch_target.get(ch, target_ref_id)
        pub_comps   = per_ch_comps.get(ch,  comp_ref_ids)
        boff        = ch_beam_offsets.get(ch) if polar_mode else None
        _publish(sock, ch, result, ref_ids, platescale,
                 target_id=pub_target, comp_ids=pub_comps,
                 beam_ids=beam_ids_arr, beam_offset=boff)

        fwhm_as = (result["fwhm"] * platescale
                   if np.isfinite(result["fwhm"]) else float("nan"))
        print(f"  {basename:45s}  "
              f"{result['n_sources']:3d} src  "
              f"FWHM={fwhm_as:.2f}\"  "
              f"peak={np.nanmax(result['max_counts']):.0f}  "
              f"flux={np.nanmax(result['fluxes']):.0f}")
        return result, is_first

    iteration = 0
    while True:
        iteration += 1
        any_new = False

        # ── Process all new files for every channel ────────────────────────
        for ch in channels:
            files     = find_fits_files(data_dirs[ch], seq_suffix, object_filter)
            new_files = [f for f in files if not dbs[ch].is_processed(f)]

            if not new_files:
                if verbose:
                    print(f"[ch{ch}] No new files.")
                continue

            any_new = True
            print(f"[ch{ch}] {len(new_files)} new file(s)...")

            for filepath in new_files:
                _process_frame(ch, filepath)

        if not watch:
            break

        if iteration == 1 and not any_new:
            print(f"[monitor] Waiting for new files. "
                  f"Polling every {interval:.0f} s — Ctrl-C to stop.")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[monitor] Stopped.")
            break

    print("[monitor] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SPARC4 raw photometry monitor (ZeroMQ publisher)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Data directories (one per channel)")
    g.add_argument("--ch1", metavar="DIR", default="",
                   help="Channel 1 (g band) raw data directory")
    g.add_argument("--ch2", metavar="DIR", default="",
                   help="Channel 2 (r band) raw data directory")
    g.add_argument("--ch3", metavar="DIR", default="",
                   help="Channel 3 (i band) raw data directory")
    g.add_argument("--ch4", metavar="DIR", default="",
                   help="Channel 4 (z band) raw data directory")
    g.add_argument("-s", "--seq_suffix", metavar="SUFFIX", default="",
                   help="Filename suffix for science frames, e.g. cr3")
    g.add_argument("-o", "--object", metavar="NAME", default="",
                   help="Filter by OBJECT keyword (case-insensitive substring)")

    g = p.add_argument_group("Photometry parameters")
    g.add_argument("-f", "--fwhm",       type=float, default=5.0,  metavar="PIX",
                   help="Estimated PSF FWHM in pixels")
    g.add_argument("-t", "--threshold",  type=float, default=10.0, metavar="SIGMA",
                   help="Detection threshold above background")
    g.add_argument("-w", "--window",     type=int,   default=25,   metavar="PIX",
                   help="Cutout half-size for Gaussian FWHM fitting")
    g.add_argument("-e", "--platescale", type=float, default=0.335,metavar="AS/PX",
                   help="Plate scale in arcsec/pixel")

    g = p.add_argument_group("Monitor behaviour")
    g.add_argument("--select-sources", action="store_true",
                   help="Open interactive source-selection on the first image")
    g.add_argument("--watch", action="store_true",
                   help="Keep watching for new files (poll loop)")
    g.add_argument("--interval", type=float, default=30.0, metavar="SEC",
                   help="Seconds between directory polls")
    g.add_argument("--db", metavar="DIR", default="",
                   help="SQLite database directory (default: first channel dir)")
    g.add_argument("--port", type=int, default=ZMQ_DEFAULT_PORT, metavar="PORT",
                   help="ZeroMQ PUB socket port")
    g.add_argument("-v", "--verbose", action="store_true")

    opts = p.parse_args()

    if not opts.seq_suffix:
        p.error("--seq_suffix is required (e.g. --seq_suffix cr3)")

    ch_args   = {1: opts.ch1, 2: opts.ch2, 3: opts.ch3, 4: opts.ch4}
    data_dirs = {}
    for ch, d in ch_args.items():
        if d:
            d = os.path.expanduser(d)
            if not os.path.isdir(d):
                print(f"[monitor] WARNING: ch{ch} dir not found: {d}")
            else:
                data_dirs[ch] = d

    if not data_dirs:
        p.error("At least one of --ch1/--ch2/--ch3/--ch4 must be given.")

    print("[monitor] Channels:")
    for ch, d in sorted(data_dirs.items()):
        print(f"  ch{ch} ({CHANNEL_LABELS[ch]})  →  {d}")

    run_monitor(
        data_dirs,
        seq_suffix=opts.seq_suffix,
        object_filter=opts.object or None,
        db_dir=opts.db,
        platescale=opts.platescale,
        threshold=opts.threshold,
        fwhm_estimate=opts.fwhm,
        window_size=opts.window,
        select_sources=opts.select_sources,
        watch=opts.watch,
        interval=opts.interval,
        zmq_port=opts.port,
        verbose=opts.verbose,
    )


if __name__ == "__main__":
    main()
