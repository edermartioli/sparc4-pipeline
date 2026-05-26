"""
sparc4_astro_utils.py
=====================
Astronomical utilities for the SPARC4 monitoring tools.

Provides sunset / sunrise / twilight times for a given observatory
and date, expressed as local decimal hours (0–24) for use as x-axis
markers in the time-series plot tools.

All computations use astropy and require no network access.

Public API
----------
compute_sun_events(date_str, lon_deg, lat_deg, alt_m, utc_offset_h)
    Returns a SunEvents dataclass with sunset, sunrise, and the three
    twilight boundaries (civil, nautical, astronomical) for both evening
    and morning.

add_twilight_lines(axes, sun_events, utc_offset_h)
    Draw vertical reference lines on a list of matplotlib Axes objects.

Observatory defaults
--------------------
OPD_LON, OPD_LAT, OPD_ALT — Pico dos Dias Observatory coordinates.

Author
------
Eder Martioli <martioli@lna.br>
Laboratório Nacional de Astrofísica — LNA/MCTI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Observatory defaults (Pico dos Dias / OPD) ───────────────────────────────
OPD_LON =  -45.5825   # deg E
OPD_LAT =  -22.5344   # deg N
OPD_ALT = 1864.0      # m


# ── Sun altitude thresholds ───────────────────────────────────────────────────
# (degrees above/below horizon)
_SUN_ALTS = {
    "sunset_sunrise":    -0.833,   # geometric horizon + refraction + solar disc
    "civil":            -6.0,
    "nautical":        -12.0,
    "astronomical":    -18.0,
}

# Colours and styles for the vertical lines
_LINE_STYLES = {
    "sunset_sunrise":  dict(color="darkorange", lw=1.4, ls="-",  alpha=0.8,
                             zorder=2),
    "civil":           dict(color="goldenrod",  lw=1.0, ls="--", alpha=0.7,
                             zorder=2),
    "nautical":        dict(color="steelblue",  lw=1.0, ls="--", alpha=0.6,
                             zorder=2),
    "astronomical":    dict(color="mediumpurple", lw=1.0, ls=":",  alpha=0.6,
                             zorder=2),
}

_LINE_LABELS = {
    "sunset_sunrise":  ("Sunset", "Sunrise"),
    "civil":           ("Civil twil.", "Civil twil."),
    "nautical":        ("Nautical twil.", "Nautical twil."),
    "astronomical":    ("Astron. twil.", "Astron. twil."),
}


@dataclass
class SunEvents:
    """Sunset, sunrise, and twilight times as local decimal hours (0–24).

    NaN indicates the event does not occur on this date (e.g. midnight sun).

    Attributes
    ----------
    sunset, sunrise : float
        Sun crosses the geometric horizon (corrected for refraction).
    civil_eve, civil_mor : float
        Civil twilight ends (evening) / begins (morning): sun at −6°.
    nautical_eve, nautical_mor : float
        Nautical twilight: sun at −12°.
    astro_eve, astro_mor : float
        Astronomical twilight: sun at −18°.
    date_str : str
        ISO date string for which the events were computed.
    """
    sunset:       float = float("nan")
    sunrise:      float = float("nan")
    civil_eve:    float = float("nan")
    civil_mor:    float = float("nan")
    nautical_eve: float = float("nan")
    nautical_mor: float = float("nan")
    astro_eve:    float = float("nan")
    astro_mor:    float = float("nan")
    date_str:     str   = ""

    def as_dict(self) -> dict:
        """Return a flat dict of all event times."""
        return {
            "sunset":       self.sunset,
            "sunrise":      self.sunrise,
            "civil_eve":    self.civil_eve,
            "civil_mor":    self.civil_mor,
            "nautical_eve": self.nautical_eve,
            "nautical_mor": self.nautical_mor,
            "astro_eve":    self.astro_eve,
            "astro_mor":    self.astro_mor,
        }


def compute_sun_events(
    date_str:     str,
    lon_deg:      float = OPD_LON,
    lat_deg:      float = OPD_LAT,
    alt_m:        float = OPD_ALT,
    utc_offset_h: float = -3.0,
) -> SunEvents:
    """Compute sunset, sunrise, and twilight times for a given date.

    Uses a fine time-grid search (1-minute resolution) to find the moments
    when the Sun crosses each altitude threshold, which is simple, robust,
    and requires no special ephemeris beyond astropy's built-in solar model.

    Parameters
    ----------
    date_str : str
        ISO date of the **evening** of the night, e.g. ``"2024-06-18"``.
        The search window covers local noon on ``date_str`` to local noon
        the following day, capturing the full night.
    lon_deg, lat_deg, alt_m : float
        Observatory geodetic coordinates.
    utc_offset_h : float
        Local UTC offset in hours (e.g. −3 for Brazil/OPD).

    Returns
    -------
    SunEvents
        All crossing times as local decimal hours (0–24).  Values that
        wrap past midnight remain > 24 so that a sunset at 18 h and
        sunrise at 06 h the next morning are represented as 18 and 30
        (i.e. 6 + 24) — this makes x-axis comparisons straightforward.
    """
    try:
        from astropy.coordinates import (EarthLocation, AltAz, get_sun,
                                          SkyCoord)
        from astropy.time import Time
        import astropy.units as u
    except ImportError:
        print("[astro_utils] astropy not available — twilight lines disabled")
        return SunEvents(date_str=date_str)

    loc = EarthLocation.from_geodetic(
        lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=alt_m * u.m)

    # Search window: local noon on date_str to local noon next day
    # = UTC noon + utc_offset correction (inverted)
    t0_str = f"{date_str}T{12 - utc_offset_h:05.2f}:00:00"  # approx local noon in UTC
    try:
        t0 = Time(date_str + "T12:00:00", format="isot", scale="utc") \
             - utc_offset_h * u.hour
    except Exception:
        return SunEvents(date_str=date_str)

    # 1-minute grid over 36 hours
    n_steps  = 36 * 60
    dt_hours = np.linspace(0, 36, n_steps)
    times    = t0 + dt_hours * u.hour

    # Compute Sun altitude at each grid point
    frame   = AltAz(obstime=times, location=loc)
    sun_alt = get_sun(times).transform_to(frame).alt.deg  # shape (n_steps,)

    # Convert grid times to local decimal hours
    local_hours = (dt_hours - utc_offset_h) % 24.0 + np.floor(
        (dt_hours - utc_offset_h) / 24.0) * 24.0
    # Simpler: local hour = UTC_hour + utc_offset, but we want continuous
    utc_hours_from_t0 = dt_hours                         # hours since t0
    # t0 is local noon (hour 12). So local_h = 12 + dt_hours
    local_h = 12.0 + dt_hours   # local hours, may exceed 24

    def _crossing(threshold: float, evening: bool) -> float:
        """Find the first crossing of `threshold` in the given half."""
        # Evening: look in first ~18 h (noon → midnight)
        # Morning: look in remaining hours (midnight → next noon)
        if evening:
            mask = local_h <= 30.0   # first half-ish
            idx_range = np.where(mask)[0]
        else:
            mask = local_h >= 18.0   # second half-ish
            idx_range = np.where(mask)[0]

        if len(idx_range) < 2:
            return float("nan")

        alt_seg  = sun_alt[idx_range]
        lh_seg   = local_h[idx_range]

        # Find sign changes
        if evening:
            # Evening: sun going DOWN through threshold
            for i in range(len(alt_seg) - 1):
                if alt_seg[i] >= threshold > alt_seg[i + 1]:
                    # Linear interpolation
                    f = (threshold - alt_seg[i]) / (alt_seg[i+1] - alt_seg[i])
                    return float(lh_seg[i] + f * (lh_seg[i+1] - lh_seg[i]))
        else:
            # Morning: sun going UP through threshold
            for i in range(len(alt_seg) - 1):
                if alt_seg[i] <= threshold < alt_seg[i + 1]:
                    f = (threshold - alt_seg[i]) / (alt_seg[i+1] - alt_seg[i])
                    return float(lh_seg[i] + f * (lh_seg[i+1] - lh_seg[i]))
        return float("nan")

    t_ss  = _crossing(_SUN_ALTS["sunset_sunrise"], evening=True)
    t_sr  = _crossing(_SUN_ALTS["sunset_sunrise"], evening=False)
    t_ce  = _crossing(_SUN_ALTS["civil"],          evening=True)
    t_cm  = _crossing(_SUN_ALTS["civil"],          evening=False)
    t_ne  = _crossing(_SUN_ALTS["nautical"],       evening=True)
    t_nm  = _crossing(_SUN_ALTS["nautical"],       evening=False)
    t_ae  = _crossing(_SUN_ALTS["astronomical"],   evening=True)
    t_am  = _crossing(_SUN_ALTS["astronomical"],   evening=False)

    evts = SunEvents(
        sunset=t_ss,   sunrise=t_sr,
        civil_eve=t_ce, civil_mor=t_cm,
        nautical_eve=t_ne, nautical_mor=t_nm,
        astro_eve=t_ae, astro_mor=t_am,
        date_str=date_str,
    )

    print(f"[astro] Sun events for {date_str}  "
          f"(lat={lat_deg:.2f}  lon={lon_deg:.2f}  UTC{utc_offset_h:+.0f}h):")
    names = [
        ("Sunset",         t_ss),
        ("Civil eve",      t_ce),
        ("Nautical eve",   t_ne),
        ("Astron. eve",    t_ae),
        ("Astron. mor",    t_am),
        ("Nautical mor",   t_nm),
        ("Civil mor",      t_cm),
        ("Sunrise",        t_sr),
    ]
    for name, val in names:
        if np.isfinite(val):
            h = int(val % 24); m = int((val % 1) * 60)
            print(f"  {name:15s} {h:02d}:{m:02d}  ({val:.4f} h)")
        else:
            print(f"  {name:15s} —")

    return evts


def add_twilight_lines(
    axes: list,
    sun_events: SunEvents,
    *,
    show_sunset_sunrise:    bool = True,
    show_civil:             bool = True,
    show_nautical:          bool = True,
    show_astronomical:      bool = True,
    legend_ax_index:        int  = 0,
) -> None:
    """Draw vertical reference lines for sun events on a list of axes.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        All panels to annotate (they share the same x-axis).
    sun_events : SunEvents
        Output of :func:`compute_sun_events`.
    show_* : bool
        Individually toggle each category of lines.
    legend_ax_index : int
        Index of the axis where the legend labels should be added (top panel).
    """
    import matplotlib.pyplot as plt

    events_to_draw = []
    if show_sunset_sunrise:
        events_to_draw.append(("sunset_sunrise",
                                sun_events.sunset, sun_events.sunrise))
    if show_civil:
        events_to_draw.append(("civil",
                                sun_events.civil_eve, sun_events.civil_mor))
    if show_nautical:
        events_to_draw.append(("nautical",
                                sun_events.nautical_eve, sun_events.nautical_mor))
    if show_astronomical:
        events_to_draw.append(("astronomical",
                                sun_events.astro_eve, sun_events.astro_mor))

    for i, ax in enumerate(axes):
        add_legend = (i == legend_ax_index)
        for key, eve_h, mor_h in events_to_draw:
            style   = _LINE_STYLES[key]
            lbl_e, lbl_m = _LINE_LABELS[key]
            for h, lbl in [(eve_h, lbl_e), (mor_h, lbl_m)]:
                if np.isfinite(h):
                    kw = dict(style)
                    if add_legend:
                        kw["label"] = lbl
                    ax.axvline(x=h, **kw)


def night_xlim(sun_events: SunEvents, margin_h: float = 0.25) -> tuple:
    """Return (xmin, xmax) spanning sunset to sunrise with a margin.

    Parameters
    ----------
    sun_events : SunEvents
    margin_h : float
        Extra margin in hours to add on each side (default 15 min).

    Returns
    -------
    (xmin, xmax) in local decimal hours, or (None, None) if events unavailable.
    """
    ss = sun_events.sunset
    sr = sun_events.sunrise
    if not (np.isfinite(ss) and np.isfinite(sr)):
        return None, None
    return ss - margin_h, sr + margin_h


def date_from_jd(jd: float, utc_offset_h: float) -> str:
    """Return the ISO date string of the evening of the night containing `jd`.

    If the local time is before noon, the previous calendar day is returned
    (since that is the "evening" of the night that started the previous day).
    """
    try:
        from astropy.time import Time
        import astropy.units as u
        t = Time(jd, format="jd", scale="utc") + utc_offset_h * u.hour
        iso = t.iso  # "YYYY-MM-DD HH:MM:SS.sss"
        date, time_part = iso.split()
        hour = int(time_part.split(":")[0])
        if hour < 12:
            # Before local noon → previous calendar day
            from datetime import date as dt_date, timedelta
            y, mo, d = map(int, date.split("-"))
            prev = dt_date(y, mo, d) - timedelta(days=1)
            return prev.isoformat()
        return date
    except Exception:
        return ""
