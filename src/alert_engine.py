"""
alert_engine.py
---------------
Self-contained analytics engine powering EpiTrack's advanced features:

  • Country breach alerts   — R₀ crossing 1.0 in last N days
  • Severity tier scoring   — CRITICAL / WARNING / WATCH / CLEAR
  • Doubling time           — ln(2) / growth_rate per country
  • Anomaly detection       — Z-score + IQR rolling flag on case series
  • Wave auto-detection     — peak-finding on smoothed curve → Wave labels
  • R₀ forecasting          — linear extrapolation + uncertainty band

All functions are pure (no Streamlit / Plotly imports) so they can be
unit-tested independently.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


# ══════════════════════════════════════════════════════════════════════════════
#  SEVERITY SCORING
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_LEVELS = {
    "CRITICAL": dict(color="#f87171", bg="rgba(248,113,113,0.10)", border="rgba(248,113,113,0.30)", icon="🔴"),
    "WARNING":  dict(color="#fbbf24", bg="rgba(251,191,36,0.10)",  border="rgba(251,191,36,0.30)",  icon="🟡"),
    "WATCH":    dict(color="#38bdf8", bg="rgba(56,189,248,0.08)",  border="rgba(56,189,248,0.22)",  icon="🔵"),
    "CLEAR":    dict(color="#34d399", bg="rgba(52,211,153,0.08)",  border="rgba(52,211,153,0.22)",  icon="🟢"),
}


def severity_tier(r0: float, growth_rate: float, doubling_days: float | None) -> str:
    """
    Classify a country into a severity tier.

    Rules (in priority order):
      CRITICAL  — R₀ > 2.0  OR  growth_rate > 0.25  OR  doubling < 5 days
      WARNING   — R₀ > 1.3  OR  growth_rate > 0.10
      WATCH     — R₀ > 1.0  OR  growth_rate > 0.0
      CLEAR     — everything else
    """
    dd = doubling_days if doubling_days is not None else 9999
    if r0 > 2.0 or growth_rate > 0.25 or dd < 5:
        return "CRITICAL"
    if r0 > 1.3 or growth_rate > 0.10:
        return "WARNING"
    if r0 > 1.0 or growth_rate > 0.0:
        return "WATCH"
    return "CLEAR"


# ══════════════════════════════════════════════════════════════════════════════
#  DOUBLING TIME
# ══════════════════════════════════════════════════════════════════════════════

def doubling_time(growth_rate: float) -> float | None:
    """
    Compute case doubling time in days from instantaneous growth rate.
    Returns None when growth is zero or negative (cases not doubling).
    """
    if growth_rate <= 1e-6:
        return None
    return np.log(2) / growth_rate


# ══════════════════════════════════════════════════════════════════════════════
#  COUNTRY BREACH ALERTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_country_alerts(df_all: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """
    Scan every country for recent R₀ behavior and return a ranked alert table.

    Returns a DataFrame with columns:
      country, r0_latest, r0_prev, r0_trend, growth_rate,
      doubling_days, severity, crossed_threshold, days_since_cross
    """
    records = []
    cutoff = df_all['date'].max() - pd.Timedelta(days=lookback_days)

    for country, grp in df_all.groupby('country'):
        grp = grp.sort_values('date')
        if len(grp) < lookback_days + 7:
            continue

        recent  = grp[grp['date'] >= cutoff]
        earlier = grp[grp['date'] <  cutoff].tail(lookback_days)

        r0_latest = float(recent['r0_estimate'].iloc[-1])
        r0_prev   = float(earlier['r0_estimate'].mean()) if len(earlier) else r0_latest
        r0_trend  = r0_latest - r0_prev                          # + = worsening

        gr_latest = float(recent['growth_rate'].iloc[-1])
        dd        = doubling_time(gr_latest)

        # Did R₀ cross 1.0 from below in the lookback window?
        r0_series = recent['r0_estimate'].values
        crossed   = bool(np.any(np.diff(np.sign(r0_series - 1.0)) > 0))

        # Days since last crossing above 1.0
        above = grp[grp['r0_estimate'] > 1.0]
        if len(above):
            days_since = int((grp['date'].max() - above['date'].min()).days)
        else:
            days_since = None

        tier = severity_tier(r0_latest, gr_latest, dd)

        records.append(dict(
            country          = country,
            r0_latest        = round(r0_latest, 3),
            r0_prev          = round(r0_prev,   3),
            r0_trend         = round(r0_trend,  3),
            growth_rate      = round(gr_latest, 4),
            doubling_days    = round(dd, 1) if dd else None,
            severity         = tier,
            crossed_threshold= crossed,
            days_since_cross = days_since,
        ))

    alert_df = pd.DataFrame(records)

    # Sort: severity priority → r0 descending
    sev_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "CLEAR": 3}
    alert_df['_sev_rank'] = alert_df['severity'].map(sev_order)
    alert_df = alert_df.sort_values(['_sev_rank', 'r0_latest'], ascending=[True, False])
    alert_df = alert_df.drop(columns=['_sev_rank']).reset_index(drop=True)
    return alert_df


# ══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(
    series: pd.Series,
    dates:  pd.Series,
    z_thresh: float = 2.8,
    window:   int   = 21,
) -> pd.DataFrame:
    """
    Flag anomalous spikes in a case series using a hybrid Z-score + IQR method.

    Algorithm:
      1. Compute rolling median + rolling std (window=`window` days).
      2. Z-score = (x - rolling_median) / rolling_std.
      3. Also flag if x > rolling 75th percentile + 1.5 * IQR.
      4. A point is anomalous if BOTH conditions hold (reduces false positives).

    Returns DataFrame: date, value, z_score, is_anomaly
    """
    vals = series.values.astype(float)
    n    = len(vals)
    z_scores   = np.zeros(n)
    is_anomaly = np.zeros(n, dtype=bool)

    for i in range(window, n):
        w = vals[max(0, i-window):i]
        med = np.median(w)
        std = np.std(w) + 1e-9
        z   = (vals[i] - med) / std
        z_scores[i] = z

        q75, q25 = np.percentile(w, [75, 25])
        iqr_upper = q75 + 1.5 * (q75 - q25)

        is_anomaly[i] = (z > z_thresh) and (vals[i] > iqr_upper)

    return pd.DataFrame({
        'date':       dates.values,
        'value':      vals,
        'z_score':    z_scores,
        'is_anomaly': is_anomaly,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  WAVE AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_waves(
    series: pd.Series,
    dates:  pd.Series,
    smooth_window:    int   = 21,
    min_prominence:   float = 0.15,   # fraction of global max
    min_wave_days:    int   = 30,
) -> list[dict]:
    """
    Automatically segment an epidemic curve into named waves.

    Steps:
      1. Smooth series with uniform (box) filter.
      2. Find peaks with minimum prominence = min_prominence × global_max.
      3. Assign trough boundaries between consecutive peaks.
      4. Return list of wave dicts: {wave_num, start, end, peak_date, peak_val}.

    Returns list of wave dicts (may be empty if no clear waves found).
    """
    vals = series.values.astype(float)
    if len(vals) < smooth_window * 2:
        return []

    smoothed   = uniform_filter1d(vals, size=smooth_window)
    global_max = smoothed.max()
    if global_max < 1:
        return []

    prominence_abs = min_prominence * global_max
    peaks, props   = find_peaks(
        smoothed,
        prominence=prominence_abs,
        distance=min_wave_days,
    )

    if len(peaks) == 0:
        return []

    date_arr = pd.to_datetime(dates.values)
    waves    = []

    for i, peak_idx in enumerate(peaks):
        # Wave start: trough before this peak (or series start)
        if i == 0:
            start_idx = 0
        else:
            prev_peak = peaks[i-1]
            trough_segment = smoothed[prev_peak:peak_idx]
            trough_offset  = int(np.argmin(trough_segment))
            start_idx      = prev_peak + trough_offset

        # Wave end: trough after this peak (or series end)
        if i == len(peaks) - 1:
            end_idx = len(vals) - 1
        else:
            next_peak = peaks[i+1]
            trough_segment = smoothed[peak_idx:next_peak]
            trough_offset  = int(np.argmin(trough_segment))
            end_idx        = peak_idx + trough_offset

        waves.append(dict(
            wave_num  = i + 1,
            start     = date_arr[start_idx],
            end       = date_arr[end_idx],
            peak_date = date_arr[peak_idx],
            peak_val  = float(smoothed[peak_idx]),
        ))

    return waves


# ══════════════════════════════════════════════════════════════════════════════
#  R₀ FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def forecast_r0(
    r0_series: pd.Series,
    dates:     pd.Series,
    horizon:   int = 14,
) -> dict:
    """
    Forecast R₀ for the next `horizon` days using:
      - Exponential weighted moving average (captures recent trend)
      - Linear trend extrapolation from last 21 days
      - Uncertainty band derived from residual std

    Returns dict:
      future_dates  : pd.DatetimeIndex
      r0_forecast   : np.ndarray
      r0_lower      : np.ndarray
      r0_upper      : np.ndarray
      trend_dir     : str  ("rising" | "falling" | "stable")
      r0_at_horizon : float
    """
    vals = r0_series.values.astype(float)
    n    = len(vals)

    # EWM smoothed baseline
    ewm_vals = pd.Series(vals).ewm(alpha=0.25).mean().values

    # Linear trend from last 21 days
    lookback  = min(21, n)
    x         = np.arange(lookback)
    y         = ewm_vals[-lookback:]
    coeffs    = np.polyfit(x, y, 1)          # slope, intercept
    slope     = coeffs[0]

    # Residual std for uncertainty
    y_hat = np.polyval(coeffs, x)
    resid_std = np.std(y - y_hat) + 0.05    # floor of 0.05

    # Project forward
    x_future   = np.arange(lookback, lookback + horizon)
    r0_forecast = np.polyval(coeffs, x_future)
    r0_forecast = np.clip(r0_forecast, 0.3, 6.0)

    # Widening uncertainty band
    uncertainty = resid_std * np.sqrt(np.arange(1, horizon + 1)) * 0.35
    r0_lower    = np.clip(r0_forecast - uncertainty, 0.3, None)
    r0_upper    = r0_forecast + uncertainty

    future_dates = pd.date_range(
        pd.to_datetime(dates.values[-1]) + pd.Timedelta(days=1),
        periods=horizon, freq='D',
    )

    # Trend direction
    if slope > 0.003:
        trend_dir = "rising"
    elif slope < -0.003:
        trend_dir = "falling"
    else:
        trend_dir = "stable"

    return dict(
        future_dates  = future_dates,
        r0_forecast   = r0_forecast,
        r0_lower      = r0_lower,
        r0_upper      = r0_upper,
        trend_dir     = trend_dir,
        r0_at_horizon = float(r0_forecast[-1]),
        slope         = float(slope),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY BUILDER  (convenience — used by dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def build_country_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per country with all key alert metrics pre-computed.
    Convenient for rendering the alert panel table.
    """
    alert_df = compute_country_alerts(df_all)
    return alert_df