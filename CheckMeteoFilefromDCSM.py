# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:48:59 2026

@author: WangX3
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import xarray as xr


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

nc_path = Path(
    r"D:\ERA5_2005-2021_dfm_mslp_u10n_v10n_chnk_rhoairalt.nc"
)

report_path = Path("meteo_nc_inspection.txt")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def summarize_array(da: xr.DataArray) -> Dict[str, Any]:
    """Return compact statistics without loading more data than needed."""
    summary = {
        "dims": da.dims,
        "shape": da.shape,
        "dtype": str(da.dtype),
        "attrs": dict(da.attrs),
    }

    try:
        sample = da

        # Reduce very large arrays before calculating statistics.
        for dim in da.dims:
            if da.sizes[dim] > 200:
                step = max(1, da.sizes[dim] // 200)
                sample = sample.isel({dim: slice(None, None, step)})

        values = np.asarray(sample.values, dtype=float)
        finite = values[np.isfinite(values)]

        if finite.size:
            summary.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "nan_fraction": float(
                        1.0 - finite.size / values.size
                    ),
                }
            )
    except Exception as exc:
        summary["statistics_error"] = str(exc)

    return summary


def find_variable(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    """Find the first matching variable name, case-insensitively."""
    lower_map = {name.lower(): name for name in ds.variables}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# ------------------------------------------------------------
# OPEN DATASET
# ------------------------------------------------------------

with xr.open_dataset(nc_path, decode_times=True) as ds:

    lines = []

    lines.append("=" * 80)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    lines.append(str(ds))
    lines.append("")

    lines.append("=" * 80)
    lines.append("GLOBAL ATTRIBUTES")
    lines.append("=" * 80)

    for key, value in ds.attrs.items():
        lines.append(f"{key}: {value}")

    lines.append("")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append("DIMENSIONS")
    lines.append("=" * 80)

    for dim, size in ds.sizes.items():
        lines.append(f"{dim}: {size}")

    lines.append("")

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append("COORDINATES")
    lines.append("=" * 80)

    for name in ds.coords:
        da = ds[name]
        lines.append(f"\n{name}")
        lines.append(f"  dims: {da.dims}")
        lines.append(f"  shape: {da.shape}")
        lines.append(f"  dtype: {da.dtype}")

        for key, value in da.attrs.items():
            lines.append(f"  {key}: {value}")

        try:
            values = da.values

            if np.issubdtype(values.dtype, np.datetime64):
                time_index = pd.DatetimeIndex(values)
                lines.append(f"  first: {time_index[0]}")
                lines.append(f"  last:  {time_index[-1]}")

                if len(time_index) > 1:
                    dt = np.diff(time_index.values).astype(
                        "timedelta64[s]"
                    ).astype(float)

                    lines.append(
                        f"  median timestep: "
                        f"{np.median(dt) / 3600:.3f} hours"
                    )
            elif np.issubdtype(values.dtype, np.number):
                finite = values[np.isfinite(values)]

                if finite.size:
                    lines.append(f"  min: {finite.min()}")
                    lines.append(f"  max: {finite.max()}")

        except Exception as exc:
            lines.append(f"  inspection error: {exc}")

    lines.append("")

    # --------------------------------------------------------
    # Data variables
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append("DATA VARIABLES")
    lines.append("=" * 80)

    for name, da in ds.data_vars.items():
        info = summarize_array(da)

        lines.append(f"\n{name}")
        lines.append(f"  dims: {info['dims']}")
        lines.append(f"  shape: {info['shape']}")
        lines.append(f"  dtype: {info['dtype']}")

        for key, value in info["attrs"].items():
            lines.append(f"  {key}: {value}")

        for key in ["min", "max", "mean", "nan_fraction"]:
            if key in info:
                lines.append(f"  {key}: {info[key]}")

        if "statistics_error" in info:
            lines.append(
                f"  statistics_error: {info['statistics_error']}"
            )

    lines.append("")

    # --------------------------------------------------------
    # Identify likely meteorological variables
    # --------------------------------------------------------

    candidates = {
    "air_pressure": [
        "air_pressure_fixed_height",
        "msl",
        "mslp",
        "air_pressure",
        "airpressure",
        "atmospheric_pressure",
    ],
    "eastward_wind": [
        "eastward_wind",
        "u10n",
        "u10",
        "windx",
        "eastward_wind_velocity",
    ],
    "northward_wind": [
        "northward_wind",
        "v10n",
        "v10",
        "windy",
        "northward_wind_velocity",
    ],
    "charnock": [
        "C",
        "chnk",
        "charnock",
        "charnock_parameter",
    ],
    "air_density": [
        "rhoairalt",
        "rhoair",
        "air_density",
    ],
    }

    identified = {
        role: find_variable(ds, names)
        for role, names in candidates.items()
    }

    lines.append("=" * 80)
    lines.append("LIKELY USABLE VARIABLES")
    lines.append("=" * 80)

    for role, variable in identified.items():
        lines.append(f"{role}: {variable}")

    lines.append("")

    # --------------------------------------------------------
    # Check time coverage
    # --------------------------------------------------------

    time_name = find_variable(
        ds,
        ["time", "valid_time", "datetime"],
    )

    if time_name:
        times = pd.DatetimeIndex(
            pd.to_datetime(ds[time_name].values)
        )

        lines.append("=" * 80)
        lines.append("TIME COVERAGE")
        lines.append("=" * 80)
        lines.append(f"time variable: {time_name}")
        lines.append(f"first time: {times[0]}")
        lines.append(f"last time:  {times[-1]}")

        target_start = pd.Timestamp("2013-01-01 00:00:00")
        target_end = pd.Timestamp("2017-12-31 23:59:59")

        covers_target = (
            times[0] <= target_start
            and times[-1] >= target_end
        )

        lines.append(
            f"covers 2013-2017: {covers_target}"
        )

        if len(times) > 1:
            dt_seconds = (
                np.diff(times.values)
                .astype("timedelta64[s]")
                .astype(float)
            )

            lines.append(
                f"minimum timestep: "
                f"{dt_seconds.min() / 3600:.3f} hours"
            )
            lines.append(
                f"median timestep: "
                f"{np.median(dt_seconds) / 3600:.3f} hours"
            )
            lines.append(
                f"maximum timestep: "
                f"{dt_seconds.max() / 3600:.3f} hours"
            )

            irregular = not np.allclose(
                dt_seconds,
                np.median(dt_seconds),
            )

            lines.append(f"irregular timestep: {irregular}")

        lines.append("")

    # --------------------------------------------------------
    # Check spatial coverage
    # --------------------------------------------------------

    lon_name = find_variable(
        ds,
        ["longitude", "lon", "x"],
    )
    lat_name = find_variable(
        ds,
        ["latitude", "lat", "y"],
    )

    lines.append("=" * 80)
    lines.append("SPATIAL COVERAGE")
    lines.append("=" * 80)
    lines.append(f"longitude/x variable: {lon_name}")
    lines.append(f"latitude/y variable:  {lat_name}")

    if lon_name and lat_name:
        lon = np.asarray(ds[lon_name].values, dtype=float)
        lat = np.asarray(ds[lat_name].values, dtype=float)

        lines.append(
            f"longitude/x range: "
            f"{np.nanmin(lon)} to {np.nanmax(lon)}"
        )
        lines.append(
            f"latitude/y range: "
            f"{np.nanmin(lat)} to {np.nanmax(lat)}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Basic suitability assessment
    # --------------------------------------------------------

    lines.append("=" * 80)
    lines.append("PRELIMINARY SUITABILITY")
    lines.append("=" * 80)

    u_name = identified["eastward_wind"]
    v_name = identified["northward_wind"]
    p_name = identified["air_pressure"]

    if u_name and v_name:
        lines.append(
            "Wind: suitable candidate for both D-Flow FM and SWAN, "
            "subject to checking units, height and spatial coverage."
        )
    else:
        lines.append(
            "Wind: required eastward/northward components were not "
            "identified automatically."
        )

    if p_name:
        lines.append(
            "Pressure: suitable candidate for D-Flow FM."
        )
    else:
        lines.append(
            "Pressure: no likely atmospheric-pressure variable found."
        )

    if identified["charnock"]:
        lines.append(
            "Charnock parameter: potentially usable by D-Flow FM "
            "if the selected wind-drag formulation expects it; "
            "not normally passed to SWAN."
        )

    if identified["air_density"]:
        lines.append(
            "Air density: potentially usable by D-Flow FM; "
            "not normally passed to SWAN."
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nReport written to: {report_path.resolve()}")