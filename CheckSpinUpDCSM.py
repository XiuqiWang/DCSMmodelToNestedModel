# -*- coding: utf-8 -*-
"""Compare partitioned DCSM-FM map water levels with Waterinfo gauges."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

map_directory = Path(r"C:\dev\DCSM-FM\Snellius-run\Spin-up")
station_dir = Path(r"C:\dev\Python\DCSMmodelRuns/StationData")
file_pattern = "DCSM-FM_0_5nm_????_map.nc"

# Put the downloaded Waterinfo CSV in map_directory, or specify its full path.
measurement_csv = station_dir/"WL_4stations_spinup.csv"

# The CSV contains no explicit time-zone field. Start with zero. If comparison
# shows a consistent one-hour phase offset, verify the Waterinfo export time
# convention before changing this to +1 or -1.
observation_time_shift_hours = 0

# Coordinates are longitude, latitude in WGS 84 / ETRS89.
points = {
    "Offshore_test": (4.043000, 52.291000),
    "Petten_Zuid": (4.650000, 52.766700),
    "IJmuiden_Buitenhaven": (4.554826, 52.462328),
    "Scheveningen": (4.263565, 52.099038),
}

# Model point name -> Waterinfo MEETPUNT_IDENTIFICATIE.
# Offshore_test is intentionally absent because the CSV has no offshore gauge.
station_mapping = {
    "Petten_Zuid": "Petten, zuid",
    "IJmuiden_Buitenhaven": "IJmuiden, buitenhaven",
    "Scheveningen": "Scheveningen",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_face_coordinates(ds):
    """Find the face-centre longitude and latitude variables."""
    possible_pairs = [
        ("mesh2d_face_x", "mesh2d_face_y"),
        ("mesh2d_flowelem_xcc", "mesh2d_flowelem_ycc"),
        ("mesh2d_face_lon", "mesh2d_face_lat"),
    ]
    for x_name, y_name in possible_pairs:
        if x_name in ds.variables and y_name in ds.variables:
            return x_name, y_name
    raise KeyError(
        "Could not identify face-coordinate variables.\n"
        f"Available variables:\n{list(ds.variables)}"
    )


def find_water_level(ds):
    """Find the D-Flow FM water-level variable."""
    for name in ["mesh2d_s1", "mesh2d_waterlevel", "waterlevel"]:
        if name in ds.variables:
            return name

    for name, variable in ds.variables.items():
        long_name = str(variable.attrs.get("long_name", "")).lower()
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        if "water level" in long_name or standard_name in {
            "sea_surface_height",
            "sea_surface_height_above_geoid",
        }:
            return name

    raise KeyError(
        "Could not identify the water-level variable.\n"
        f"Available variables:\n{list(ds.variables)}"
    )


def nearest_distance_squared(x, y, longitude, latitude):
    """Approximate squared distance in degrees, scaled in longitude."""
    longitude_scale = np.cos(np.deg2rad(latitude))
    return ((x - longitude) * longitude_scale) ** 2 + (y - latitude) ** 2


def load_waterinfo_csv(csv_file):
    """Read the supplied semicolon-delimited Waterinfo export."""
    observations = pd.read_csv(csv_file, sep=";", dtype=str)

    required = {
        "MEETPUNT_IDENTIFICATIE",
        "WAARNEMINGDATUM",
        "WAARNEMINGTIJD",
        "NUMERIEKEWAARDE",
        "EENHEID_CODE",
        "HOEDANIGHEID_CODE",
        "KWALITEITSOORDEEL_CODE",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing Waterinfo CSV columns: {sorted(missing)}")

    units = set(observations["EENHEID_CODE"].dropna().unique())
    datums = set(observations["HOEDANIGHEID_CODE"].dropna().unique())
    if units != {"cm"}:
        raise ValueError(f"Expected water levels in cm, found: {sorted(units)}")
    if datums != {"NAP"}:
        raise ValueError(f"Expected vertical datum NAP, found: {sorted(datums)}")

    observations["time"] = pd.to_datetime(
        observations["WAARNEMINGDATUM"]
        + " "
        + observations["WAARNEMINGTIJD"],
        dayfirst=True,
        errors="raise",
    ) + pd.to_timedelta(observation_time_shift_hours, unit="h")

    # Waterinfo values are centimetres relative to NAP; convert to metres.
    observations["water_level_m_NAP"] = (
        pd.to_numeric(observations["NUMERIEKEWAARDE"], errors="coerce")
        / 100.0
    )

    observations = observations.dropna(subset=["time", "water_level_m_NAP"])

    quality_counts = observations["KWALITEITSOORDEEL_CODE"].value_counts()
    print("Waterinfo quality-code counts:")
    print(quality_counts.to_string())
    print(
        "NOTE: values with quality code 25 are retained because the export "
        "does not define that code. Check the Waterinfo code list before "
        "deciding whether to exclude them."
    )

    output = {}
    for model_name, station_name in station_mapping.items():
        station = observations.loc[
            observations["MEETPUNT_IDENTIFICATIE"] == station_name,
            ["time", "water_level_m_NAP"],
        ].copy()

        if station.empty:
            print(f"WARNING: station not found in CSV: {station_name}")
            continue

        # Average duplicate timestamps if any occur.
        output[model_name] = (
            station.groupby("time")["water_level_m_NAP"]
            .mean()
            .sort_index()
            .rename(model_name)
        )

    return output


def observations_at_model_times(observed, model_index):
    """Interpolate 10-minute observations onto the model timestamps."""
    target = pd.DatetimeIndex(model_index)
    combined_index = observed.index.union(target).sort_values()
    interpolated = (
        observed.reindex(combined_index)
        .interpolate(method="time", limit_area="inside")
        .reindex(target)
    )
    interpolated.index = target
    return interpolated


# ---------------------------------------------------------------------------
# Load observations
# ---------------------------------------------------------------------------

if not measurement_csv.is_file():
    raise FileNotFoundError(f"Waterinfo CSV not found:\n{measurement_csv}")

observed_series = load_waterinfo_csv(measurement_csv)


# ---------------------------------------------------------------------------
# Locate map files and nearest model cells
# ---------------------------------------------------------------------------

map_files = sorted(map_directory.glob(file_pattern))
if not map_files:
    raise FileNotFoundError(f"No files match:\n{map_directory / file_pattern}")

print(f"Found {len(map_files)} map files.")
if len(map_files) != 32:
    print("WARNING: expected 32 partition map files.")

results = {
    name: {
        "distance_squared": np.inf,
        "series": None,
        "file": None,
        "face_index": None,
        "cell_lon": None,
        "cell_lat": None,
    }
    for name in points
}

for file_number, map_file in enumerate(map_files, start=1):
    print(f"[{file_number:02d}/{len(map_files):02d}] Reading {map_file.name}")

    with xr.open_dataset(
        map_file, decode_times=True, mask_and_scale=True
    ) as ds:
        x_name, y_name = find_face_coordinates(ds)
        water_level_name = find_water_level(ds)
        x = np.asarray(ds[x_name].values).squeeze()
        y = np.asarray(ds[y_name].values).squeeze()
        water_level = ds[water_level_name]

        if "time" not in water_level.dims:
            raise ValueError(
                f"{water_level_name} has no time dimension in {map_file.name}"
            )

        face_dimensions = [d for d in water_level.dims if d != "time"]
        if len(face_dimensions) != 1:
            raise ValueError(
                f"Unexpected dimensions for {water_level_name}: "
                f"{water_level.dims}"
            )
        face_dimension = face_dimensions[0]

        for point_name, (longitude, latitude) in points.items():
            distance_squared = nearest_distance_squared(
                x, y, longitude, latitude
            )
            distance_squared[~np.isfinite(distance_squared)] = np.inf
            face_index = int(np.argmin(distance_squared))
            candidate_distance = float(distance_squared[face_index])

            if candidate_distance >= results[point_name]["distance_squared"]:
                continue

            series = water_level.isel({face_dimension: face_index}).load()
            values = np.asarray(series.values).squeeze()
            if not np.any(np.isfinite(values)):
                continue

            results[point_name].update(
                {
                    "distance_squared": candidate_distance,
                    "series": series,
                    "file": map_file.name,
                    "face_index": face_index,
                    "cell_lon": float(x[face_index]),
                    "cell_lat": float(y[face_index]),
                }
            )


# ---------------------------------------------------------------------------
# Build model series and calculate matched statistics
# ---------------------------------------------------------------------------

model_series = {}
matched_observations = {}
statistics = []

for point_name, result in results.items():
    if result["series"] is None:
        print(f"WARNING: no valid wet cell found for {point_name}")
        continue

    source = result["series"]
    model = pd.Series(
        np.asarray(source.values).squeeze(),
        index=pd.to_datetime(source["time"].values),
        name=point_name,
    ).sort_index()
    model_series[point_name] = model

    distance_km = np.sqrt(result["distance_squared"]) * 111.0
    print(f"\n{point_name}:")
    print(f"  requested: {points[point_name]}")
    print(
        f"  selected:  ({result['cell_lon']:.6f}, "
        f"{result['cell_lat']:.6f})"
    )
    print(f"  distance:  {distance_km:.3f} km")
    print(f"  partition: {result['file']}")
    print(f"  face:      {result['face_index']}")

    if point_name not in observed_series:
        continue

    matched = observations_at_model_times(
        observed_series[point_name], model.index
    )
    matched_observations[point_name] = matched

    valid = model.notna() & matched.notna()
    if not valid.any():
        print("  WARNING: no overlapping model and observation times")
        continue

    difference = model.loc[valid] - matched.loc[valid]
    statistics.append(
        {
            "station": point_name,
            "n": int(valid.sum()),
            "bias_m": difference.mean(),
            "mae_m": difference.abs().mean(),
            "rmse_m": np.sqrt(np.mean(difference**2)),
            "correlation": model.loc[valid].corr(matched.loc[valid]),
        }
    )

if not model_series:
    raise RuntimeError("No valid model series were extracted.")

# ---------------------------------------------------------------------------
# Plot model and observations
# ---------------------------------------------------------------------------

figure, axes = plt.subplots(
    len(model_series),
    1,
    figsize=(12, 3.0 * len(model_series)),
    sharex=True,
)
if len(model_series) == 1:
    axes = [axes]

for axis, (point_name, model) in zip(axes, model_series.items()):
    if point_name in observed_series:
        observed = observed_series[point_name]
        # Limit the displayed observations to the modelled interval.
        observed = observed.loc[
            (observed.index >= model.index.min())
            & (observed.index <= model.index.max())
        ]
        axis.plot(
            observed.index.to_pydatetime(),
            observed.to_numpy(),
            color="0.55",
            linewidth=0.8,
            label="Waterinfo observation (10 min, m NAP)",
            zorder=1,
        )

    axis.plot(
        model.index.to_pydatetime(),
        model.to_numpy(),
        color="tab:blue",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="DCSM map output",
        zorder=2,
    )

    if point_name in matched_observations:
        matched = matched_observations[point_name]
        axis.scatter(
            matched.index.to_pydatetime(),
            matched.to_numpy(),
            color="tab:orange",
            marker="x",
            s=22,
            label="Observation at DCSM output time",
            zorder=3,
        )

    axis.set_title(point_name)
    axis.set_ylabel("Water level (m)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time")
figure.suptitle(
    "DCSM final spin-up month: model versus Waterinfo",
    fontsize=14,
)
figure.tight_layout()
plt.show()
