#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare time-averaged DCSM and nested-model transect profiles.

Stage 1 reads the four D-Flow FM history files and writes a compact CSV cache.
Stage 2 reads that CSV and creates one multi-panel figure per transect.  Set
RUN_ANALYSIS=False after the first successful run to revise plots quickly.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# USER SETTINGS
# =============================================================================

HIST_DIRECTORY = Path(r"C:\dev\Python\DCSMmodelToNestedModel\hist_files")

FILES = {
    "DCSM": "DCSM-FM_0_5nm_0000_his.nc",
    "Nested BC 10 min": "FlowFM_Apr2013_BC10min_his.nc",
    "Nested BC 30 min": "FlowFM_Apr2013_BC30min_his.nc",
    "Nested BC 60 min": "FlowFM_Apr2013_BC60min_his.nc",
}

TRANSECTS = {
    "Alongshore 8": ["Alongshore-8-{:02d}".format(i) for i in range(1, 8)],
    "Alongshore 20": ["Alongshore-20-{:02d}".format(i) for i in range(1, 6)],
    "Cross-shore": ["Crossshore{:02d}".format(i) for i in range(1, 6)],
}

TIME_SERIES_STATIONS = ["PETTZD", "IJMDBTHVN"]

# The original DCSM cross-shore transect contained seven points. DCSM points
# 01 and 03 were removed when constructing the nested-model transect, after
# which the five retained points were renumbered consecutively. Keys below are
# the common/nested names used in plots; values are their source names in DCSM.
DCSM_STATION_ALIASES = {
    "Crossshore01": "Crossshore02",
    "Crossshore02": "Crossshore04",
    "Crossshore03": "Crossshore05",
    "Crossshore04": "Crossshore06",
    "Crossshore05": "Crossshore07",
}

# END_TIME is exclusive: the analysis includes 15–30 April and stops at May 1.
START_TIME = "2013-04-15 00:00:00"
END_TIME = "2013-05-01 00:00:00"

RUN_ANALYSIS = False
RUN_PLOTTING = True

CACHE_CSV = HIST_DIRECTORY / "transect_profile_statistics_20130415_20130501.csv"
TIME_SERIES_CSV = HIST_DIRECTORY / "station_timeseries_20130415_20130501.csv"
FIGURE_DIRECTORY = HIST_DIRECTORY / "transect_profile_figures"

# Coordinates from this run define the common distance axis. All model series
# are matched by station name, so DCSM and nested coordinate systems may differ.
DISTANCE_REFERENCE_RUN = "Nested BC 10 min"

COLORS = {
    "DCSM": "0.35",
    "Nested BC 10 min": "black",
    "Nested BC 30 min": "#2878B5",
    "Nested BC 60 min": "#E07B39",
}
LINESTYLES = {
    "DCSM": "--",
    "Nested BC 10 min": "-",
    "Nested BC 30 min": "-",
    "Nested BC 60 min": "-",
}


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def decode_station_names(values):
    """Decode fixed-width NetCDF names for byte, Unicode, or ASCII integers."""
    array = np.asarray(values)
    names = []
    for row in array:
        chars = []
        for value in np.atleast_1d(row):
            if isinstance(value, (bytes, np.bytes_)):
                chars.append(bytes(value).decode("utf-8", errors="replace"))
            elif isinstance(value, (int, np.integer)):
                code = int(value)
                chars.append(chr(code) if code else "")
            else:
                chars.append(str(value))
        names.append("".join(chars).replace("\x00", "").strip())
    return names


def find_variable(ds, preferred, attribute_terms=()):
    """Find a variable by preferred names, then by long/standard-name terms."""
    lower_lookup = {name.lower(): name for name in ds.variables}
    for candidate in preferred:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    for name, variable in ds.variables.items():
        text = "{} {}".format(
            variable.attrs.get("long_name", ""),
            variable.attrs.get("standard_name", ""),
        ).lower()
        if any(term.lower() in text for term in attribute_terms):
            return name
    return None


def station_dimension(ds):
    if "station" in ds.dims:
        return "station"
    name_var = ds["station_name"]
    return name_var.dims[0]


def select_analysis_period(ds):
    start = np.datetime64(START_TIME)
    end = np.datetime64(END_TIME)
    selected = ds.where((ds.time >= start) & (ds.time < end), drop=True)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(
            "No data between {} and {}. File covers {} to {}.".format(
                START_TIME,
                END_TIME,
                pd.to_datetime(ds.time.values[0]),
                pd.to_datetime(ds.time.values[-1]),
            )
        )
    return selected


def vertical_dimension(data_array, station_dim):
    ignored = {"time", station_dim}
    candidates = [dim for dim in data_array.dims if dim not in ignored]
    # Remove singleton auxiliary dimensions where possible.
    non_singleton = [dim for dim in candidates if data_array.sizes[dim] > 1]
    if len(non_singleton) == 1:
        return non_singleton[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def mean_over_existing(data_array, dimensions):
    dims = [dim for dim in dimensions if dim in data_array.dims]
    return data_array.mean(dim=dims, skipna=True) if dims else data_array


def surface_bottom(data_array, ds, station_dim):
    """Return surface and bottom series using z coordinates when available."""
    vdim = vertical_dimension(data_array, station_dim)
    if vdim is None:
        warnings.warn(
            "{} has no vertical dimension; surface and bottom are identical.".format(
                data_array.name
            )
        )
        return data_array, data_array

    z_name = find_variable(
        ds,
        ["zcoordinate_c", "mesh2d_flowelem_zw", "mesh2d_layer_z"],
        ["layer centre", "layer center", "vertical coordinate"],
    )

    bottom_index = 0
    surface_index = data_array.sizes[vdim] - 1
    if z_name is not None and vdim in ds[z_name].dims:
        z_profile = mean_over_existing(ds[z_name], ["time", station_dim])
        z_values = np.asarray(z_profile.values, dtype=float).squeeze()
        if (
            z_values.ndim == 1
            and len(z_values) == data_array.sizes[vdim]
            and np.any(np.isfinite(z_values))
        ):
            # Ignore any individually missing layers, but only use the
            # coordinate-based method when at least one valid elevation exists.
            bottom_index = int(np.nanargmin(z_values))
            surface_index = int(np.nanargmax(z_values))
        else:
            warnings.warn(
                "The vertical coordinates for {} are entirely missing or "
                "incompatible. Falling back to D-Flow FM history ordering: "
                "first layer = bottom, last layer = surface.".format(
                    data_array.name
                )
            )

    return (
        data_array.isel({vdim: surface_index}),
        data_array.isel({vdim: bottom_index}),
    )


def current_magnitude(ds, station_dim):
    magnitude_name = find_variable(
        ds,
        ["velocity_magnitude", "mesh2d_ucmag", "current_speed"],
        ["velocity magnitude", "current speed"],
    )
    if magnitude_name is not None:
        return ds[magnitude_name]

    x_name = find_variable(ds, ["x_velocity", "mesh2d_ucx", "ucx"])
    y_name = find_variable(ds, ["y_velocity", "mesh2d_ucy", "ucy"])
    if x_name is None or y_name is None:
        raise KeyError("Could not find velocity magnitude or x/y velocity components.")
    return np.hypot(ds[x_name], ds[y_name]).rename("derived_velocity_magnitude")


def coordinate_names(ds):
    x_name = find_variable(
        ds,
        ["station_x_coordinate", "station_x", "x_coordinate"],
        ["station x-coordinate", "station longitude"],
    )
    y_name = find_variable(
        ds,
        ["station_y_coordinate", "station_y", "y_coordinate"],
        ["station y-coordinate", "station latitude"],
    )
    return x_name, y_name


def point_statistics(ds, point_index, station_dim):
    point = ds.isel({station_dim: point_index})

    wl_name = find_variable(
        point,
        ["waterlevel", "mesh2d_s1", "water_level"],
        ["water level", "sea_surface_height"],
    )
    sal_name = find_variable(point, ["salinity", "mesh2d_sa1"], ["salinity"])
    rho_name = find_variable(
        point,
        ["potential_density", "density", "mesh2d_rho"],
        ["potential density", "water density"],
    )
    if wl_name is None or sal_name is None or rho_name is None:
        raise KeyError(
            "Required variable missing. Found waterlevel={}, salinity={}, density={}".format(
                wl_name, sal_name, rho_name
            )
        )

    speed = current_magnitude(point, station_dim)
    speed_vdim = vertical_dimension(speed, station_dim)
    if speed_vdim is not None:
        speed = speed.mean(speed_vdim, skipna=True)

    sal_surface, sal_bottom = surface_bottom(point[sal_name], point, station_dim)
    rho_surface, rho_bottom = surface_bottom(point[rho_name], point, station_dim)

    def time_mean(array):
        result = mean_over_existing(array, ["time"])
        return float(np.asarray(result.values).squeeze())

    surface_salinity = time_mean(sal_surface)
    bottom_salinity = time_mean(sal_bottom)
    surface_density = time_mean(rho_surface)
    bottom_density = time_mean(rho_bottom)

    return {
        "mean_water_level_m": time_mean(point[wl_name]),
        "mean_current_magnitude_m_s": time_mean(speed),
        "mean_surface_salinity": surface_salinity,
        "mean_bottom_salinity": bottom_salinity,
        "mean_salinity_difference_bottom_minus_surface": (
            bottom_salinity - surface_salinity
        ),
        "mean_surface_density_kg_m3": surface_density,
        "mean_bottom_density_kg_m3": bottom_density,
        "mean_density_difference_bottom_minus_surface_kg_m3": (
            bottom_density - surface_density
        ),
    }


def station_time_series(ds, point_index, station_dim):
    """Extract instantaneous station series used in the comparison plots."""
    point = ds.isel({station_dim: point_index})

    wl_name = find_variable(
        point,
        ["waterlevel", "mesh2d_s1", "water_level"],
        ["water level", "sea_surface_height"],
    )
    sal_name = find_variable(point, ["salinity", "mesh2d_sa1"], ["salinity"])
    rho_name = find_variable(
        point,
        ["potential_density", "density", "mesh2d_rho"],
        ["potential density", "water density"],
    )
    temperature_name = find_variable(
        point,
        ["temperature", "water_temperature", "mesh2d_tem1"],
        ["water temperature", "temperature"],
    )
    missing = {
        "water level": wl_name,
        "salinity": sal_name,
        "density": rho_name,
        "temperature": temperature_name,
    }
    missing = [label for label, name in missing.items() if name is None]
    if missing:
        raise KeyError("Missing station time-series variables: {}".format(missing))

    speed = current_magnitude(point, station_dim)
    speed_vdim = vertical_dimension(speed, station_dim)
    if speed_vdim is not None:
        speed = speed.mean(speed_vdim, skipna=True)

    temperature = point[temperature_name]
    temperature_vdim = vertical_dimension(temperature, station_dim)
    if temperature_vdim is not None:
        temperature = temperature.mean(temperature_vdim, skipna=True)

    sal_surface, sal_bottom = surface_bottom(point[sal_name], point, station_dim)
    rho_surface, rho_bottom = surface_bottom(point[rho_name], point, station_dim)

    times = pd.to_datetime(point["time"].values)

    def values_1d(array, label):
        values = np.asarray(array.values, dtype=float).squeeze()
        if values.ndim != 1 or len(values) != len(times):
            raise ValueError(
                "{} did not reduce to one value per time record; shape={}".format(
                    label, values.shape
                )
            )
        return values

    surface_density = values_1d(rho_surface, "surface density")
    bottom_density = values_1d(rho_bottom, "bottom density")

    return pd.DataFrame(
        {
            "time": times,
            "water_level_m": values_1d(point[wl_name], "water level"),
            "depth_averaged_current_magnitude_m_s": values_1d(
                speed, "current magnitude"
            ),
            "depth_averaged_temperature_degC": values_1d(
                temperature, "temperature"
            ),
            "surface_salinity": values_1d(sal_surface, "surface salinity"),
            "bottom_salinity": values_1d(sal_bottom, "bottom salinity"),
            "density_difference_bottom_minus_surface_kg_m3": (
                bottom_density - surface_density
            ),
        }
    )


def cumulative_distance_km(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(np.abs(x) <= 180.0) and np.all(np.abs(y) <= 90.0):
        lon1, lon2 = np.deg2rad(x[:-1]), np.deg2rad(x[1:])
        lat1, lat2 = np.deg2rad(y[:-1]), np.deg2rad(y[1:])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
            dlon / 2.0
        ) ** 2
        segments = 6371.0088 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    else:
        segments = np.hypot(np.diff(x), np.diff(y)) / 1000.0
    return np.concatenate(([0.0], np.cumsum(segments)))


# =============================================================================
# ANALYSIS STAGE
# =============================================================================

def run_analysis():
    rows = []
    time_series_frames = []
    reference_coordinates = {}

    for run_name, filename in FILES.items():
        path = HIST_DIRECTORY / filename
        if not path.is_file():
            raise FileNotFoundError("Missing history file: {}".format(path))

        print("Reading {}: {}".format(run_name, path.name))
        with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as source:
            if "station_name" not in source:
                raise KeyError("station_name is missing from {}".format(path.name))

            ds = select_analysis_period(source)
            station_dim = station_dimension(ds)
            station_names = decode_station_names(ds["station_name"].values)
            name_to_index = {name: i for i, name in enumerate(station_names)}

            requested = [name for points in TRANSECTS.values() for name in points]
            source_names = [
                DCSM_STATION_ALIASES.get(name, name)
                if run_name == "DCSM"
                else name
                for name in requested
            ]
            missing = [name for name in source_names if name not in name_to_index]
            if missing:
                raise KeyError(
                    "Stations missing from {}:\n{}\n\nAvailable:\n{}".format(
                        path.name, missing, station_names
                    )
                )

            missing_time_series = [
                name for name in TIME_SERIES_STATIONS if name not in name_to_index
            ]
            if missing_time_series:
                raise KeyError(
                    "Time-series stations missing from {}: {}".format(
                        path.name, missing_time_series
                    )
                )

            x_name, y_name = coordinate_names(ds)
            for transect_name, point_names in TRANSECTS.items():
                for order, point_name in enumerate(point_names, start=1):
                    source_point_name = (
                        DCSM_STATION_ALIASES.get(point_name, point_name)
                        if run_name == "DCSM"
                        else point_name
                    )
                    index = name_to_index[source_point_name]
                    statistics = point_statistics(ds, index, station_dim)
                    row = {
                        "run": run_name,
                        "transect": transect_name,
                        "point_order": order,
                        "point_name": point_name,
                        "source_point_name": source_point_name,
                    }
                    row.update(statistics)
                    rows.append(row)

                    if run_name == DISTANCE_REFERENCE_RUN:
                        if x_name is not None and y_name is not None:
                            x = float(ds[x_name].isel({station_dim: index}).values)
                            y = float(ds[y_name].isel({station_dim: index}).values)
                            reference_coordinates[point_name] = (x, y)

            for station_name in TIME_SERIES_STATIONS:
                station_frame = station_time_series(
                    ds, name_to_index[station_name], station_dim
                )
                station_frame.insert(0, "station", station_name)
                station_frame.insert(0, "run", run_name)
                time_series_frames.append(station_frame)

    table = pd.DataFrame(rows)

    for transect_name, point_names in TRANSECTS.items():
        if all(name in reference_coordinates for name in point_names):
            xy = [reference_coordinates[name] for name in point_names]
            distances = cumulative_distance_km(
                [item[0] for item in xy], [item[1] for item in xy]
            )
        else:
            warnings.warn(
                "Coordinates unavailable for {}; using point order.".format(transect_name)
            )
            distances = np.arange(len(point_names), dtype=float)

        distance_map = dict(zip(point_names, distances))
        mask = table["transect"] == transect_name
        table.loc[mask, "distance_km"] = table.loc[mask, "point_name"].map(
            distance_map
        )

    table.to_csv(CACHE_CSV, index=False, float_format="%.8f")
    print("Saved analysis cache: {}".format(CACHE_CSV))
    time_series_table = pd.concat(time_series_frames, ignore_index=True)
    time_series_table.to_csv(
        TIME_SERIES_CSV,
        index=False,
        float_format="%.8f",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print("Saved time-series cache: {}".format(TIME_SERIES_CSV))
    return table, time_series_table


# =============================================================================
# PLOTTING STAGE
# =============================================================================

PLOT_FIELDS = [
    ("mean_water_level_m", r"$\langle\eta\rangle_t$ (m)"),
    (
        "mean_current_magnitude_m_s",
        r"$\langle\overline{|\mathbf{u}|}\rangle_t$ (m s$^{-1}$)",
    ),
    ("mean_surface_salinity", r"$\langle S_\mathrm{surface}\rangle_t$"),
    ("mean_bottom_salinity", r"$\langle S_\mathrm{bottom}\rangle_t$"),
    (
        "mean_density_difference_bottom_minus_surface_kg_m3",
        r"$\langle\Delta\rho\rangle_t$ (kg m$^{-3}$)",
    ),
]


def run_plotting(table=None):
    if table is None:
        if not CACHE_CSV.is_file():
            raise FileNotFoundError(
                "Analysis cache not found. Set RUN_ANALYSIS=True first: {}".format(
                    CACHE_CSV
                )
            )
        table = pd.read_csv(CACHE_CSV)

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for transect_name, point_names in TRANSECTS.items():
        subset = table[table["transect"] == transect_name].copy()
        figure, axes = plt.subplots(
            len(PLOT_FIELDS),
            1,
            figsize=(11.5, 14.5),
            sharex=True,
            constrained_layout=True,
        )

        for run_name in FILES:
            run_data = subset[subset["run"] == run_name].sort_values("point_order")
            if len(run_data) != len(point_names):
                warnings.warn(
                    "{} has {} rather than {} points for {}".format(
                        run_name, len(run_data), len(point_names), transect_name
                    )
                )
            for axis, (field, ylabel) in zip(axes, PLOT_FIELDS):
                axis.plot(
                    run_data["distance_km"].to_numpy(),
                    run_data[field].to_numpy(),
                    marker="o",
                    markersize=4.5,
                    linewidth=1.6,
                    color=COLORS.get(run_name),
                    linestyle=LINESTYLES.get(run_name, "-"),
                    label=run_name,
                )
                axis.set_ylabel(ylabel, fontsize=12)
                axis.grid(True, alpha=0.28)

        axes[0].legend(ncol=2, fontsize=10, loc="best")
        axes[-1].set_xlabel("Distance along transect (km)")

        reference = subset[subset["run"] == DISTANCE_REFERENCE_RUN].sort_values(
            "point_order"
        )
        axes[-1].set_xticks(reference["distance_km"].to_numpy())
        axes[-1].set_xticklabels(
            reference["point_name"].str.replace("Alongshore-", "A-").str.replace(
                "Crossshore", "C-"
            ),
            rotation=35,
            ha="right",
        )

        figure.suptitle(
            "{}: time-averaged profiles\n{} to {}".format(
                transect_name, START_TIME, END_TIME
            ),
            fontsize=14,
        )
        figure.show()


TIME_SERIES_PLOT_FIELDS = [
    ("water_level_m", r"$\eta$ (m)"),
    (
        "depth_averaged_current_magnitude_m_s",
        r"$\overline{|\mathbf{u}|}$ (m s$^{-1}$)",
    ),
    (
        "depth_averaged_temperature_degC",
        r"$\overline{T}$ ($^\circ$C)",
    ),
    ("surface_salinity", r"$S_\mathrm{surface}$"),
    ("bottom_salinity", r"$S_\mathrm{bottom}$"),
    (
        "density_difference_bottom_minus_surface_kg_m3",
        r"$\Delta\rho$ (kg m$^{-3}$)",
    ),
]


def run_time_series_plotting(table=None):
    """Plot the requested station series from the cached analysis results."""
    if table is None:
        if not TIME_SERIES_CSV.is_file():
            raise FileNotFoundError(
                "Time-series cache not found. Set RUN_ANALYSIS=True first: {}".format(
                    TIME_SERIES_CSV
                )
            )
        table = pd.read_csv(TIME_SERIES_CSV, parse_dates=["time"])
    else:
        table = table.copy()
        table["time"] = pd.to_datetime(table["time"])

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for station_name in TIME_SERIES_STATIONS:
        station_data = table[table["station"] == station_name]
        figure, axes = plt.subplots(
            len(TIME_SERIES_PLOT_FIELDS),
            1,
            figsize=(14, 16),
            sharex=True,
            constrained_layout=True,
        )

        for run_name in FILES:
            run_data = station_data[station_data["run"] == run_name].sort_values(
                "time"
            )
            # Plain Python datetimes avoid older Matplotlib/newer pandas
            # multidimensional-indexing incompatibilities.
            plot_times = np.asarray(
                pd.DatetimeIndex(run_data["time"]).to_pydatetime(), dtype=object
            )
            for axis, (field, ylabel) in zip(axes, TIME_SERIES_PLOT_FIELDS):
                axis.plot(
                    plot_times,
                    run_data[field].to_numpy(),
                    color=COLORS.get(run_name),
                    linestyle=LINESTYLES.get(run_name, "-"),
                    linewidth=1.0,
                    label=run_name,
                )
                axis.set_ylabel(ylabel,fontsize=12)
                axis.grid(True, alpha=0.28)

        axes[0].legend(ncol=2, fontsize=10, loc="best")
        axes[-1].set_xlabel("Time")
        figure.suptitle(
            "{}: DCSM and nested-model time series\n{} to {}".format(
                station_name, START_TIME, END_TIME
            ),
            fontsize=14,
        )
        figure.show()


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

def main():
    profile_table = None
    time_series_table = None
    if RUN_ANALYSIS:
        profile_table, time_series_table = run_analysis()
    if RUN_PLOTTING:
        # run_plotting(profile_table)
        run_time_series_plotting(time_series_table)


if __name__ == "__main__":
    main()

