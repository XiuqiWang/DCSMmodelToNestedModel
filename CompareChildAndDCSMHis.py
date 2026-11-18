# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:11:13 2026

@author: WangX3
"""
"""Compare one nested D-Flow FM history file with a DCSM history file.

The analysis stage reads both NetCDF files and creates compact CSV caches.
The plotting stage reads those caches and makes:

* time-averaged profiles along two alongshore and one cross-shore transect;
* time series at PETTZD and IJGEUL.

After the first successful analysis, set RUN_ANALYSIS=False to change figures
without rereading the large NetCDF files.
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
    "Child": "FlowFM_20130101_20130501_his.nc",
}

TRANSECTS = {
    "Alongshore 8": ["Alongshore-8-{:02d}".format(i) for i in range(1, 8)],
    "Alongshore 20": ["Alongshore-20-{:02d}".format(i) for i in range(1, 6)],
    "Cross-shore": ["Crossshore{:02d}".format(i) for i in range(1, 6)],
}

TIME_SERIES_STATIONS = ["PETTZD", "IJGEUL"]

# Common plotted name -> actual name in the DCSM history file.
# The DCSM cross-shore transect originally had seven points. Points 01 and 03
# were removed from the child model and the remaining five were renumbered.
DCSM_STATION_ALIASES = {
    "Crossshore01": "Crossshore02",
    "Crossshore02": "Crossshore04",
    "Crossshore03": "Crossshore05",
    "Crossshore04": "Crossshore06",
    "Crossshore05": "Crossshore07",
    "PETTZD": "PETTZD",
    "IJGEUL": "IJGEUL",
}

# END_TIME is exclusive.
START_TIME = "2013-02-01 00:00:00"
END_TIME = "2013-05-01 00:00:00"

RUN_ANALYSIS = True
RUN_PROFILE_PLOTTING = True
RUN_TIME_SERIES_PLOTTING = True
SAVE_FIGURES = True
SHOW_FIGURES = True

CACHE_CSV = HIST_DIRECTORY / "child_dcsm_transect_statistics_20130201_20130501.csv"
TIME_SERIES_CSV = HIST_DIRECTORY / "child_dcsm_station_timeseries_20130201_20130501.csv"
FIGURE_DIRECTORY = HIST_DIRECTORY / "child_dcsm_comparison_figures"

# Child coordinates define the common horizontal distance axis.
DISTANCE_REFERENCE_RUN = "Child"

COLORS = {"DCSM": "0.35", "Child": "#D55E00"}
LINESTYLES = {"DCSM": "--", "Child": "-"}

# Vertical-position convention used in panels 3 and 4 of every figure.
# Model identity is represented only by colour in these panels.
SURFACE_LINESTYLE = "-"
BOTTOM_LINESTYLE = ":"
SURFACE_MARKER = "o"
BOTTOM_MARKER = "s"

# Applied only to the third transect figure (Cross-shore).
CROSS_SHORE_M2_PHASE_LIMITS_DEG = (160.0, 170.0)

# Principal lunar semidiurnal constituent. Harmonic phase is reported for the
# cosine convention eta = mean + A*cos(omega*t - phi), relative to this epoch.
M2_PERIOD_HOURS = 12.4206012
M2_PHASE_EPOCH = np.datetime64("2013-01-01T00:00:00")


# =============================================================================
# NETCDF HELPERS
# =============================================================================

def decode_station_names(values):
    """Decode fixed-width station names stored as bytes, strings or integers."""
    array = np.asarray(values)
    if array.ndim == 1 and array.dtype.kind in {"U", "S", "O"}:
        return [
            (value.decode("utf-8", errors="replace") if isinstance(value, bytes)
             else str(value)).replace("\x00", "").strip()
            for value in array
        ]

    names = []
    for row in array:
        characters = []
        for value in np.atleast_1d(row):
            if isinstance(value, (bytes, np.bytes_)):
                characters.append(bytes(value).decode("utf-8", errors="replace"))
            elif isinstance(value, (int, np.integer)):
                characters.append(chr(int(value)) if int(value) else "")
            else:
                characters.append(str(value))
        names.append("".join(characters).replace("\x00", "").strip())
    return names


def find_variable(ds, preferred, attribute_terms=()):
    """Find a variable by name, then by standard_name/long_name text."""
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in preferred:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for name, variable in ds.variables.items():
        description = "{} {}".format(
            variable.attrs.get("long_name", ""),
            variable.attrs.get("standard_name", ""),
        ).lower()
        if any(term.lower() in description for term in attribute_terms):
            return name
    return None


def station_dimension(ds):
    if "station" in ds.dims:
        return "station"
    return ds["station_name"].dims[0]


def select_analysis_period(ds):
    start = np.datetime64(START_TIME)
    end = np.datetime64(END_TIME)
    selected = ds.where((ds.time >= start) & (ds.time < end), drop=True)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(
            "No records between {} and {}. File covers {} to {}.".format(
                START_TIME,
                END_TIME,
                pd.to_datetime(ds.time.values[0]),
                pd.to_datetime(ds.time.values[-1]),
            )
        )
    return selected


def source_station_name(run_name, common_name):
    if run_name == "DCSM":
        return DCSM_STATION_ALIASES.get(common_name, common_name)
    return common_name


def station_index(station_names, requested_name, filename):
    """Resolve a station exactly, with a case-insensitive fallback."""
    exact = {name: index for index, name in enumerate(station_names)}
    if requested_name in exact:
        return exact[requested_name]

    folded = {name.casefold(): index for index, name in enumerate(station_names)}
    if requested_name.casefold() in folded:
        return folded[requested_name.casefold()]

    raise KeyError(
        "Station {!r} is missing from {}.\nAvailable stations:\n{}".format(
            requested_name, filename, "\n".join(station_names)
        )
    )


def vertical_dimension(data_array, station_dim):
    candidates = [dim for dim in data_array.dims if dim not in {"time", station_dim}]
    non_singleton = [dim for dim in candidates if data_array.sizes[dim] > 1]
    if len(non_singleton) == 1:
        return non_singleton[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def mean_over_existing(data_array, dimensions):
    dimensions = [dim for dim in dimensions if dim in data_array.dims]
    return data_array.mean(dim=dimensions, skipna=True) if dimensions else data_array


def surface_bottom(data_array, ds, station_dim):
    """Return surface and bottom series, using z coordinates when possible."""
    vertical_dim = vertical_dimension(data_array, station_dim)
    if vertical_dim is None:
        warnings.warn(
            "{} has no vertical dimension; using the same values for surface and bottom."
            .format(data_array.name)
        )
        return data_array, data_array

    bottom_index = 0
    surface_index = data_array.sizes[vertical_dim] - 1
    z_name = find_variable(
        ds,
        ["zcoordinate_c", "mesh2d_flowelem_zw", "mesh2d_layer_z"],
        ["layer centre", "layer center", "vertical coordinate"],
    )

    if z_name is not None and vertical_dim in ds[z_name].dims:
        z_profile = mean_over_existing(ds[z_name], ["time", station_dim])
        z_values = np.asarray(z_profile.values, dtype=float).squeeze()
        if (
            z_values.ndim == 1
            and z_values.size == data_array.sizes[vertical_dim]
            and np.any(np.isfinite(z_values))
        ):
            bottom_index = int(np.nanargmin(z_values))
            surface_index = int(np.nanargmax(z_values))
        else:
            warnings.warn(
                "Invalid z coordinates for {}; using first layer as bottom and last "
                "layer as surface.".format(data_array.name)
            )

    return (
        data_array.isel({vertical_dim: surface_index}),
        data_array.isel({vertical_dim: bottom_index}),
    )


def velocity_components(ds):
    """Return horizontal velocity components needed for tidal-ellipse fitting."""
    x_name = find_variable(ds, ["x_velocity", "mesh2d_ucx", "ucx"])
    y_name = find_variable(ds, ["y_velocity", "mesh2d_ucy", "ucy"])
    if x_name is None or y_name is None:
        raise KeyError("Could not find x/y velocity components.")
    return ds[x_name], ds[y_name]


def current_magnitude(ds):
    magnitude_name = find_variable(
        ds,
        ["velocity_magnitude", "mesh2d_ucmag", "current_speed"],
        ["velocity magnitude", "current speed"],
    )
    if magnitude_name is not None:
        return ds[magnitude_name]
    velocity_x, velocity_y = velocity_components(ds)
    return np.hypot(velocity_x, velocity_y).rename("derived_current_magnitude")


def harmonic_fit(data_array, period_hours=M2_PERIOD_HOURS):
    """Fit mean, trend, cosine and sine terms at one tidal frequency.

    The fitted convention is

        value = mean + trend*t + A*cos(omega*t - phi)

    where phase is measured in degrees relative to M2_PHASE_EPOCH.
    """
    values = np.asarray(data_array.values, dtype=float).squeeze()
    times = np.asarray(data_array["time"].values).astype("datetime64[ns]")
    if values.ndim != 1 or values.size != times.size:
        raise ValueError(
            "{} must contain one value per time for harmonic fitting; shape={}".format(
                data_array.name, values.shape
            )
        )

    seconds = (times - M2_PHASE_EPOCH).astype("timedelta64[ns]").astype(float) / 1e9
    valid = np.isfinite(values) & np.isfinite(seconds)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Too few valid records for M2 harmonic fitting.")

    seconds = seconds[valid]
    values = values[valid]
    omega = 2.0 * np.pi / (period_hours * 3600.0)
    trend_time = (seconds - seconds.mean()) / 86400.0
    design = np.column_stack(
        (
            np.ones(seconds.size),
            trend_time,
            np.cos(omega * seconds),
            np.sin(omega * seconds),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None) 
    #least-square fitting U=U0+trend*t+Acos(omega*t)+Bsin(omega*t)
    cosine_coefficient = float(coefficients[2])
    sine_coefficient = float(coefficients[3])
    amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
    phase = float(
        np.degrees(np.arctan2(sine_coefficient, cosine_coefficient)) % 360.0
    )
    return amplitude, phase, cosine_coefficient, sine_coefficient


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


def required_variables(point):
    names = {
        "water_level": find_variable(
            point,
            ["waterlevel", "mesh2d_s1", "water_level"],
            ["water level", "sea_surface_height"],
        ),
        "salinity": find_variable(point, ["salinity", "mesh2d_sa1"], ["salinity"]),
        "density": find_variable(
            point,
            ["potential_density", "density", "mesh2d_rho"],
            ["potential density", "water density"],
        ),
        "temperature": find_variable(
            point,
            ["temperature", "water_temperature", "mesh2d_tem1"],
            ["water temperature", "temperature"],
        ),
    }
    missing = [label for label, name in names.items() if name is None]
    if missing:
        raise KeyError("Missing variables: {}".format(", ".join(missing)))
    return names


def point_statistics(ds, point_index, station_dim):
    point = ds.isel({station_dim: point_index})
    names = required_variables(point)

    speed = current_magnitude(point)
    speed_vertical_dim = vertical_dimension(speed, station_dim)
    if speed_vertical_dim is not None:
        speed = speed.mean(speed_vertical_dim, skipna=True)

    velocity_x, velocity_y = velocity_components(point)
    velocity_x_vertical_dim = vertical_dimension(velocity_x, station_dim)
    velocity_y_vertical_dim = vertical_dimension(velocity_y, station_dim)
    if velocity_x_vertical_dim is not None:
        velocity_x = velocity_x.mean(velocity_x_vertical_dim, skipna=True)
    if velocity_y_vertical_dim is not None:
        velocity_y = velocity_y.mean(velocity_y_vertical_dim, skipna=True)

    sal_surface, sal_bottom = surface_bottom(
        point[names["salinity"]], point, station_dim
    )
    temperature_surface, temperature_bottom = surface_bottom(
        point[names["temperature"]], point, station_dim
    )
    rho_surface, rho_bottom = surface_bottom(
        point[names["density"]], point, station_dim
    )

    def time_mean(data_array):
        result = mean_over_existing(data_array, ["time"])
        return float(np.asarray(result.values).squeeze())

    water_level = point[names["water_level"]]
    m2_water_level_amplitude, m2_water_level_phase, _, _ = harmonic_fit(water_level)

    # Fit the M2 constituent independently to the depth-averaged x/y velocity
    # components. The largest singular value of the cosine/sine coefficient
    # matrix is the semi-major amplitude of the M2 current ellipse: the peak
    # M2 tidal velocity irrespective of ellipse orientation.
    _, _, ux_cosine, ux_sine = harmonic_fit(velocity_x)
    _, _, uy_cosine, uy_sine = harmonic_fit(velocity_y)
    tidal_ellipse_matrix = np.array(
        [[ux_cosine, ux_sine], [uy_cosine, uy_sine]], dtype=float
    )
    m2_peak_velocity = float(
        np.linalg.svd(tidal_ellipse_matrix, compute_uv=False)[0]
    )# semi-major-axis amplitude of the matrix

    surface_salinity = time_mean(sal_surface)
    bottom_salinity = time_mean(sal_bottom)
    surface_temperature = time_mean(temperature_surface)
    bottom_temperature = time_mean(temperature_bottom)
    surface_density = time_mean(rho_surface)
    bottom_density = time_mean(rho_bottom)

    return {
        "mean_water_level_m": time_mean(water_level),
        "m2_water_level_amplitude_m": m2_water_level_amplitude,
        "m2_water_level_phase_deg": m2_water_level_phase,
        "mean_current_magnitude_m_s": time_mean(speed),
        "m2_peak_tidal_velocity_m_s": m2_peak_velocity,
        "mean_surface_salinity": surface_salinity,
        "mean_bottom_salinity": bottom_salinity,
        "mean_salinity_difference_bottom_minus_surface": (
            bottom_salinity - surface_salinity
        ),
        "mean_surface_temperature_degC": surface_temperature,
        "mean_bottom_temperature_degC": bottom_temperature,
        "mean_temperature_difference_surface_minus_bottom_degC": (
            surface_temperature - bottom_temperature
        ),
        "mean_surface_density_kg_m3": surface_density,
        "mean_bottom_density_kg_m3": bottom_density,
        "mean_density_difference_bottom_minus_surface_kg_m3": (
            bottom_density - surface_density
        ),
    }


def station_time_series(ds, point_index, station_dim):
    point = ds.isel({station_dim: point_index})
    names = required_variables(point)

    speed = current_magnitude(point)
    speed_vertical_dim = vertical_dimension(speed, station_dim)
    if speed_vertical_dim is not None:
        speed = speed.mean(speed_vertical_dim, skipna=True)

    sal_surface, sal_bottom = surface_bottom(
        point[names["salinity"]], point, station_dim
    )
    temperature_surface, temperature_bottom = surface_bottom(
        point[names["temperature"]], point, station_dim
    )
    rho_surface, rho_bottom = surface_bottom(
        point[names["density"]], point, station_dim
    )
    times = pd.to_datetime(point.time.values)

    def values_1d(data_array, label):
        values = np.asarray(data_array.values, dtype=float).squeeze()
        if values.ndim != 1 or values.size != len(times):
            raise ValueError(
                "{} did not reduce to one value per time; shape={}".format(
                    label, values.shape
                )
            )
        return values

    surface_density = values_1d(rho_surface, "surface density")
    bottom_density = values_1d(rho_bottom, "bottom density")

    return pd.DataFrame({
        "time": times,
        "water_level_m": values_1d(point[names["water_level"]], "water level"),
        "depth_averaged_current_magnitude_m_s": values_1d(speed, "current magnitude"),
        "surface_salinity": values_1d(sal_surface, "surface salinity"),
        "bottom_salinity": values_1d(sal_bottom, "bottom salinity"),
        "surface_temperature_degC": values_1d(
            temperature_surface, "surface temperature"
        ),
        "bottom_temperature_degC": values_1d(
            temperature_bottom, "bottom temperature"
        ),
        "density_difference_bottom_minus_surface_kg_m3": (
            bottom_density - surface_density
        ),
    })


def cumulative_distance_km(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(np.abs(x) <= 180.0) and np.all(np.abs(y) <= 90.0):
        lon1, lon2 = np.deg2rad(x[:-1]), np.deg2rad(x[1:])
        lat1, lat2 = np.deg2rad(y[:-1]), np.deg2rad(y[1:])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        segments = 6371.0088 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    else:
        segments = np.hypot(np.diff(x), np.diff(y)) / 1000.0
    return np.concatenate(([0.0], np.cumsum(segments)))


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis():
    profile_rows = []
    time_series_frames = []
    reference_coordinates = {}

    requested_transect_points = [
        point for points in TRANSECTS.values() for point in points
    ]

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
            station_names = decode_station_names(ds.station_name.values)
            x_name, y_name = coordinate_names(ds)

            # Resolve all names before doing expensive calculations.
            resolved = {}
            for common_name in requested_transect_points + TIME_SERIES_STATIONS:
                actual_name = source_station_name(run_name, common_name)
                resolved[common_name] = (
                    actual_name,
                    station_index(station_names, actual_name, path.name),
                )

            for transect_name, point_names in TRANSECTS.items():
                for order, common_name in enumerate(point_names, start=1):
                    actual_name, index = resolved[common_name]
                    row = {
                        "run": run_name,
                        "transect": transect_name,
                        "point_order": order,
                        "point_name": common_name,
                        "source_point_name": actual_name,
                    }
                    row.update(point_statistics(ds, index, station_dim))
                    profile_rows.append(row)

                    if (
                        run_name == DISTANCE_REFERENCE_RUN
                        and x_name is not None
                        and y_name is not None
                    ):
                        reference_coordinates[common_name] = (
                            float(ds[x_name].isel({station_dim: index}).values),
                            float(ds[y_name].isel({station_dim: index}).values),
                        )

            for common_name in TIME_SERIES_STATIONS:
                actual_name, index = resolved[common_name]
                frame = station_time_series(ds, index, station_dim)
                frame.insert(0, "source_station_name", actual_name)
                frame.insert(0, "station", common_name)
                frame.insert(0, "run", run_name)
                time_series_frames.append(frame)

    profile_table = pd.DataFrame(profile_rows)

    for transect_name, point_names in TRANSECTS.items():
        if all(point in reference_coordinates for point in point_names):
            coordinates = [reference_coordinates[point] for point in point_names]
            distances = cumulative_distance_km(
                [coordinate[0] for coordinate in coordinates],
                [coordinate[1] for coordinate in coordinates],
            )
        else:
            warnings.warn(
                "Child coordinates unavailable for {}; using point order."
                .format(transect_name)
            )
            distances = np.arange(len(point_names), dtype=float)

        distance_map = dict(zip(point_names, distances))
        mask = profile_table.transect == transect_name
        profile_table.loc[mask, "distance_km"] = (
            profile_table.loc[mask, "point_name"].map(distance_map)
        )

    profile_table.to_csv(CACHE_CSV, index=False, float_format="%.8f")
    time_series_table = pd.concat(time_series_frames, ignore_index=True)
    time_series_table.to_csv(
        TIME_SERIES_CSV,
        index=False,
        float_format="%.8f",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print("Saved: {}".format(CACHE_CSV))
    print("Saved: {}".format(TIME_SERIES_CSV))
    return profile_table, time_series_table


# =============================================================================
# PLOTTING
# =============================================================================

STATION_HYDRO_FIELDS = [
    ("water_level_m", r"$\eta$ (m)"),
    (
        "depth_averaged_current_magnitude_m_s",
        r"$\overline{|\mathbf{u}|}$ (m s$^{-1}$)",
    ),
]


def finish_figure(figure, filename):
    if SAVE_FIGURES:
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIRECTORY / filename
        figure.savefig(path, dpi=220, bbox_inches="tight")
        print("Saved: {}".format(path))
    if SHOW_FIGURES:
        figure.show()
    else:
        plt.close(figure)


def run_profile_plotting(table=None):
    if table is None:
        if not CACHE_CSV.is_file():
            raise FileNotFoundError(
                "Missing cache. Set RUN_ANALYSIS=True first: {}".format(CACHE_CSV)
            )
        table = pd.read_csv(CACHE_CSV)

    for transect_name, point_names in TRANSECTS.items():
        subset = table[table.transect == transect_name]
        figure, axes = plt.subplots(
            5,
            1,
            figsize=(11.5, 14.5),
            sharex=True,
            constrained_layout=True,
        )

        # Panel 1: M2 water-level amplitude and phase. Phase is unwrapped along
        # each transect so that crossings through 0/360 degrees remain smooth.
        amplitude_handles = []
        amplitude_labels = []
        phase_axis = axes[0].twinx()
        dcsm_unwrapped_phase = None
        for run_name in FILES:
            run_data = subset[subset.run == run_name].sort_values("point_order")
            amplitude_line = axes[0].plot(
                run_data.distance_km.to_numpy(),
                run_data.m2_water_level_amplitude_m.to_numpy(),
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                color=COLORS[run_name],
                linestyle=LINESTYLES[run_name],
                label="{} $A_{{M_2}}$".format(run_name),
            )[0]
            amplitude_handles.append(amplitude_line)
            amplitude_labels.append("{} $A_{{M_2}}$".format(run_name))

            phase = np.degrees(
                np.unwrap(np.radians(run_data.m2_water_level_phase_deg.to_numpy()))
            )
            if run_name == "DCSM":
                dcsm_unwrapped_phase = phase.copy()
            elif dcsm_unwrapped_phase is not None:
                phase += 360.0 * np.round(
                    (np.nanmean(dcsm_unwrapped_phase) - np.nanmean(phase)) / 360.0
                )
            phase_axis.plot(
                run_data.distance_km.to_numpy(),
                phase,
                marker="^",
                markersize=4.2,
                linewidth=1.3,
                color=COLORS[run_name],
                linestyle=":",
                label=(r"{} $\phi_{{M_2}}$").format(run_name),
            )

        axes[0].set_ylabel(r"$A_{M_2}$ (m)", fontsize=12)
        phase_axis.set_ylabel(
            r"$\phi_{M_2}$ (degrees; relative to 2013-01-01)", fontsize=12
        )
        if transect_name == "Cross-shore":
            phase_axis.set_ylim(*CROSS_SHORE_M2_PHASE_LIMITS_DEG)
        axes[0].grid(True, alpha=0.28)
        phase_handles, phase_labels = phase_axis.get_legend_handles_labels()
        axes[0].legend(
            amplitude_handles + phase_handles,
            amplitude_labels + phase_labels,
            fontsize=12,
            ncol=2,
            loc="best",
        )

        # Panel 2: peak M2 tidal velocity (semi-major current-ellipse amplitude).
        for run_name in FILES:
            run_data = subset[subset.run == run_name].sort_values("point_order")
            axes[1].plot(
                run_data.distance_km.to_numpy(),
                run_data.m2_peak_tidal_velocity_m_s.to_numpy(),
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                color=COLORS[run_name],
                linestyle=LINESTYLES[run_name],
                label=run_name,
            )

        axes[1].set_ylabel(r"Peak $M_2$ tidal velocity (m s$^{-1}$)", fontsize=12)
        axes[1].grid(True, alpha=0.28)
        axes[1].legend(fontsize=12, loc="best")

        # Panel 3: absolute surface and bottom salinity. Colour identifies the
        # model; line style and marker identify the vertical position.
        for run_name in FILES:
            run_data = subset[subset.run == run_name].sort_values("point_order")
            axes[2].plot(
                run_data.distance_km.to_numpy(),
                run_data.mean_surface_salinity.to_numpy(),
                color=COLORS[run_name],
                linestyle=SURFACE_LINESTYLE,
                marker=SURFACE_MARKER,
                markersize=4.2,
                linewidth=1.5,
                label="{} surface".format(run_name),
            )
            axes[2].plot(
                run_data.distance_km.to_numpy(),
                run_data.mean_bottom_salinity.to_numpy(),
                color=COLORS[run_name],
                linestyle=BOTTOM_LINESTYLE,
                marker=BOTTOM_MARKER,
                markersize=4.0,
                linewidth=1.5,
                label="{} bottom".format(run_name),
            )
        axes[2].set_ylabel(r"$\langle S\rangle_t$", fontsize=12)
        axes[2].grid(True, alpha=0.28)
        axes[2].legend(fontsize=12, ncol=2, loc="best")

        # Panel 4: absolute surface and bottom temperature.
        for run_name in FILES:
            run_data = subset[subset.run == run_name].sort_values("point_order")
            axes[3].plot(
                run_data.distance_km.to_numpy(),
                run_data.mean_surface_temperature_degC.to_numpy(),
                color=COLORS[run_name],
                linestyle=SURFACE_LINESTYLE,
                marker=SURFACE_MARKER,
                markersize=4.2,
                linewidth=1.5,
                label="{} surface".format(run_name),
            )
            axes[3].plot(
                run_data.distance_km.to_numpy(),
                run_data.mean_bottom_temperature_degC.to_numpy(),
                color=COLORS[run_name],
                linestyle=BOTTOM_LINESTYLE,
                marker=BOTTOM_MARKER,
                markersize=4.0,
                linewidth=1.5,
                label="{} bottom".format(run_name),
            )
        axes[3].set_ylabel(r"$\langle T\rangle_t$ ($^\circ$C)", fontsize=12)
        axes[3].grid(True, alpha=0.28)
        axes[3].legend(fontsize=12, ncol=2, loc="best")

        # Panel 5: combined density stratification.
        for run_name in FILES:
            run_data = subset[subset.run == run_name].sort_values("point_order")
            axes[4].plot(
                run_data.distance_km.to_numpy(),
                run_data.mean_density_difference_bottom_minus_surface_kg_m3.to_numpy(),
                color=COLORS[run_name],
                linestyle=LINESTYLES[run_name],
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                label=run_name,
            )
        axes[4].set_ylabel(
            r"$\langle\rho_b-\rho_s\rangle_t$ (kg m$^{-3}$)", fontsize=12
        )
        axes[4].grid(True, alpha=0.28)
        axes[4].legend(fontsize=12, loc="best")

        axes[-1].set_xlabel("Distance along transect (km)")
        reference = subset[subset.run == DISTANCE_REFERENCE_RUN].sort_values(
            "point_order"
        )
        axes[-1].set_xticks(reference.distance_km.to_numpy())
        axes[-1].set_xticklabels(
            reference.point_name.str.replace("Alongshore-", "A-", regex=False)
            .str.replace("Crossshore", "C-", regex=False),
            rotation=35,
            ha="right",
        )
        figure.suptitle(
            "{}: $M_2$ tide and time-averaged tracers\n{} to {}".format(
                transect_name, START_TIME, END_TIME
            ),
            fontsize=14,
        )
        safe_name = transect_name.lower().replace(" ", "_").replace("-", "_")
        finish_figure(figure, "profile_{}.png".format(safe_name))


def run_time_series_plotting(table=None):
    if table is None:
        if not TIME_SERIES_CSV.is_file():
            raise FileNotFoundError(
                "Missing cache. Set RUN_ANALYSIS=True first: {}".format(
                    TIME_SERIES_CSV
                )
            )
        table = pd.read_csv(TIME_SERIES_CSV, parse_dates=["time"])
    else:
        table = table.copy()
        table["time"] = pd.to_datetime(table.time)

    for station_name in TIME_SERIES_STATIONS:
        station_data = table[table.station == station_name]
        figure, axes = plt.subplots(
            5,
            1,
            figsize=(14, 15),
            sharex=True,
            constrained_layout=True,
        )

        for run_name in FILES:
            run_data = station_data[station_data.run == run_name].sort_values("time")
            plot_times = np.asarray(
                pd.DatetimeIndex(run_data.time).to_pydatetime(), dtype=object
            )
            # Panels 1-2: instantaneous hydrodynamics.
            for axis, (field, ylabel) in zip(axes[:2], STATION_HYDRO_FIELDS):
                axis.plot(
                    plot_times,
                    run_data[field].to_numpy(),
                    color=COLORS[run_name],
                    linestyle=LINESTYLES[run_name],
                    linewidth=1.0,
                    label=run_name,
                )
                axis.set_ylabel(ylabel, fontsize=11)
                axis.grid(True, alpha=0.28)

            # Panel 3: surface and bottom salinity.
            axes[2].plot(
                plot_times,
                run_data.surface_salinity.to_numpy(),
                color=COLORS[run_name],
                linestyle=SURFACE_LINESTYLE,
                linewidth=1.0,
                label="{} surface".format(run_name),
            )
            axes[2].plot(
                plot_times,
                run_data.bottom_salinity.to_numpy(),
                color=COLORS[run_name],
                linestyle=BOTTOM_LINESTYLE,
                linewidth=1.0,
                label="{} bottom".format(run_name),
            )

            # Panel 4: surface and bottom temperature.
            axes[3].plot(
                plot_times,
                run_data.surface_temperature_degC.to_numpy(),
                color=COLORS[run_name],
                linestyle=SURFACE_LINESTYLE,
                linewidth=1.0,
                label="{} surface".format(run_name),
            )
            axes[3].plot(
                plot_times,
                run_data.bottom_temperature_degC.to_numpy(),
                color=COLORS[run_name],
                linestyle=BOTTOM_LINESTYLE,
                linewidth=1.0,
                label="{} bottom".format(run_name),
            )

            # Panel 5: bottom-minus-surface density difference.
            axes[4].plot(
                plot_times,
                run_data.density_difference_bottom_minus_surface_kg_m3.to_numpy(),
                color=COLORS[run_name],
                linestyle=LINESTYLES[run_name],
                linewidth=1.0,
                label=run_name,
            )

        axes[0].legend(fontsize=12, loc="best")
        axes[1].legend(fontsize=12, loc="best")
        axes[2].set_ylabel(r"$S$", fontsize=14)
        axes[2].grid(True, alpha=0.28)
        axes[2].legend(fontsize=12, ncol=2, loc="best")
        axes[3].set_ylabel(r"$T$ ($^\circ$C)", fontsize=12)
        axes[3].grid(True, alpha=0.28)
        axes[3].legend(fontsize=12, ncol=2, loc="best")
        axes[4].set_ylabel(r"$\rho_b-\rho_s$ (kg m$^{-3}$)", fontsize=12)
        axes[4].grid(True, alpha=0.28)
        axes[4].legend(fontsize=12, loc="best")
        axes[-1].set_xlabel("Time")
        figure.suptitle(
            "{}: DCSM and child-model time series\n{} to {}"
            .format(station_name, START_TIME, END_TIME),
            fontsize=14,
        )
        finish_figure(figure, "timeseries_{}.png".format(station_name.lower()))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    profile_table = None
    time_series_table = None

    if RUN_ANALYSIS:
        profile_table, time_series_table = run_analysis()
    if RUN_PROFILE_PLOTTING:
        run_profile_plotting(profile_table)
    if RUN_TIME_SERIES_PLOTTING:
        run_time_series_plotting(time_series_table)

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()

