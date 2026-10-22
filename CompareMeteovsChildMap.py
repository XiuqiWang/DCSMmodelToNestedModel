# -*- coding: utf-8 -*-
"""
Compare projected meteorological forcing against wind written at observation
stations in FlowFM_his.nc.

Python 3.9 compatible.

The comparison timestamps are taken directly from FlowFM_his.nc. No regular
time interval is imposed. The meteorological forcing is interpolated in space
to each observation coordinate and in time to the available history times.

Outputs:
    his_vs_meteo_timeseries.csv
    his_vs_meteo_summary.csv
    timeseries_ObservationPoint01.png
    timeseries_ObservationPoint02.png
    timeseries_ObservationPoint03.png
    his_vs_meteo_scatter.png

Required packages:
    numpy, pandas, xarray, netCDF4, scipy, matplotlib

Install with:
    conda install numpy pandas xarray netcdf4 scipy matplotlib
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =====================================================================
# USER SETTINGS
# =====================================================================

meteo_nc = Path(
    r"C:\dev\DCSM-FM\TestMeteo_flowonly\dflowfm\meteo"
    r"\ERA5_child_test_RDNew_flowgrid.nc"
)

his_nc = Path(
    r"C:\dev\DCSM-FM\TestMeteo_flowonly\dflowfm\output\FlowFM_his.nc"
)

output_directory = Path(
    r"C:\dev\DCSM-FM\TestMeteo_flowonly\validation"
)

# Optional comparison window. Set either value to None to use the complete
# corresponding time coverage from FlowFM_his.nc.
start_time = pd.Timestamp("2013-12-06 00:00:00")
end_time = pd.Timestamp("2013-12-06 03:00:00")

# These coordinates should match the stations written to FlowFM_his.nc.
observation_points = {
    "ObservationPoint01": (41285.0, 478351.0),
    "ObservationPoint02": (56902.0, 503196.0),
    "ObservationPoint03": (71099.0, 534785.0),
}

# Maximum permitted distance between the requested coordinate and the station
# coordinate stored in FlowFM_his.nc.
maximum_station_distance_m = 100.0

meteo_spatial_method = "linear"

# Candidate names in the projected meteorological NetCDF.
METEO_TIME = ["time"]
METEO_X = ["x", "X"]
METEO_Y = ["y", "Y"]
METEO_U = ["x_wind", "windx", "eastward_wind"]
METEO_V = ["y_wind", "windy", "northward_wind"]

# Common D-Flow FM history variable names.
HIS_TIME = ["time"]
HIS_STATION_NAME = [
    "station_name",
    "stations",
    "station_id",
]
HIS_STATION_X = [
    "station_x_coordinate",
    "station_x",
    "x_station",
]
HIS_STATION_Y = [
    "station_y_coordinate",
    "station_y",
    "y_station",
]
HIS_U = [
    "station_windx",
    "windx",
    "x_wind",
    "station_wind_x",
    "eastward_wind",
]
HIS_V = [
    "station_windy",
    "windy",
    "y_wind",
    "station_wind_y",
    "northward_wind",
]


# =====================================================================
# HELPERS
# =====================================================================

def find_name(
    dataset: xr.Dataset,
    candidates: List[str],
    role: str,
) -> str:
    """Find an exact or conservative case-insensitive variable match."""
    for name in candidates:
        if name in dataset.variables or name in dataset.coords:
            return name

    available = list(dataset.variables) + list(dataset.coords)

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for name in available:
            if name.lower() == candidate_lower:
                return name

    raise KeyError(
        "Could not find {0}. Tried: {1}\nAvailable variables:\n{2}"
        .format(
            role,
            ", ".join(candidates),
            "\n".join(sorted(dataset.variables)),
        )
    )


def decode_names(values: np.ndarray) -> List[str]:
    """Decode common NetCDF station-name encodings."""
    array = np.asarray(values)

    # Character array: (station, name_strlen)
    if array.ndim == 2 and array.dtype.kind in ("S", "U"):
        decoded = []
        for row in array:
            if array.dtype.kind == "S":
                text = b"".join(row.tolist()).decode(
                    "utf-8",
                    errors="replace",
                )
            else:
                text = "".join(row.tolist())
            decoded.append(text.strip().strip("\x00"))
        return decoded

    decoded = []
    for value in array.reshape(-1):
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        decoded.append(text.strip().strip("\x00"))

    return decoded


def sort_coordinate(
    data: xr.DataArray,
    coordinate_name: str,
) -> xr.DataArray:
    """Sort a DataArray when a one-dimensional coordinate decreases."""
    coordinate = np.asarray(
        data[coordinate_name].values,
        dtype=float,
    )

    if coordinate.size > 1 and np.any(np.diff(coordinate) < 0):
        return data.sortby(coordinate_name)

    return data


def speed(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    return np.hypot(u, v)


def direction_from(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Meteorological direction FROM which wind blows, clockwise from projected
    grid north.

    0/360 = north, 90 = east, 180 = south, 270 = west.
    """
    direction_to = np.degrees(np.arctan2(u, v)) % 360.0
    result = (direction_to + 180.0) % 360.0

    return np.where(
        speed(u, v) > 1.0e-8,
        result,
        np.nan,
    )


def angle_difference(
    history_direction: np.ndarray,
    forcing_direction: np.ndarray,
) -> np.ndarray:
    """Smallest signed history-minus-forcing angular difference."""
    return (
        (
            history_direction
            - forcing_direction
            + 180.0
        ) % 360.0
    ) - 180.0


def correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    mask = np.isfinite(first) & np.isfinite(second)

    if np.sum(mask) < 2:
        return np.nan

    if (
        np.std(first[mask]) == 0.0
        or np.std(second[mask]) == 0.0
    ):
        return np.nan

    return float(
        np.corrcoef(
            first[mask],
            second[mask],
        )[0, 1]
    )


def station_dimension(
    data: xr.DataArray,
    time_name: str,
) -> str:
    """Identify the station dimension in a history wind variable."""
    candidates = [
        dimension
        for dimension in data.dims
        if dimension != time_name
        and data.sizes[dimension] > 1
    ]

    for dimension in candidates:
        lower = dimension.lower()
        if "station" in lower or "obs" in lower:
            return dimension

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not identify station dimension for {0}; dimensions: {1}"
        .format(data.name, data.dims)
    )


def reduce_time_station(
    data: xr.DataArray,
    time_name: str,
    station_dim: str,
) -> xr.DataArray:
    """Remove singleton extra dimensions and return time-by-station data."""
    result = data

    for dimension in list(result.dims):
        if dimension not in (time_name, station_dim):
            if result.sizes[dimension] == 1:
                result = result.isel(
                    {dimension: 0},
                    drop=True,
                )
            else:
                raise ValueError(
                    "{0} has unsupported non-singleton dimension "
                    "{1} of size {2}."
                    .format(
                        result.name,
                        dimension,
                        result.sizes[dimension],
                    )
                )

    return result.transpose(time_name, station_dim)


def select_history_times(
    all_times: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Select actual FlowFM_his.nc timestamps inside the requested window."""
    mask = np.ones(len(all_times), dtype=bool)

    if start_time is not None:
        mask &= all_times >= start_time

    if end_time is not None:
        mask &= all_times <= end_time

    selected = all_times[mask]

    if len(selected) == 0:
        raise ValueError(
            "No FlowFM_his.nc times occur inside the requested interval. "
            "History coverage is {0} to {1}."
            .format(all_times[0], all_times[-1])
        )

    return selected


def nearest_station(
    station_x: np.ndarray,
    station_y: np.ndarray,
    point_x: float,
    point_y: float,
) -> Tuple[int, float]:
    """Find the nearest valid history station."""
    station_x = np.asarray(station_x, dtype=float).reshape(-1)
    station_y = np.asarray(station_y, dtype=float).reshape(-1)

    valid = np.isfinite(station_x) & np.isfinite(station_y)

    if not np.any(valid):
        raise ValueError(
            "No valid station coordinates exist in FlowFM_his.nc."
        )

    valid_indices = np.where(valid)[0]
    distances = np.hypot(
        station_x[valid] - point_x,
        station_y[valid] - point_y,
    )

    local_index = int(np.argmin(distances))

    return (
        int(valid_indices[local_index]),
        float(distances[local_index]),
    )


def interpolate_meteo(
    dataset: xr.Dataset,
    x_coordinate: float,
    y_coordinate: float,
    target_times: pd.DatetimeIndex,
    time_name: str,
    x_name: str,
    y_name: str,
    u_name: str,
    v_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate projected meteo forcing to one station and history times."""
    u_data = sort_coordinate(
        sort_coordinate(dataset[u_name], x_name),
        y_name,
    )
    v_data = sort_coordinate(
        sort_coordinate(dataset[v_name], x_name),
        y_name,
    )

    u_point = u_data.interp(
        {
            x_name: x_coordinate,
            y_name: y_coordinate,
        },
        method=meteo_spatial_method,
    )
    v_point = v_data.interp(
        {
            x_name: x_coordinate,
            y_name: y_coordinate,
        },
        method=meteo_spatial_method,
    )

    forcing_times = pd.DatetimeIndex(
        pd.to_datetime(dataset[time_name].values)
    )

    if (
        target_times[0] < forcing_times[0]
        or target_times[-1] > forcing_times[-1]
    ):
        raise ValueError(
            "Meteo forcing covers {0} to {1}, but selected history times "
            "cover {2} to {3}."
            .format(
                forcing_times[0],
                forcing_times[-1],
                target_times[0],
                target_times[-1],
            )
        )

    time_target = xr.DataArray(
        target_times.values,
        dims=("comparison_time",),
    )

    u_values = u_point.interp(
        {time_name: time_target}
    ).values
    v_values = v_point.interp(
        {time_name: time_target}
    ).values

    return (
        np.asarray(u_values, dtype=float).reshape(-1),
        np.asarray(v_values, dtype=float).reshape(-1),
    )


def extract_history_at_station(
    data: xr.DataArray,
    station_dim: str,
    station_index: int,
    time_name: str,
    selected_times: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Select a history station and actual history timestamps.

    Uses .sel rather than interpolation because selected_times are taken
    directly from FlowFM_his.nc.
    """
    station_series = data.isel(
        {station_dim: station_index}
    )

    selected = station_series.sel(
        {time_name: selected_times.values}
    )

    return np.asarray(
        selected.values,
        dtype=float,
    ).reshape(-1)


def add_one_to_one_line(
    axis,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    values = np.concatenate(
        [
            np.asarray(x_values, dtype=float).reshape(-1),
            np.asarray(y_values, dtype=float).reshape(-1),
        ]
    )
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return

    lower = float(np.min(finite))
    upper = float(np.max(finite))

    if lower == upper:
        padding = max(1.0, abs(lower) * 0.1)
    else:
        padding = 0.05 * (upper - lower)

    lower -= padding
    upper += padding

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle=":",
        label="1:1",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)


# =====================================================================
# PLOTS
# =====================================================================

def plot_station_timeseries(
    table: pd.DataFrame,
    station: str,
    output_path: Path,
) -> None:
    subset = table[
        table["station"] == station
    ].copy()

    time_values = subset["time"].to_numpy()

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 12),
        sharex=True,
    )

    comparisons = [
        (
            "forcing_x_wind_m_s",
            "history_x_wind_m_s",
            "x wind (m/s)",
        ),
        (
            "forcing_y_wind_m_s",
            "history_y_wind_m_s",
            "y wind (m/s)",
        ),
        (
            "forcing_speed_m_s",
            "history_speed_m_s",
            "speed (m/s)",
        ),
        (
            "forcing_direction_from_deg",
            "history_direction_from_deg",
            "direction from (°)",
        ),
    ]

    for axis, (
        forcing_column,
        history_column,
        ylabel,
    ) in zip(axes, comparisons):
        axis.plot(
            time_values,
            subset[forcing_column].to_numpy(),
            label="Projected meteo forcing",
        )
        axis.plot(
            time_values,
            subset[history_column].to_numpy(),
            linestyle="--",
            label="FlowFM history",
        )
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.legend()

    axes[-1].set_ylim(0.0, 360.0)
    axes[-1].set_yticks([0, 90, 180, 270, 360])
    axes[-1].set_xlabel("Time")

    figure.suptitle(
        "{0}: FlowFM history wind versus projected forcing"
        .format(station)
    )
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_scatter(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 10),
    )

    specifications = [
        (
            axes[0, 0],
            "forcing_x_wind_m_s",
            "history_x_wind_m_s",
            "x wind",
            "m/s",
        ),
        (
            axes[0, 1],
            "forcing_y_wind_m_s",
            "history_y_wind_m_s",
            "y wind",
            "m/s",
        ),
        (
            axes[1, 0],
            "forcing_speed_m_s",
            "history_speed_m_s",
            "Wind speed",
            "m/s",
        ),
        (
            axes[1, 1],
            "forcing_direction_from_deg",
            "history_direction_from_deg",
            "Wind direction from",
            "degrees",
        ),
    ]

    for axis, forcing_column, history_column, title, unit in specifications:
        for station in observation_points:
            subset = table[
                table["station"] == station
            ]

            axis.scatter(
                subset[forcing_column].to_numpy(),
                subset[history_column].to_numpy(),
                label=station,
            )

        add_one_to_one_line(
            axis,
            table[forcing_column].to_numpy(),
            table[history_column].to_numpy(),
        )

        axis.set_xlabel(
            "Projected meteo forcing ({0})".format(unit)
        )
        axis.set_ylabel(
            "FlowFM history ({0})".format(unit)
        )
        axis.set_title(title)
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        "FlowFM history wind versus projected meteorological forcing"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    if not meteo_nc.is_file():
        raise FileNotFoundError(
            "Meteorological file not found: {0}"
            .format(meteo_nc)
        )

    if not his_nc.is_file():
        raise FileNotFoundError(
            "History file not found: {0}"
            .format(his_nc)
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    with xr.open_dataset(
        meteo_nc,
        decode_times=True,
        engine="netcdf4",
    ) as meteo, xr.open_dataset(
        his_nc,
        decode_times=True,
        engine="netcdf4",
    ) as history:

        meteo_time = find_name(
            meteo,
            METEO_TIME,
            "meteo time",
        )
        meteo_x = find_name(
            meteo,
            METEO_X,
            "meteo x coordinate",
        )
        meteo_y = find_name(
            meteo,
            METEO_Y,
            "meteo y coordinate",
        )
        meteo_u = find_name(
            meteo,
            METEO_U,
            "meteo x-wind variable",
        )
        meteo_v = find_name(
            meteo,
            METEO_V,
            "meteo y-wind variable",
        )

        history_time = find_name(
            history,
            HIS_TIME,
            "history time",
        )
        history_name = find_name(
            history,
            HIS_STATION_NAME,
            "history station names",
        )
        history_x = find_name(
            history,
            HIS_STATION_X,
            "history station x coordinate",
        )
        history_y = find_name(
            history,
            HIS_STATION_Y,
            "history station y coordinate",
        )
        history_u_name = find_name(
            history,
            HIS_U,
            "history x-wind variable",
        )
        history_v_name = find_name(
            history,
            HIS_V,
            "history y-wind variable",
        )

        station_dim = station_dimension(
            history[history_u_name],
            history_time,
        )

        history_u = reduce_time_station(
            history[history_u_name],
            history_time,
            station_dim,
        )
        history_v = reduce_time_station(
            history[history_v_name],
            history_time,
            station_dim,
        )

        station_names = decode_names(
            history[history_name].values
        )
        station_x = np.asarray(
            history[history_x].values,
            dtype=float,
        ).reshape(-1)
        station_y = np.asarray(
            history[history_y].values,
            dtype=float,
        ).reshape(-1)

        all_history_times = pd.DatetimeIndex(
            pd.to_datetime(history[history_time].values)
        )
        selected_times = select_history_times(
            all_history_times
        )

        if len(selected_times) > 1:
            time_differences = np.diff(
                selected_times.values.astype("datetime64[s]")
            ).astype("timedelta64[s]").astype(float)

            print(
                "History time resolution: median={0:.1f} s, "
                "minimum={1:.1f} s, maximum={2:.1f} s"
                .format(
                    float(np.median(time_differences)),
                    float(np.min(time_differences)),
                    float(np.max(time_differences)),
                )
            )
        else:
            print(
                "Only one history time is available in the selected window."
            )

        print("Meteo variables:")
        print(
            "  {0}, {1}, {2}, {3}, {4}"
            .format(
                meteo_time,
                meteo_x,
                meteo_y,
                meteo_u,
                meteo_v,
            )
        )
        print("History variables:")
        print(
            "  {0}, {1}, {2}, {3}, {4}, {5}"
            .format(
                history_time,
                history_name,
                history_x,
                history_y,
                history_u_name,
                history_v_name,
            )
        )
        print("History station dimension: {0}".format(station_dim))
        print(
            "Selected history times: {0} to {1} ({2} records)"
            .format(
                selected_times[0],
                selected_times[-1],
                len(selected_times),
            )
        )
        print("")

        for requested_name, (
            point_x,
            point_y,
        ) in observation_points.items():

            # Prefer exact station-name matching when available.
            matching_indices = [
                index
                for index, existing_name in enumerate(station_names)
                if existing_name.strip().lower()
                == requested_name.strip().lower()
            ]

            if matching_indices:
                station_index = matching_indices[0]
                station_distance = float(
                    np.hypot(
                        station_x[station_index] - point_x,
                        station_y[station_index] - point_y,
                    )
                )
            else:
                station_index, station_distance = nearest_station(
                    station_x,
                    station_y,
                    point_x,
                    point_y,
                )

            if station_distance > maximum_station_distance_m:
                raise ValueError(
                    "{0}: selected history station '{1}' is {2:.2f} m "
                    "from the requested coordinate, exceeding the "
                    "{3:.2f} m limit."
                    .format(
                        requested_name,
                        station_names[station_index],
                        station_distance,
                        maximum_station_distance_m,
                    )
                )

            forcing_u, forcing_v = interpolate_meteo(
                dataset=meteo,
                x_coordinate=point_x,
                y_coordinate=point_y,
                target_times=selected_times,
                time_name=meteo_time,
                x_name=meteo_x,
                y_name=meteo_y,
                u_name=meteo_u,
                v_name=meteo_v,
            )

            history_u_values = extract_history_at_station(
                data=history_u,
                station_dim=station_dim,
                station_index=station_index,
                time_name=history_time,
                selected_times=selected_times,
            )
            history_v_values = extract_history_at_station(
                data=history_v,
                station_dim=station_dim,
                station_index=station_index,
                time_name=history_time,
                selected_times=selected_times,
            )

            forcing_speed = speed(
                forcing_u,
                forcing_v,
            )
            history_speed = speed(
                history_u_values,
                history_v_values,
            )

            forcing_direction = direction_from(
                forcing_u,
                forcing_v,
            )
            history_direction = direction_from(
                history_u_values,
                history_v_values,
            )

            direction_error = angle_difference(
                history_direction,
                forcing_direction,
            )

            print(
                "{0}: history station '{1}', index={2}, "
                "distance={3:.3f} m"
                .format(
                    requested_name,
                    station_names[station_index],
                    station_index,
                    station_distance,
                )
            )

            for index, timestamp in enumerate(selected_times):
                records.append(
                    {
                        "station": requested_name,
                        "history_station_name": (
                            station_names[station_index]
                        ),
                        "history_station_index": station_index,
                        "requested_x_m": point_x,
                        "requested_y_m": point_y,
                        "history_station_x_m": (
                            station_x[station_index]
                        ),
                        "history_station_y_m": (
                            station_y[station_index]
                        ),
                        "station_distance_m": station_distance,
                        "time": timestamp,
                        "forcing_x_wind_m_s": forcing_u[index],
                        "forcing_y_wind_m_s": forcing_v[index],
                        "history_x_wind_m_s": (
                            history_u_values[index]
                        ),
                        "history_y_wind_m_s": (
                            history_v_values[index]
                        ),
                        "forcing_speed_m_s": forcing_speed[index],
                        "history_speed_m_s": history_speed[index],
                        "speed_error_history_minus_forcing_m_s": (
                            history_speed[index]
                            - forcing_speed[index]
                        ),
                        "forcing_direction_from_deg": (
                            forcing_direction[index]
                        ),
                        "history_direction_from_deg": (
                            history_direction[index]
                        ),
                        "direction_error_history_minus_forcing_deg": (
                            direction_error[index]
                        ),
                    }
                )

    table = pd.DataFrame.from_records(records)

    if table.empty:
        raise RuntimeError(
            "No comparison records were generated."
        )

    timeseries_csv = (
        output_directory / "his_vs_meteo_timeseries.csv"
    )
    table.to_csv(
        timeseries_csv,
        index=False,
        float_format="%.8f",
    )

    summary_records = []

    for station in observation_points:
        subset = table[
            table["station"] == station
        ]

        summary = {
            "station": station,
            "history_station_name": (
                subset["history_station_name"].iloc[0]
            ),
            "station_distance_m": (
                subset["station_distance_m"].iloc[0]
            ),
            "number_of_times": len(subset),
        }

        for label, forcing_column, history_column in [
            (
                "x_wind",
                "forcing_x_wind_m_s",
                "history_x_wind_m_s",
            ),
            (
                "y_wind",
                "forcing_y_wind_m_s",
                "history_y_wind_m_s",
            ),
            (
                "speed",
                "forcing_speed_m_s",
                "history_speed_m_s",
            ),
        ]:
            forcing_values = subset[
                forcing_column
            ].to_numpy(dtype=float)
            history_values = subset[
                history_column
            ].to_numpy(dtype=float)

            errors = history_values - forcing_values

            summary[label + "_bias_m_s"] = float(
                np.nanmean(errors)
            )
            summary[label + "_rmse_m_s"] = float(
                np.sqrt(np.nanmean(errors ** 2))
            )
            summary[label + "_maximum_absolute_error_m_s"] = float(
                np.nanmax(np.abs(errors))
            )
            summary[label + "_correlation"] = correlation(
                forcing_values,
                history_values,
            )

        direction_errors = subset[
            "direction_error_history_minus_forcing_deg"
        ].to_numpy(dtype=float)

        summary["direction_mean_absolute_error_deg"] = float(
            np.nanmean(np.abs(direction_errors))
        )
        summary["direction_maximum_absolute_error_deg"] = float(
            np.nanmax(np.abs(direction_errors))
        )

        summary_records.append(summary)

    summary_table = pd.DataFrame.from_records(
        summary_records
    )

    summary_csv = (
        output_directory / "his_vs_meteo_summary.csv"
    )
    summary_table.to_csv(
        summary_csv,
        index=False,
        float_format="%.8f",
    )

    for station in observation_points:
        plot_station_timeseries(
            table=table,
            station=station,
            output_path=(
                output_directory
                / ("timeseries_" + station + ".png")
            ),
        )

    plot_scatter(
        table=table,
        output_path=(
            output_directory
            / "his_vs_meteo_scatter.png"
        ),
    )

    print("")
    print("Comparison finished.")
    print("Detailed CSV:")
    print("  {0}".format(timeseries_csv.resolve()))
    print("Summary CSV:")
    print("  {0}".format(summary_csv.resolve()))
    print("Scatter plot:")
    print(
        "  {0}".format(
            (
                output_directory
                / "his_vs_meteo_scatter.png"
            ).resolve()
        )
    )
    print("")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
