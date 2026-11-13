# -*- coding: utf-8 -*-
"""Extract named DCSM-FM history stations to 3D child-model BC files.

The child boundary PLI files must already exist. Their support-point names
must exactly match the station names stored in the DCSM history file.
No PLI, EXT, coordinate transformation, or velocity rotation is performed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# USER SETTINGS
# =============================================================================
HISTORY_FILE = Path(r"C:\dev\Python\DCSMmodelToNestedModel\hist_files\DCSM-FM_0_5nm_60min_his.nc")
OUTPUT_DIRECTORY = Path(r"bc_files")

T_bc = 60 #min

# Recommended child-model vertical grid:
#   Kmx         = 10
#   Layertype   = 1
#   StretchType = 1
#   StretchCoef = 4.0 5.9 8.7 12.7 18.7 18.7 12.7 8.7 5.9 4.0
#
# Layer THICKNESSES in percent, ordered BED -> SURFACE. They must be positive
# and sum to 100. BC profile positions are calculated at the layer centres.
SIGMA_LAYER_THICKNESS_PERCENT = np.array(
    [4.0, 5.9, 8.7, 12.7, 18.7, 18.7, 12.7, 8.7, 5.9, 4.0],
    dtype=float,
)

# Only history stations matching at least one expression are exported.
# Names are preserved exactly; they are not renamed or renumbered.
BOUNDARY_STATION_PATTERNS = [
    r"^North(?:[-_ ]|\d)",
    r"^Offshore(?:[-_ ]|\d)",
    r"^South(?:[-_ ]|\d)",
]

# Optional exact list. When non-empty, it replaces the pattern selection.
# This is the safest mode because every expected PLI point is stated exactly.
# Example:
# BOUNDARY_STATION_NAMES = [
#     "North_0001", "North_0002", "Offshore_0001", "South_0001"
# ]
BOUNDARY_STATION_NAMES: list[str] = []

# If None, the first history timestamp is used as the BC reference time.
# Example alternative: pd.Timestamp("2013-01-01 00:00:00")
BC_REFERENCE_TIME: pd.Timestamp | None = None

# Keep zero unless a verified parent-to-child vertical-datum difference exists.
WATERLEVEL_OFFSET_M = 0.0

OUTPUT_WATERLEVEL = f"nested_waterlevel_{T_bc}min.bc"
OUTPUT_VELOCITY = f"nested_uxuyadvectionvelocity_{T_bc}min.bc"
OUTPUT_SALINITY = f"nested_salinity_{T_bc}min.bc"
OUTPUT_TEMPERATURE = f"nested_temperature_{T_bc}min.bc"


# =============================================================================
# EXPECTED HISTORY VARIABLES
# =============================================================================

CANDIDATES = {
    "station_name": ["station_name", "stations"],
    "waterlevel": ["waterlevel", "s1"],
    "waterdepth": ["waterdepth", "water_depth", "hs"],
    "x_velocity": ["x_velocity", "ucx", "u"],
    "y_velocity": ["y_velocity", "ucy", "v"],
    "salinity": ["salinity", "sa1", "salt"],
    "temperature": ["temperature", "tem1", "water_temperature"],
    "zcoordinate_c": ["zcoordinate_c", "z_coordinate_c", "zcor"],
}


# =============================================================================
# HELPERS
# =============================================================================

def sigma_layer_centres(thickness_percent: np.ndarray) -> np.ndarray:
    """Return layer-centre positions in percent upward from the bed."""
    thickness = np.asarray(thickness_percent, dtype=float)
    if thickness.ndim != 1 or thickness.size == 0:
        raise ValueError("SIGMA_LAYER_THICKNESS_PERCENT must be a 1-D array.")
    if np.any(thickness <= 0):
        raise ValueError("All sigma-layer thicknesses must be positive.")
    if not np.isclose(thickness.sum(), 100.0, atol=1e-8):
        raise ValueError(
            "SIGMA_LAYER_THICKNESS_PERCENT must sum to 100; "
            f"found {thickness.sum():.12g}."
        )
    lower_interfaces = np.concatenate(([0.0], np.cumsum(thickness)[:-1]))
    return lower_interfaces + 0.5 * thickness


def find_variable(ds: xr.Dataset, role: str) -> str:
    for name in CANDIDATES[role]:
        if name in ds.variables:
            return name
    raise KeyError(
        f"Could not find {role!r}. Tried {CANDIDATES[role]}.\n"
        f"Available variables:\n{list(ds.variables)}"
    )


def decode_station_names(values: np.ndarray) -> list[str]:
    """Decode NetCDF string arrays and two-dimensional character arrays."""
    array = np.asarray(values)
    if array.ndim == 2 and array.dtype.kind in {"S", "U"}:
        names = []
        for row in array:
            name = "".join(
                item.decode("utf-8")
                if isinstance(item, (bytes, np.bytes_))
                else str(item)
                for item in row
            )
            names.append(name.replace("\x00", "").strip())
    else:
        names = []
        for item in array.reshape(-1):
            name = (
                item.decode("utf-8")
                if isinstance(item, (bytes, np.bytes_))
                else str(item)
            )
            names.append(name.replace("\x00", "").strip())

    if any(not name for name in names):
        raise ValueError("At least one empty station name was found.")
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"History station names are not unique: {duplicates}")
    return names


def select_boundary_station_indices(station_names: list[str]) -> list[int]:
    """Select stations while preserving history-file order and exact names."""
    if BOUNDARY_STATION_NAMES:
        if len(BOUNDARY_STATION_NAMES) != len(set(BOUNDARY_STATION_NAMES)):
            raise ValueError("BOUNDARY_STATION_NAMES contains duplicates.")
        missing = [name for name in BOUNDARY_STATION_NAMES if name not in station_names]
        if missing:
            raise ValueError(
                "These explicitly requested boundary names are absent from "
                f"the history file: {missing}"
            )
        return [station_names.index(name) for name in BOUNDARY_STATION_NAMES]

    indices = [
        i
        for i, name in enumerate(station_names)
        if any(
            re.search(pattern, name, flags=re.IGNORECASE)
            for pattern in BOUNDARY_STATION_PATTERNS
        )
    ]
    if not indices:
        raise ValueError(
            "No history stations match BOUNDARY_STATION_PATTERNS. "
            "Inspect station_name and update the patterns or provide "
            "BOUNDARY_STATION_NAMES explicitly."
        )
    return indices


def identify_dimensions(
    ds: xr.Dataset,
    station_name_var: str,
    waterlevel_var: str,
    z_var: str,
) -> tuple[str, str, str]:
    if "time" not in ds.variables:
        raise KeyError("The history file has no time variable.")
    time_dim = ds["time"].dims[0]

    station_dims = [d for d in ds[waterlevel_var].dims if d != time_dim]
    if len(station_dims) != 1:
        raise ValueError(
            f"Cannot identify station dimension from {waterlevel_var}: "
            f"{ds[waterlevel_var].dims}"
        )
    station_dim = station_dims[0]
    if station_dim not in ds[station_name_var].dims:
        raise ValueError(
            f"{station_name_var} does not use station dimension {station_dim}."
        )

    layer_dims = [d for d in ds[z_var].dims if d not in {time_dim, station_dim}]
    if len(layer_dims) != 1:
        raise ValueError(
            f"Cannot identify vertical dimension from {z_var}: {ds[z_var].dims}"
        )
    return time_dim, station_dim, layer_dims[0]


def require_dims(
    da: xr.DataArray,
    expected: tuple[str, ...],
    role: str,
) -> xr.DataArray:
    missing = [dimension for dimension in expected if dimension not in da.dims]
    if missing:
        raise ValueError(
            f"{role} variable {da.name!r} lacks dimensions {missing}; "
            f"found {da.dims}."
        )
    return da.transpose(*expected)


def interpolate_profile(
    z_source: np.ndarray,
    values_source: np.ndarray,
    z_target: np.ndarray,
) -> np.ndarray:
    """Interpolate vertically; hold closest values outside valid centres."""
    z = np.asarray(z_source, dtype=float)
    values = np.asarray(values_source, dtype=float)
    target = np.asarray(z_target, dtype=float)
    valid = np.isfinite(z) & np.isfinite(values)

    if valid.sum() == 0:
        return np.full(target.shape, np.nan)
    if valid.sum() == 1:
        return np.full(target.shape, values[valid][0])

    z = z[valid]
    values = values[valid]
    order = np.argsort(z)
    z = z[order]
    values = values[order]
    z, unique_indices = np.unique(z, return_index=True)
    values = values[unique_indices]
    if z.size == 1:
        return np.full(target.shape, values[0])
    return np.interp(target, z, values, left=values[0], right=values[-1])


def time_unit_and_seconds(times: pd.DatetimeIndex) -> tuple[str, np.ndarray]:
    reference = (
        pd.Timestamp(times[0])
        if BC_REFERENCE_TIME is None
        else pd.Timestamp(BC_REFERENCE_TIME)
    )
    elapsed = ((times - reference) / pd.Timedelta(seconds=1)).to_numpy(float)
    if elapsed[0] < 0 or np.any(np.diff(elapsed) <= 0):
        raise ValueError("BC times must be non-negative and strictly increasing.")
    rounded = np.rint(elapsed)
    if not np.allclose(elapsed, rounded, atol=1e-6):
        raise ValueError("Times are not integer seconds from BC_REFERENCE_TIME.")
    return (
        f"seconds since {reference:%Y-%m-%d %H:%M:%S} +00:00",
        rounded.astype(np.int64),
    )


def forcing_prefix(
    handle,
    station_name: str,
    function: str,
    time_unit: str,
    vertical_positions: np.ndarray,
) -> None:
    handle.write("[Forcing]\n")
    handle.write(f"name              = {station_name}\n")
    handle.write(f"function          = {function}\n")
    handle.write("timeInterpolation = linear\n")
    if function == "t3d":
        positions = " ".join(f"{position:.8g}" for position in vertical_positions)
        handle.write(f"vertPositions     = {positions}\n")
        handle.write("vertInterpolation = linear\n")
        handle.write("vertPositionType  = percBed\n")
    handle.write("quantity          = time\n")
    handle.write(f"unit              = {time_unit}\n")


def write_scalar_t3d_header(
    handle,
    station_name: str,
    time_unit: str,
    vertical_positions: np.ndarray,
    quantity: str,
    unit: str,
) -> None:
    forcing_prefix(handle, station_name, "t3d", time_unit, vertical_positions)
    for layer_number in range(1, len(vertical_positions) + 1):
        handle.write(f"quantity          = {quantity}\n")
        handle.write(f"unit              = {unit}\n")
        handle.write(f"vertPositionIndex = {layer_number}\n")


def write_velocity_t3d_header(
    handle,
    station_name: str,
    time_unit: str,
    vertical_positions: np.ndarray,
) -> None:
    forcing_prefix(handle, station_name, "t3d", time_unit, vertical_positions)
    handle.write("vector            = uxuyadvectionvelocitybnd:ux,uy\n")
    for layer_number in range(1, len(vertical_positions) + 1):
        handle.write("quantity          = ux\n")
        handle.write("unit              = m/s\n")
        handle.write(f"vertPositionIndex = {layer_number}\n")
        handle.write("quantity          = uy\n")
        handle.write("unit              = m/s\n")
        handle.write(f"vertPositionIndex = {layer_number}\n")


def write_data_row(handle, time_seconds: int, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("A non-finite value would be written to a BC file.")
    handle.write(f"{time_seconds:12d}")
    for value in array:
        handle.write(f" {value:.8g}")
    handle.write("\n")


# =============================================================================
# CONVERSION
# =============================================================================

def main() -> None:
    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(f"History file not found: {HISTORY_FILE}")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    vertical_positions = sigma_layer_centres(SIGMA_LAYER_THICKNESS_PERCENT)
    vertical_fractions = vertical_positions / 100.0
    print("BC vertical positions (% upward from bed):")
    print(" ".join(f"{value:.3f}" for value in vertical_positions))

    with xr.open_dataset(HISTORY_FILE, decode_times=True, mask_and_scale=True) as ds:
        variable_names = {role: find_variable(ds, role) for role in CANDIDATES}
        station_names = decode_station_names(
            ds[variable_names["station_name"]].values
        )
        selected_indices = select_boundary_station_indices(station_names)

        print(f"Selected {len(selected_indices)} boundary stations:")
        for index in selected_indices:
            print(f"  {station_names[index]}")

        time_dim, station_dim, layer_dim = identify_dimensions(
            ds,
            variable_names["station_name"],
            variable_names["waterlevel"],
            variable_names["zcoordinate_c"],
        )
        times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        time_unit, time_seconds = time_unit_and_seconds(times)

        waterlevel = require_dims(
            ds[variable_names["waterlevel"]],
            (time_dim, station_dim),
            "waterlevel",
        ).load()
        waterdepth = require_dims(
            ds[variable_names["waterdepth"]],
            (time_dim, station_dim),
            "waterdepth",
        ).load()
        expected_3d_dims = (time_dim, station_dim, layer_dim)
        z_coordinate = require_dims(
            ds[variable_names["zcoordinate_c"]],
            expected_3d_dims,
            "zcoordinate_c",
        ).load()
        x_velocity = require_dims(
            ds[variable_names["x_velocity"]],
            expected_3d_dims,
            "x_velocity",
        ).load()
        y_velocity = require_dims(
            ds[variable_names["y_velocity"]],
            expected_3d_dims,
            "y_velocity",
        ).load()
        salinity = require_dims(
            ds[variable_names["salinity"]],
            expected_3d_dims,
            "salinity",
        ).load()
        temperature = require_dims(
            ds[variable_names["temperature"]],
            expected_3d_dims,
            "temperature",
        ).load()

        output_paths = {
            "waterlevel": OUTPUT_DIRECTORY / OUTPUT_WATERLEVEL,
            "velocity": OUTPUT_DIRECTORY / OUTPUT_VELOCITY,
            "salinity": OUTPUT_DIRECTORY / OUTPUT_SALINITY,
            "temperature": OUTPUT_DIRECTORY / OUTPUT_TEMPERATURE,
        }

        with output_paths["waterlevel"].open(
            "w", encoding="utf-8", newline="\n"
        ) as f_wl, output_paths["velocity"].open(
            "w", encoding="utf-8", newline="\n"
        ) as f_uv, output_paths["salinity"].open(
            "w", encoding="utf-8", newline="\n"
        ) as f_sal, output_paths["temperature"].open(
            "w", encoding="utf-8", newline="\n"
        ) as f_tem:
            for station_index in selected_indices:
                station_name = station_names[station_index]
                wl_values = np.asarray(
                    waterlevel.isel({station_dim: station_index}), dtype=float
                )
                depth_values = np.asarray(
                    waterdepth.isel({station_dim: station_index}), dtype=float
                )
                z_values = np.asarray(
                    z_coordinate.isel({station_dim: station_index}), dtype=float
                )
                ux_values = np.asarray(
                    x_velocity.isel({station_dim: station_index}), dtype=float
                )
                uy_values = np.asarray(
                    y_velocity.isel({station_dim: station_index}), dtype=float
                )
                sal_values = np.asarray(
                    salinity.isel({station_dim: station_index}), dtype=float
                )
                tem_values = np.asarray(
                    temperature.isel({station_dim: station_index}), dtype=float
                )

                bed_values = wl_values - depth_values
                target_z = (
                    bed_values[:, None]
                    + depth_values[:, None] * vertical_fractions[None, :]
                )
                shape = (len(times), len(vertical_positions))
                ux_output = np.full(shape, np.nan)
                uy_output = np.full(shape, np.nan)
                sal_output = np.full(shape, np.nan)
                tem_output = np.full(shape, np.nan)

                for time_index, timestamp in enumerate(times):
                    if (
                        not np.isfinite(depth_values[time_index])
                        or depth_values[time_index] <= 0
                    ):
                        raise ValueError(
                            f"{station_name} is dry or has invalid depth at "
                            f"{timestamp}."
                        )
                    ux_output[time_index] = interpolate_profile(
                        z_values[time_index], ux_values[time_index], target_z[time_index]
                    )
                    uy_output[time_index] = interpolate_profile(
                        z_values[time_index], uy_values[time_index], target_z[time_index]
                    )
                    sal_output[time_index] = interpolate_profile(
                        z_values[time_index], sal_values[time_index], target_z[time_index]
                    )
                    tem_output[time_index] = interpolate_profile(
                        z_values[time_index], tem_values[time_index], target_z[time_index]
                    )

                # Names are written exactly as stored in station_name.
                forcing_prefix(
                    f_wl, station_name, "timeseries", time_unit, vertical_positions
                )
                f_wl.write("quantity          = waterlevelbnd\n")
                f_wl.write("unit              = m\n")
                for seconds, value in zip(
                    time_seconds, wl_values + WATERLEVEL_OFFSET_M
                ):
                    write_data_row(f_wl, int(seconds), np.array([value]))
                f_wl.write("\n")

                write_velocity_t3d_header(
                    f_uv, station_name, time_unit, vertical_positions
                )
                for time_index, seconds in enumerate(time_seconds):
                    # Column order: ux1, uy1, ux2, uy2, ...
                    velocity_row = np.column_stack(
                        (ux_output[time_index], uy_output[time_index])
                    ).reshape(-1)
                    write_data_row(f_uv, int(seconds), velocity_row)
                f_uv.write("\n")

                write_scalar_t3d_header(
                    f_sal,
                    station_name,
                    time_unit,
                    vertical_positions,
                    "salinitybnd",
                    "ppt",
                )
                for time_index, seconds in enumerate(time_seconds):
                    write_data_row(f_sal, int(seconds), sal_output[time_index])
                f_sal.write("\n")

                write_scalar_t3d_header(
                    f_tem,
                    station_name,
                    time_unit,
                    vertical_positions,
                    "temperaturebnd",
                    "degC",
                )
                for time_index, seconds in enumerate(time_seconds):
                    write_data_row(f_tem, int(seconds), tem_output[time_index])
                f_tem.write("\n")

    print("\nFinished writing BC files:")
    for path in output_paths.values():
        print(f"  {path.resolve()}")
    print(
        "\nIMPORTANT: every [Forcing] name must exactly match the "
        "support-point name expected from the existing PLI files."
    )


if __name__ == "__main__":
    main()
