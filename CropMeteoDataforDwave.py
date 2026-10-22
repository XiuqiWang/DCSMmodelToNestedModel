# -*- coding: utf-8 -*-
"""
Create D-Waves/SWAN spatially varying wind files (.amu and .amv)
from the DCSM ERA5 meteorological NetCDF.

The script:
1. Reads the Cartesian SWAN/D-Waves .grd file.
2. Builds a regular EPSG:28992 meteorological grid around the wave grid.
3. Adds a user-defined spatial buffer.
4. Selects a user-defined time interval from the ERA5 NetCDF.
5. Interpolates ERA5 eastward/northward 10 m wind to that grid.
6. Rotates the geographic east/north vector components into local
   EPSG:28992 x/y components.
7. Writes Deltares FileVersion 1.03 files of type
   meteo_on_equidistant_grid:
       .amu -> quantity1 = x_wind
       .amv -> quantity1 = y_wind

Compatible with Python 3.9.

Required packages:
    numpy
    pandas
    xarray
    netCDF4
    scipy
    pyproj

Install with:
    conda install numpy pandas xarray netcdf4 scipy pyproj
"""

from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Transformer


# =====================================================================
# USER SETTINGS
# =====================================================================

# Original DCSM ERA5 NetCDF.
source_nc = Path(
    r"D:\ERA5_2005-2021_dfm_mslp_u10n_v10n_chnk_rhoairalt.nc"
)

# D-Waves/SWAN curvilinear computational grid.
wave_grid_file = Path(
    r"C:\dev\DCSM-FM\CHC4-couple\wave\Wave_new.grd"
)

# Output wind files referenced by MeteoFile in the .mdw.
output_amu = Path(
    r"ERA5_wave_test.amu"
)
output_amv = Path(
    r"ERA5_wave_test.amv"
)

# User-defined forcing interval. Endpoints are included when present.
start_time = pd.Timestamp("2013-12-05 00:00:00")
end_time = pd.Timestamp("2013-12-06 04:00:00")

# Time origin written in each TIME line.
# It is convenient to match the D-Waves ReferenceDate.
time_origin = pd.Timestamp("2013-12-05 00:00:00")

# CRS used by the Cartesian wave grid.
wave_grid_crs = "EPSG:28992"

# Geographic CRS used by the DCSM ERA5 coordinates.
source_crs = "EPSG:4326"

# Regular output meteo grid spacing and buffer.
#
# ERA5 is much coarser than the wave grid, so 10 km is normally adequate.
# Reducing this value creates a larger file but does not add information
# beyond the original ERA5 resolution.
output_dx_m = 10000.0
output_dy_m = 10000.0
buffer_m = 20000.0

# Source variable and coordinate names confirmed from the inspected file.
time_name = "time"
source_x_dimension = "x"
source_y_dimension = "y"
source_longitude_variable = "X"
source_latitude_variable = "Y"
source_u_variable = "eastward_wind"
source_v_variable = "northward_wind"

# D-Waves missing-value marker.
nodata_value = -999.0

# Number formatting.
values_per_line = None  # None means one complete output-grid row per text line.
value_format = "{:12.5f}"


# =====================================================================
# GRID READING
# =====================================================================

def read_delft3d_curvilinear_grid(
    grid_path: Path,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Read x/y coordinates from a Delft3D RGFGRID-style .grd file.

    The file contains:
        n_size ETA records of x coordinates,
        followed by n_size ETA records of y coordinates.
    """
    if not grid_path.is_file():
        raise FileNotFoundError(
            "Wave grid file not found: {0}".format(grid_path)
        )

    lines = grid_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    coordinate_system = ""
    missing_value = -999.999

    for line in lines:
        lower_line = line.lower()

        if "coordinate system" in lower_line and "=" in line:
            coordinate_system = line.split("=", 1)[1].strip()

        if "missing value" in lower_line and "=" in line:
            try:
                missing_value = float(line.split("=", 1)[1])
            except ValueError:
                pass

    dimension_line_index = None
    m_size = None
    n_size = None

    for line_index, line in enumerate(lines):
        if re.match(r"^\s*\d+\s+\d+\s*$", line):
            fields = line.split()
            m_size = int(fields[0])
            n_size = int(fields[1])
            dimension_line_index = line_index
            break

    if dimension_line_index is None:
        raise ValueError(
            "Could not find the grid dimensions in {0}."
            .format(grid_path)
        )

    records: List[List[float]] = []
    line_index = dimension_line_index + 1

    while line_index < len(lines):
        line = lines[line_index]

        if line.lstrip().upper().startswith("ETA="):
            fields = line.replace("ETA=", " ").split()

            # The first value is the ETA row number.
            values = [float(value) for value in fields[1:]]
            line_index += 1

            while (
                len(values) < m_size
                and line_index < len(lines)
                and not lines[line_index].lstrip().upper().startswith("ETA=")
            ):
                values.extend(
                    float(value)
                    for value in lines[line_index].split()
                )
                line_index += 1

            if len(values) != m_size:
                raise ValueError(
                    "Expected {0} values in an ETA record but found {1}."
                    .format(m_size, len(values))
                )

            records.append(values)
        else:
            line_index += 1

    expected_records = 2 * n_size

    if len(records) != expected_records:
        raise ValueError(
            "Expected {0} coordinate records, found {1}."
            .format(expected_records, len(records))
        )

    coordinate_array = np.asarray(records, dtype=float)

    x_grid = coordinate_array[:n_size, :]
    y_grid = coordinate_array[n_size:, :]

    return x_grid, y_grid, coordinate_system, missing_value


def valid_wave_grid_extent(
    grid_path: Path,
) -> Tuple[float, float, float, float]:
    """Return xmin, xmax, ymin, ymax of all valid wave-grid nodes."""
    x_grid, y_grid, coordinate_system, missing_value = (
        read_delft3d_curvilinear_grid(grid_path)
    )

    if coordinate_system.lower() != "cartesian":
        raise ValueError(
            "The script expects a Cartesian wave grid. Found: {0}"
            .format(coordinate_system)
        )

    tolerance = max(1.0e-8, abs(missing_value) * 1.0e-10)

    valid = (
        np.isfinite(x_grid)
        & np.isfinite(y_grid)
        & (np.abs(x_grid - missing_value) > tolerance)
        & (np.abs(y_grid - missing_value) > tolerance)
    )

    if not np.any(valid):
        raise ValueError("No valid wave-grid coordinates were found.")

    return (
        float(np.min(x_grid[valid])),
        float(np.max(x_grid[valid])),
        float(np.min(y_grid[valid])),
        float(np.max(y_grid[valid])),
    )


# =====================================================================
# TARGET METEO GRID
# =====================================================================

def snap_down(value: float, spacing: float) -> float:
    return float(np.floor(value / spacing) * spacing)


def snap_up(value: float, spacing: float) -> float:
    return float(np.ceil(value / spacing) * spacing)


def build_output_grid(
    grid_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regular EPSG:28992 grid covering the SWAN grid plus a buffer.

    Returned x and y arrays are cell-centre coordinates, increasing from
    west to east and south to north.
    """
    xmin, xmax, ymin, ymax = valid_wave_grid_extent(grid_path)

    xmin = snap_down(xmin - buffer_m, output_dx_m)
    xmax = snap_up(xmax + buffer_m, output_dx_m)
    ymin = snap_down(ymin - buffer_m, output_dy_m)
    ymax = snap_up(ymax + buffer_m, output_dy_m)

    x_centres = np.arange(
        xmin,
        xmax + 0.5 * output_dx_m,
        output_dx_m,
        dtype=float,
    )
    y_centres = np.arange(
        ymin,
        ymax + 0.5 * output_dy_m,
        output_dy_m,
        dtype=float,
    )

    if x_centres.size < 2 or y_centres.size < 2:
        raise ValueError("The generated meteorological grid is too small.")

    print(
        "Wave-grid extent: "
        "x={0:.3f} to {1:.3f}, y={2:.3f} to {3:.3f} m"
        .format(xmin + buffer_m, xmax - buffer_m,
                ymin + buffer_m, ymax - buffer_m)
    )
    print(
        "Output meteo grid: "
        "{0} columns x {1} rows, dx={2:g} m, dy={3:g} m"
        .format(
            x_centres.size,
            y_centres.size,
            output_dx_m,
            output_dy_m,
        )
    )
    print(
        "Output centres: "
        "x={0:.3f} to {1:.3f}, y={2:.3f} to {3:.3f}"
        .format(
            x_centres[0],
            x_centres[-1],
            y_centres[0],
            y_centres[-1],
        )
    )

    return x_centres, y_centres


# =====================================================================
# VECTOR-BASIS TRANSFORMATION
# =====================================================================

def geographic_basis_in_projected_xy(
    longitude: np.ndarray,
    latitude: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local projected x/y components of geographic unit vectors.

    The ERA5 variables are eastward and northward velocities. D-Waves
    reads x_wind/y_wind in the Cartesian model coordinate system.

    A 100 m geodesic perturbation is projected into EPSG:28992 to obtain:
        e_x, e_y : projected direction of geographic east
        n_x, n_y : projected direction of geographic north
    """
    geod = Geod(ellps="WGS84")
    to_projected = Transformer.from_crs(
        source_crs,
        wave_grid_crs,
        always_xy=True,
    )

    lon_east, lat_east, _ = geod.fwd(
        longitude,
        latitude,
        np.full(longitude.shape, 90.0),
        np.full(longitude.shape, 100.0),
    )
    lon_north, lat_north, _ = geod.fwd(
        longitude,
        latitude,
        np.full(longitude.shape, 0.0),
        np.full(longitude.shape, 100.0),
    )

    x_east, y_east = to_projected.transform(lon_east, lat_east)
    x_north, y_north = to_projected.transform(lon_north, lat_north)

    east_dx = np.asarray(x_east) - target_x
    east_dy = np.asarray(y_east) - target_y
    north_dx = np.asarray(x_north) - target_x
    north_dy = np.asarray(y_north) - target_y

    east_length = np.hypot(east_dx, east_dy)
    north_length = np.hypot(north_dx, north_dy)

    e_x = east_dx / east_length
    e_y = east_dy / east_length
    n_x = north_dx / north_length
    n_y = north_dy / north_length

    return e_x, e_y, n_x, n_y


# =====================================================================
# SOURCE INTERPOLATION
# =====================================================================

def prepare_source_variable(
    source: xr.Dataset,
    variable_name: str,
) -> xr.DataArray:
    """
    Attach longitude and latitude as coordinates and swap source dimensions
    to those coordinate names for xarray interpolation.
    """
    data = source[variable_name].assign_coords(
        longitude=(
            source_x_dimension,
            np.asarray(
                source[source_longitude_variable].values,
                dtype=float,
            ),
        ),
        latitude=(
            source_y_dimension,
            np.asarray(
                source[source_latitude_variable].values,
                dtype=float,
            ),
        ),
    )

    data = data.swap_dims(
        {
            source_x_dimension: "longitude",
            source_y_dimension: "latitude",
        }
    )

    # xarray interpolation requires monotonic coordinates.
    if np.any(np.diff(data["longitude"].values) < 0):
        data = data.sortby("longitude")

    if np.any(np.diff(data["latitude"].values) < 0):
        data = data.sortby("latitude")

    return data


def interpolate_wind_to_output_grid(
    x_centres: np.ndarray,
    y_centres: np.ndarray,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Interpolate source wind and return projected x/y wind arrays.

    Output shapes:
        (ntime, nrows, ncols)
    with y increasing from south to north.
    """
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time.")

    if not source_nc.is_file():
        raise FileNotFoundError(
            "Source meteorological file not found: {0}"
            .format(source_nc)
        )

    x_mesh, y_mesh = np.meshgrid(
        x_centres,
        y_centres,
    )

    to_geographic = Transformer.from_crs(
        wave_grid_crs,
        source_crs,
        always_xy=True,
    )

    longitude, latitude = to_geographic.transform(
        x_mesh,
        y_mesh,
    )

    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    e_x, e_y, n_x, n_y = geographic_basis_in_projected_xy(
        longitude=longitude,
        latitude=latitude,
        target_x=x_mesh,
        target_y=y_mesh,
    )

    with xr.open_dataset(
        source_nc,
        decode_times=True,
        engine="netcdf4",
    ) as source:

        required = [
            time_name,
            source_longitude_variable,
            source_latitude_variable,
            source_u_variable,
            source_v_variable,
        ]

        missing = [
            name
            for name in required
            if name not in source.variables
        ]

        if missing:
            raise KeyError(
                "Source NetCDF is missing: {0}"
                .format(", ".join(missing))
            )

        source_times = pd.DatetimeIndex(
            pd.to_datetime(source[time_name].values)
        )

        if start_time < source_times[0] or end_time > source_times[-1]:
            raise ValueError(
                "Requested time range is outside source coverage: "
                "{0} to {1}."
                .format(source_times[0], source_times[-1])
            )

        selected_times = source_times[
            (source_times >= start_time)
            & (source_times <= end_time)
        ]

        if selected_times.size == 0:
            raise ValueError(
                "No source time records occur in the requested interval."
            )

        u_source = prepare_source_variable(
            source,
            source_u_variable,
        ).sel(
            {time_name: slice(start_time, end_time)}
        )

        v_source = prepare_source_variable(
            source,
            source_v_variable,
        ).sel(
            {time_name: slice(start_time, end_time)}
        )

        target_longitude = xr.DataArray(
            longitude,
            dims=("meteo_y", "meteo_x"),
        )
        target_latitude = xr.DataArray(
            latitude,
            dims=("meteo_y", "meteo_x"),
        )

        print("Interpolating eastward wind...")
        u_east = u_source.interp(
            longitude=target_longitude,
            latitude=target_latitude,
            method="linear",
        ).load()

        print("Interpolating northward wind...")
        v_north = v_source.interp(
            longitude=target_longitude,
            latitude=target_latitude,
            method="linear",
        ).load()

        u_east_values = np.asarray(u_east.values, dtype=float)
        v_north_values = np.asarray(v_north.values, dtype=float)

    if (
        np.isnan(u_east_values).any()
        or np.isnan(v_north_values).any()
    ):
        raise ValueError(
            "Interpolation produced NaN values. Increase the source crop "
            "coverage or reduce the output buffer."
        )

    # Transform geographic east/north components into EPSG:28992 x/y.
    x_wind = (
        u_east_values * e_x[np.newaxis, :, :]
        + v_north_values * n_x[np.newaxis, :, :]
    )
    y_wind = (
        u_east_values * e_y[np.newaxis, :, :]
        + v_north_values * n_y[np.newaxis, :, :]
    )

    # Rotation should preserve speed to numerical precision.
    source_speed = np.hypot(u_east_values, v_north_values)
    target_speed = np.hypot(x_wind, y_wind)

    maximum_speed_error = float(
        np.nanmax(np.abs(source_speed - target_speed))
    )

    print(
        "Maximum speed difference after coordinate rotation: "
        "{0:.3e} m/s".format(maximum_speed_error)
    )

    return selected_times, x_wind, y_wind


# =====================================================================
# DELTARES METEO WRITER
# =====================================================================

def elapsed_hours(
    timestamp: pd.Timestamp,
    origin: pd.Timestamp,
) -> float:
    return float(
        (timestamp - origin) / pd.Timedelta(hours=1)
    )


def write_data_row(
    file_object,
    row_values: np.ndarray,
) -> None:
    """
    Write one grid row.

    D-Waves uses free-formatted input. By default, each complete grid row
    is written on one text line.
    """
    values = np.asarray(row_values, dtype=float)

    if values_per_line is None:
        file_object.write(
            " ".join(
                value_format.format(value)
                for value in values
            )
        )
        file_object.write("\n")
        return

    for start_index in range(0, len(values), values_per_line):
        part = values[
            start_index:start_index + values_per_line
        ]
        file_object.write(
            " ".join(
                value_format.format(value)
                for value in part
            )
        )
        file_object.write("\n")


def write_deltares_meteo_file(
    output_path: Path,
    quantity: str,
    times: pd.DatetimeIndex,
    values: np.ndarray,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
) -> None:
    """
    Write FileVersion 1.03 meteo_on_equidistant_grid format.

    Important ordering:
    - Header location is the centre of the lower-left cell.
    - Each time block is written from the highest-y row to the lowest-y row,
      matching the D-Waves manual example:
          do j = nrows, 1, -1
    - Within each row, x increases from west to east.
    """
    if quantity not in ("x_wind", "y_wind"):
        raise ValueError("Unsupported quantity: {0}".format(quantity))

    if values.shape != (
        len(times),
        len(y_centres),
        len(x_centres),
    ):
        raise ValueError(
            "Unexpected data shape {0} for {1}."
            .format(values.shape, quantity)
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    origin_string = time_origin.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with output_path.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as file_object:

        file_object.write(
            "### Generated from DCSM ERA5 by "
            "create_dwaves_wind_files.py\n"
        )
        file_object.write(
            "### Cartesian grid: {0}\n"
            .format(wave_grid_crs)
        )
        file_object.write(
            "FileVersion      = 1.03\n"
        )
        file_object.write(
            "filetype         = meteo_on_equidistant_grid\n"
        )
        file_object.write(
            "NODATA_value     = {0:g}\n"
            .format(nodata_value)
        )
        file_object.write(
            "n_cols           = {0}\n"
            .format(len(x_centres))
        )
        file_object.write(
            "n_rows           = {0}\n"
            .format(len(y_centres))
        )
        file_object.write(
            "grid_unit        = m\n"
        )
        file_object.write(
            "x_llcenter       = {0:.3f}\n"
            .format(x_centres[0])
        )
        file_object.write(
            "y_llcenter       = {0:.3f}\n"
            .format(y_centres[0])
        )
        file_object.write(
            "dx               = {0:.3f}\n"
            .format(output_dx_m)
        )
        file_object.write(
            "dy               = {0:.3f}\n"
            .format(output_dy_m)
        )
        file_object.write(
            "n_quantity       = 1\n"
        )
        file_object.write(
            "quantity1        = {0}\n"
            .format(quantity)
        )
        file_object.write(
            "unit1            = m s-1\n"
        )

        for time_index, timestamp in enumerate(times):
            hours = elapsed_hours(
                pd.Timestamp(timestamp),
                time_origin,
            )

            file_object.write(
                "TIME = {0:.10g} hours since {1} +00:00\n"
                .format(hours, origin_string)
            )

            # Highest-y row first, lowest-y row last.
            for row_index in range(
                len(y_centres) - 1,
                -1,
                -1,
            ):
                row = values[
                    time_index,
                    row_index,
                    :,
                ]

                row_to_write = np.where(
                    np.isfinite(row),
                    row,
                    nodata_value,
                )

                write_data_row(
                    file_object,
                    row_to_write,
                )


# =====================================================================
# VALIDATION
# =====================================================================

def validate_time_settings(
    times: pd.DatetimeIndex,
) -> None:
    if times.size == 0:
        raise ValueError("No output times are available.")

    if time_origin > times[0]:
        # Negative times are technically representable, but avoiding them
        # makes the forcing easier to inspect and configure.
        raise ValueError(
            "time_origin is later than the first forcing record."
        )

    differences = np.diff(
        times.values.astype("datetime64[s]")
    ).astype("timedelta64[s]").astype(float)

    if differences.size and np.any(differences <= 0):
        raise ValueError(
            "Forcing times must be strictly increasing."
        )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    x_centres, y_centres = build_output_grid(
        wave_grid_file
    )

    times, x_wind, y_wind = (
        interpolate_wind_to_output_grid(
            x_centres=x_centres,
            y_centres=y_centres,
        )
    )

    validate_time_settings(times)

    write_deltares_meteo_file(
        output_path=output_amu,
        quantity="x_wind",
        times=times,
        values=x_wind,
        x_centres=x_centres,
        y_centres=y_centres,
    )

    write_deltares_meteo_file(
        output_path=output_amv,
        quantity="y_wind",
        times=times,
        values=y_wind,
        x_centres=x_centres,
        y_centres=y_centres,
    )

    print("")
    print("Finished writing D-Waves wind files:")
    print("  {0}".format(output_amu.resolve()))
    print("  {0}".format(output_amv.resolve()))
    print(
        "Time range: {0} to {1} ({2} records)"
        .format(times[0], times[-1], len(times))
    )
    print("")
    print("Add these lines to the [General] section of the .mdw:")
    print("  MeteoFile = {0}".format(output_amu.name))
    print("  MeteoFile = {0}".format(output_amv.name))
    print("")
    print("Also set:")
    print("  FlowWind  = 0")
    print("  WindGrowth = true")


if __name__ == "__main__":
    main()