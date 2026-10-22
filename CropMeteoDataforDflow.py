# -*- coding: utf-8 -*-
"""
Crop and reproject DCSM ERA5 meteorological forcing for a Cartesian
D-Flow FM model in EPSG:28992 (RD New), using the D-Flow FM grid extent
and an automatically selected regular meteorological-grid spacing.

Compatible with Python 3.9.

The output NetCDF is written on a regular projected grid and contains:
    x(x)  standard_name = projection_x_coordinate
    y(y)  standard_name = projection_y_coordinate
    time(time)
    air_pressure(time, y, x)
    eastward_wind(time, y, x)
    northward_wind(time, y, x)
    charnock(time, y, x)
    spatial_ref

The wind vector is interpolated from ERA5 longitude/latitude and rotated
from geographic east/north into the projected x/y axes of EPSG:28992.

Use the output with:

    QUANTITY = airpressure_windx_windy_charnock
    FILENAME = meteo/ERA5_child_test_RDNew_flowgrid.nc
    FILETYPE = 11
    METHOD   = 3
    OPERAND  = O

Required packages:
    numpy, pandas, xarray, netCDF4, scipy, pyproj

Install with:
    conda install numpy pandas xarray netcdf4 scipy pyproj
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Geod, Transformer
from scipy.spatial import cKDTree


# =====================================================================
# USER SETTINGS
# =====================================================================

source_nc = Path(
    r"D:\ERA5_2005-2021_dfm_mslp_u10n_v10n_chnk_rhoairalt.nc"
)

# D-Flow FM unstructured grid used to define the forcing extent and
# determine an appropriate regular meteorological-grid spacing.
flow_grid_file = Path(
    r"C:\dev\DCSM-FM\TestMeteo_flowonly\dflowfm\flowgrid.nc"
)

output_nc = Path(
    r"C:\dev\DCSM-FM\TestMeteo_flowonly\dflowfm\meteo"
    r"\ERA5_child_test_RDNew_flowgrid.nc"
)

start_time = pd.Timestamp("2013-12-05 00:00:00")
end_time = pd.Timestamp("2013-12-06 04:00:00")

# Target model CRS.
target_crs = "EPSG:28992"

# ERA5 source CRS.
source_crs = "EPSG:4326"

# Automatic regular meteorological-grid spacing. The script estimates:
#   1. nearest-neighbour spacing of D-Flow FM face centres; and
#   2. native ERA5 spacing near the model-domain centre.
# It then uses the finer of:
#   - twice the median D-Flow FM face spacing; and
#   - one quarter of the ERA5 spacing,
# constrained to the range below and rounded to a convenient value.
automatic_grid_spacing = True
manual_target_spacing_m = 5000.0
minimum_target_spacing_m = 2000.0
maximum_target_spacing_m = 10000.0
source_cells_per_target_cell = 4.0
mesh_median_spacing_multiplier = 2.0

# Maximum number of flow faces sampled for the nearest-neighbour estimate.
maximum_mesh_sample_points = 100000

# Buffer around the flow model. The actual buffer is at least two target
# meteo cells, even when this value is smaller.
buffer_m = 20000.0

# Confirmed source names.
source_time_name = "time"
source_x_dim = "x"
source_y_dim = "y"
source_lon_name = "X"
source_lat_name = "Y"
source_pressure_name = "air_pressure_fixed_height"
source_u_name = "eastward_wind"
source_v_name = "northward_wind"
source_charnock_name = "C"

compression_level = 4
fill_value = np.float32(9.96921e36)


# =====================================================================
# READ D-FLOW FM GRID AND CHOOSE TARGET SPACING
# =====================================================================

FLOW_FACE_X_CANDIDATES = [
    "mesh2d_face_x",
    "mesh2d_flowelem_xcc",
    "FlowElem_xcc",
    "flowelem_xcc",
    "mesh2d_face_x_coordinate",
]

FLOW_FACE_Y_CANDIDATES = [
    "mesh2d_face_y",
    "mesh2d_flowelem_ycc",
    "FlowElem_ycc",
    "flowelem_ycc",
    "mesh2d_face_y_coordinate",
]

FLOW_NODE_X_CANDIDATES = [
    "mesh2d_node_x",
    "NetNode_x",
    "netnode_x",
]

FLOW_NODE_Y_CANDIDATES = [
    "mesh2d_node_y",
    "NetNode_y",
    "netnode_y",
]


def first_existing_name(
    dataset: xr.Dataset,
    candidates: List[str],
) -> str:
    """Return the first candidate present in a dataset."""
    for name in candidates:
        if name in dataset.variables or name in dataset.coords:
            return name

    raise KeyError(
        "Could not find any of these variables: {0}. Available names: {1}"
        .format(
            ", ".join(candidates),
            ", ".join(sorted(dataset.variables)),
        )
    )


def read_flow_grid_points(
    grid_path: Path,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Read representative projected points from a D-Flow FM net/grid file.

    Face centres are preferred because wind is ultimately interpolated to
    flow cells. Node coordinates are used only as a fallback.
    """
    if not grid_path.is_file():
        raise FileNotFoundError(
            "D-Flow FM grid file not found: {0}".format(grid_path)
        )

    with xr.open_dataset(
        grid_path,
        decode_times=False,
        engine="netcdf4",
    ) as grid:
        try:
            x_name = first_existing_name(grid, FLOW_FACE_X_CANDIDATES)
            y_name = first_existing_name(grid, FLOW_FACE_Y_CANDIDATES)
            point_type = "flow-face centres"
        except KeyError:
            x_name = first_existing_name(grid, FLOW_NODE_X_CANDIDATES)
            y_name = first_existing_name(grid, FLOW_NODE_Y_CANDIDATES)
            point_type = "flow-grid nodes"

        x = np.asarray(grid[x_name].values, dtype=float).reshape(-1)
        y = np.asarray(grid[y_name].values, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        raise ValueError(
            "Fewer than two valid coordinates were found in {0}."
            .format(grid_path)
        )

    return x, y, x_name, y_name


def estimate_mesh_spacing(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Estimate local mesh spacing from nearest-neighbour distances.

    Returns the 25th, 50th and 75th percentiles. Duplicate coordinates are
    excluded from the estimate.
    """
    points = np.column_stack([x, y])

    if points.shape[0] > maximum_mesh_sample_points:
        # Deterministic evenly distributed sample; no random seed needed.
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            maximum_mesh_sample_points,
            dtype=int,
        )
        points = points[indices]

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = np.asarray(distances[:, 1], dtype=float)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]

    if nearest.size == 0:
        raise ValueError(
            "Could not estimate non-zero nearest-neighbour mesh spacing."
        )

    p25, p50, p75 = np.percentile(nearest, [25.0, 50.0, 75.0])
    return float(p25), float(p50), float(p75)


def estimate_source_spacing_m(
    source: xr.Dataset,
    centre_x: float,
    centre_y: float,
) -> Tuple[float, float, float]:
    """
    Estimate native ERA5 spacing near the flow-domain centre.

    Returns zonal spacing, meridional spacing and their smaller value.
    """
    longitudes = np.asarray(source[source_lon_name].values, dtype=float)
    latitudes = np.asarray(source[source_lat_name].values, dtype=float)

    if longitudes.size < 2 or latitudes.size < 2:
        raise ValueError("Source longitude/latitude arrays are too small.")

    to_geographic = Transformer.from_crs(
        target_crs,
        source_crs,
        always_xy=True,
    )
    centre_lon, centre_lat = to_geographic.transform(centre_x, centre_y)

    ix = int(np.argmin(np.abs(longitudes - centre_lon)))
    iy = int(np.argmin(np.abs(latitudes - centre_lat)))

    ix0 = max(0, min(ix, longitudes.size - 2))
    iy0 = max(0, min(iy, latitudes.size - 2))

    geod = Geod(ellps="WGS84")

    _, _, dx = geod.inv(
        float(longitudes[ix0]),
        float(latitudes[iy]),
        float(longitudes[ix0 + 1]),
        float(latitudes[iy]),
    )
    _, _, dy = geod.inv(
        float(longitudes[ix]),
        float(latitudes[iy0]),
        float(longitudes[ix]),
        float(latitudes[iy0 + 1]),
    )

    dx = abs(float(dx))
    dy = abs(float(dy))
    return dx, dy, min(dx, dy)


def nice_spacing(value: float) -> float:
    """Round upward to a convenient 1, 2, 2.5, 5 or 10 × power-of-ten."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Target spacing must be positive and finite.")

    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    normalized = value / scale

    for candidate in [1.0, 2.0, 2.5, 5.0, 10.0]:
        if normalized <= candidate:
            return float(candidate * scale)

    return float(10.0 * scale)


def choose_target_spacing(
    mesh_median_m: float,
    source_spacing_m: float,
) -> float:
    """Choose and round the regular meteorological-grid spacing."""
    if not automatic_grid_spacing:
        raw_spacing = float(manual_target_spacing_m)
    else:
        mesh_based = (
            mesh_median_spacing_multiplier * mesh_median_m
        )
        source_based = (
            source_spacing_m / source_cells_per_target_cell
        )

        # Use the finer requirement. This limits the additional interpolation
        # error while avoiding an excessively fine regular forcing grid.
        raw_spacing = min(mesh_based, source_based)

    bounded = min(
        maximum_target_spacing_m,
        max(minimum_target_spacing_m, raw_spacing),
    )

    rounded = nice_spacing(bounded)

    # nice_spacing rounds upward; apply the upper limit again.
    return min(maximum_target_spacing_m, rounded)


def snap_down(value: float, spacing: float) -> float:
    return float(np.floor(value / spacing) * spacing)


def snap_up(value: float, spacing: float) -> float:
    return float(np.ceil(value / spacing) * spacing)


def build_target_grid(
    grid_path: Path,
    source: xr.Dataset,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build an automatically spaced regular EPSG:28992 forcing grid.

    Returns x, y, target_dx and target_dy.
    """
    flow_x, flow_y, x_name, y_name = read_flow_grid_points(grid_path)

    xmin = float(np.min(flow_x))
    xmax = float(np.max(flow_x))
    ymin = float(np.min(flow_y))
    ymax = float(np.max(flow_y))

    mesh_p25, mesh_p50, mesh_p75 = estimate_mesh_spacing(flow_x, flow_y)

    centre_x = 0.5 * (xmin + xmax)
    centre_y = 0.5 * (ymin + ymax)
    source_dx, source_dy, source_min = estimate_source_spacing_m(
        source,
        centre_x,
        centre_y,
    )

    spacing = choose_target_spacing(mesh_p50, source_min)
    target_dx = spacing
    target_dy = spacing

    effective_buffer = max(buffer_m, 2.0 * spacing)

    xmin_out = snap_down(xmin - effective_buffer, target_dx)
    xmax_out = snap_up(xmax + effective_buffer, target_dx)
    ymin_out = snap_down(ymin - effective_buffer, target_dy)
    ymax_out = snap_up(ymax + effective_buffer, target_dy)

    x = np.arange(
        xmin_out,
        xmax_out + 0.5 * target_dx,
        target_dx,
        dtype=float,
    )
    y = np.arange(
        ymin_out,
        ymax_out + 0.5 * target_dy,
        target_dy,
        dtype=float,
    )

    if x.size < 2 or y.size < 2:
        raise ValueError("Generated target grid is too small.")

    print("D-Flow FM coordinate variables: {0}, {1}".format(x_name, y_name))
    print(
        "Flow-grid extent: x={0:.1f} to {1:.1f}, y={2:.1f} to {3:.1f} m"
        .format(xmin, xmax, ymin, ymax)
    )
    print(
        "Nearest-neighbour flow spacing: p25={0:.1f}, "
        "median={1:.1f}, p75={2:.1f} m"
        .format(mesh_p25, mesh_p50, mesh_p75)
    )
    print(
        "Native source spacing near domain centre: "
        "dx={0:.1f}, dy={1:.1f} m"
        .format(source_dx, source_dy)
    )
    print(
        "Selected meteo spacing: dx=dy={0:.1f} m "
        "(automatic={1})"
        .format(spacing, automatic_grid_spacing)
    )
    print(
        "Effective buffer: {0:.1f} m; target grid: {1} x {2}"
        .format(effective_buffer, x.size, y.size)
    )

    return x, y, target_dx, target_dy


# =====================================================================
# SOURCE PREPARATION AND VECTOR ROTATION
# =====================================================================

def prepare_source_variable(
    source: xr.Dataset,
    variable_name: str,
) -> xr.DataArray:
    """
    Attach source longitude/latitude as real xarray coordinates.
    """
    data = source[variable_name].assign_coords(
        longitude=(
            source_x_dim,
            np.asarray(source[source_lon_name].values, dtype=float),
        ),
        latitude=(
            source_y_dim,
            np.asarray(source[source_lat_name].values, dtype=float),
        ),
    )

    data = data.swap_dims(
        {
            source_x_dim: "longitude",
            source_y_dim: "latitude",
        }
    )

    if np.any(np.diff(data["longitude"].values) < 0):
        data = data.sortby("longitude")

    if np.any(np.diff(data["latitude"].values) < 0):
        data = data.sortby("latitude")

    return data


def projected_geographic_basis(
    lon: np.ndarray,
    lat: np.ndarray,
    projected_x: np.ndarray,
    projected_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return projected components of geographic east and north unit vectors.

    ERA5 u/v are geographic eastward/northward. D-Flow FM on a Cartesian
    grid needs wind components aligned with projected x/y.
    """
    geod = Geod(ellps="WGS84")
    to_projected = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    distance_m = np.full(lon.shape, 100.0)

    lon_e, lat_e, _ = geod.fwd(
        lon,
        lat,
        np.full(lon.shape, 90.0),
        distance_m,
    )
    lon_n, lat_n, _ = geod.fwd(
        lon,
        lat,
        np.full(lon.shape, 0.0),
        distance_m,
    )

    x_e, y_e = to_projected.transform(lon_e, lat_e)
    x_n, y_n = to_projected.transform(lon_n, lat_n)

    east_dx = np.asarray(x_e) - projected_x
    east_dy = np.asarray(y_e) - projected_y
    north_dx = np.asarray(x_n) - projected_x
    north_dy = np.asarray(y_n) - projected_y

    east_length = np.hypot(east_dx, east_dy)
    north_length = np.hypot(north_dx, north_dy)

    east_x = east_dx / east_length
    east_y = east_dy / east_length
    north_x = north_dx / north_length
    north_y = north_dy / north_length

    return east_x, east_y, north_x, north_y


def interpolate_source(
    source_data: xr.DataArray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """Interpolate one source field to the projected target points."""
    lon_da = xr.DataArray(
        target_lon,
        dims=("y", "x"),
    )
    lat_da = xr.DataArray(
        target_lat,
        dims=("y", "x"),
    )

    interpolated = source_data.interp(
        longitude=lon_da,
        latitude=lat_da,
        method="linear",
    ).load()

    values = np.asarray(interpolated.values, dtype=float)

    if np.isnan(values).any():
        raise ValueError(
            "Interpolation produced NaNs. Increase the source coverage "
            "or reduce the target buffer."
        )

    return values


# =====================================================================
# OUTPUT DATASET
# =====================================================================

def make_crs_variable(crs: CRS) -> xr.DataArray:
    """Create a scalar CF grid-mapping variable."""
    cf_attrs = crs.to_cf()

    cf_attrs.update(
        {
            "spatial_ref": crs.to_wkt(),
            "crs_wkt": crs.to_wkt(),
            "epsg_code": "EPSG:{0}".format(crs.to_epsg()),
            "EPSG_code": "EPSG:{0}".format(crs.to_epsg()),
        }
    )

    return xr.DataArray(
        np.int32(crs.to_epsg()),
        attrs=cf_attrs,
    )


def write_projected_meteo() -> None:
    """Create the projected D-Flow FM meteorological NetCDF."""
    if not source_nc.is_file():
        raise FileNotFoundError(
            "Source NetCDF not found: {0}".format(source_nc)
        )

    if end_time < start_time:
        raise ValueError("end_time must not precede start_time.")

    with xr.open_dataset(
        source_nc,
        decode_times=True,
        engine="netcdf4",
    ) as source:

        x, y, target_dx_m, target_dy_m = build_target_grid(
            flow_grid_file,
            source,
        )
        x_mesh, y_mesh = np.meshgrid(x, y)

        to_geographic = Transformer.from_crs(
            target_crs,
            source_crs,
            always_xy=True,
        )

        lon, lat = to_geographic.transform(x_mesh, y_mesh)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)

        east_x, east_y, north_x, north_y = projected_geographic_basis(
            lon=lon,
            lat=lat,
            projected_x=x_mesh,
            projected_y=y_mesh,
        )

        required = [
            source_time_name,
            source_lon_name,
            source_lat_name,
            source_pressure_name,
            source_u_name,
            source_v_name,
            source_charnock_name,
        ]

        missing = [
            name
            for name in required
            if name not in source.variables
        ]

        if missing:
            raise KeyError(
                "Source file is missing: {0}".format(", ".join(missing))
            )

        source_times = pd.DatetimeIndex(
            pd.to_datetime(source[source_time_name].values)
        )

        if start_time < source_times[0] or end_time > source_times[-1]:
            raise ValueError(
                "Requested time range lies outside source coverage "
                "{0} to {1}."
                .format(source_times[0], source_times[-1])
            )

        pressure_src = prepare_source_variable(
            source,
            source_pressure_name,
        ).sel(
            {source_time_name: slice(start_time, end_time)}
        )

        u_src = prepare_source_variable(
            source,
            source_u_name,
        ).sel(
            {source_time_name: slice(start_time, end_time)}
        )

        v_src = prepare_source_variable(
            source,
            source_v_name,
        ).sel(
            {source_time_name: slice(start_time, end_time)}
        )

        charnock_src = prepare_source_variable(
            source,
            source_charnock_name,
        ).sel(
            {source_time_name: slice(start_time, end_time)}
        )

        output_times = pd.DatetimeIndex(
            pd.to_datetime(pressure_src[source_time_name].values)
        )

        if output_times.size == 0:
            raise ValueError(
                "No source records fall in the requested time interval."
            )

        print("Interpolating air pressure...")
        pressure = interpolate_source(
            pressure_src,
            lon,
            lat,
        )

        print("Interpolating eastward wind...")
        u_east = interpolate_source(
            u_src,
            lon,
            lat,
        )

        print("Interpolating northward wind...")
        v_north = interpolate_source(
            v_src,
            lon,
            lat,
        )

        print("Interpolating Charnock parameter...")
        charnock = interpolate_source(
            charnock_src,
            lon,
            lat,
        )

    # Rotate geographic east/north components to projected x/y.
    x_wind = (
        u_east * east_x[np.newaxis, :, :]
        + v_north * north_x[np.newaxis, :, :]
    )
    y_wind = (
        u_east * east_y[np.newaxis, :, :]
        + v_north * north_y[np.newaxis, :, :]
    )

    source_speed = np.hypot(u_east, v_north)
    projected_speed = np.hypot(x_wind, y_wind)
    speed_error = np.nanmax(
        np.abs(source_speed - projected_speed)
    )

    print(
        "Maximum wind-speed change from basis rotation: "
        "{0:.3e} m/s".format(float(speed_error))
    )

    target_crs_object = CRS.from_user_input(target_crs)

    dataset = xr.Dataset(
        coords={
            "time": (
                "time",
                output_times.values.astype("datetime64[ns]"),
                {
                    "standard_name": "time",
                    "long_name": "time",
                    "axis": "T",
                },
            ),
            "x": (
                "x",
                x,
                {
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate of projection",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                "y",
                y,
                {
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate of projection",
                    "units": "m",
                    "axis": "Y",
                },
            ),
        },
        data_vars={
            "spatial_ref": make_crs_variable(target_crs_object),

            # Canonical EC/D-Flow FM names:
            "air_pressure": (
                ("time", "y", "x"),
                pressure.astype(np.float32),
                {
                    "standard_name": "air_pressure",
                    "long_name": "mean sea level air pressure",
                    "units": "Pa",
                    "coordinates": "y x",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "eastward_wind": (
                ("time", "y", "x"),
                x_wind.astype(np.float32),
                {
                    # The reader identifies projected coordinates through
                    # x/y; the variable represents projected model-x wind.
                    "standard_name": "eastward_wind",
                    "long_name": (
                        "10 metre wind component along projected x axis"
                    ),
                    "units": "m s-1",
                    "coordinates": "y x",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "northward_wind": (
                ("time", "y", "x"),
                y_wind.astype(np.float32),
                {
                    "standard_name": "northward_wind",
                    "long_name": (
                        "10 metre wind component along projected y axis"
                    ),
                    "units": "m s-1",
                    "coordinates": "y x",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "charnock": (
                ("time", "y", "x"),
                charnock.astype(np.float32),
                {
                    "standard_name": "charnock",
                    "long_name": "Charnock parameter",
                    "units": "1",
                    "coordinates": "y x",
                    "grid_mapping": "spatial_ref",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": (
                "DCSM ERA5 forcing reprojected to EPSG:28992 "
                "for D-Flow FM"
            ),
            "source": str(source_nc),
            "history": (
                "Created by create_dflowfm_projected_meteo_auto_spacing.py"
            ),
            "target_crs": target_crs,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "buffer_m": float(buffer_m),
            "target_dx_m": float(target_dx_m),
            "target_dy_m": float(target_dy_m),
            "flow_grid_file": str(flow_grid_file),
        },
    )

    # Explicitly encode time in a D-Flow FM-friendly form.
    time_units = (
        "seconds since {0} +00:00"
        .format(output_times[0].strftime("%Y-%m-%d %H:%M:%S"))
    )

    encoding = {
        "time": {
            "dtype": "float64",
            "units": time_units,
            "calendar": "gregorian",
        },
        "x": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "y": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "spatial_ref": {
            "dtype": "int32",
            "_FillValue": None,
        },
    }

    for variable_name in [
        "air_pressure",
        "eastward_wind",
        "northward_wind",
        "charnock",
    ]:
        encoding[variable_name] = {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "_FillValue": fill_value,
        }

    output_nc.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_netcdf(
        output_nc,
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )

    dataset.close()

    # Reopen and validate exactly what was written.
    with xr.open_dataset(
        output_nc,
        decode_times=True,
        engine="netcdf4",
    ) as check:
        required_output = [
            "time",
            "x",
            "y",
            "spatial_ref",
            "air_pressure",
            "eastward_wind",
            "northward_wind",
            "charnock",
        ]

        absent = [
            name
            for name in required_output
            if name not in check.variables
        ]

        if absent:
            raise RuntimeError(
                "Output validation failed; missing: {0}"
                .format(", ".join(absent))
            )

        if (
            check["x"].attrs.get("standard_name")
            != "projection_x_coordinate"
        ):
            raise RuntimeError(
                "Output x coordinate lacks projection_x_coordinate."
            )

        if (
            check["y"].attrs.get("standard_name")
            != "projection_y_coordinate"
        ):
            raise RuntimeError(
                "Output y coordinate lacks projection_y_coordinate."
            )

        print("")
        print("Projected meteorological NetCDF created successfully.")
        print("Output: {0}".format(output_nc.resolve()))
        print("Dimensions: {0}".format(dict(check.sizes)))
        print(
            "Time: {0} to {1}"
            .format(
                pd.Timestamp(check["time"].values[0]),
                pd.Timestamp(check["time"].values[-1]),
            )
        )
        print(
            "x standard_name: {0}"
            .format(check["x"].attrs.get("standard_name"))
        )
        print(
            "y standard_name: {0}"
            .format(check["y"].attrs.get("standard_name"))
        )
        print(
            "CRS grid_mapping_name: {0}"
            .format(
                check["spatial_ref"].attrs.get(
                    "grid_mapping_name"
                )
            )
        )


if __name__ == "__main__":
    write_projected_meteo()
