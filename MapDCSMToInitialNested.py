#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create D-Flow FM child initial fields from partitioned DCSM map files.

Outputs
-------
1. initial_waterlevel_YYYYMMDD_HHMMSS.xyz
2. initial_velocity_x_YYYYMMDD_HHMMSS.xyz
3. initial_velocity_y_YYYYMMDD_HHMMSS.xyz
4. initial_salinity_temperature_YYYYMMDD_HHMMSS.nc
5. initial_fields.ext             (an EXT snippet to merge/check)
6. interpolation_diagnostics.csv

The scalar NetCDF deliberately follows the proven DCSM/CMEMS schema:
``thetao(time, depth, latitude, longitude)`` and
``so(time, depth, latitude, longitude)``.  Two identical records bracket the
initial instant so D-Flow FM can initialize/nudge without extrapolating time.

This script does NOT manufacture a solver restart.  D-Flow FM should create
its own restart after a short child-model adjustment run.

Dependencies
------------
conda install -c conda-forge xarray netcdf4 numpy scipy pyproj pandas
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

MAP_DIRECTORY = Path(r"C:\dev\Python\DCSMmodelToNestedModel\map_files_DCSM")
MAP_PATTERN = "*map.nc"
CHILD_GRID = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\Nested_grid\flowgrid.nc"
)
OUTPUT_DIRECTORY = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\Nested_initial"
)

TARGET_TIME = np.datetime64("2013-01-01T00:00:00")

# Horizontal interpolation. IDW_NEIGHBOURS=4 is a stable default for DCSM.
IDW_NEIGHBOURS = 4
IDW_POWER = 2.0
SOURCE_BUFFER_DEGREES = 0.35

# Regular lon/lat grid used by the salinity-temperature nudge NetCDF.
# 0.025 degrees is about 1.7 km east-west at 52 degrees N. This is finer than
# the 0.5 nautical-mile DCSM cell size only by a modest factor and avoids a
# needlessly large forcing file.
# Resolution of the regular RD New nudge grid.
NUDGE_RESOLUTION_METRES = 2000.0

# The second record is identical and brackets the initial time.
NUDGE_SECOND_RECORD_HOURS = 24.0

# The child grid is EPSG:28992 (Amersfoort / RD New).
CHILD_CRS = "EPSG:28992"
GEOGRAPHIC_CRS = "EPSG:4326"

# Abort if the requested timestamp is farther from a map record than this.
MAX_TIME_DIFFERENCE_SECONDS = 1.0


# DCSM variable names confirmed by the supplied file inventory.
VARS = {
    "x": "mesh2d_face_x",
    "y": "mesh2d_face_y",
    "waterlevel": "mesh2d_s1",
    "velocity_x": "mesh2d_ucxa",
    "velocity_y": "mesh2d_ucya",
    "salinity": "mesh2d_sa1",
    "temperature": "mesh2d_tem1",
    "layer_z": "mesh2d_layer_z",
    "layer_sigma": "mesh2d_layer_sigma",
    "sigma_z_depth": "mesh2d_sigmazdepth",
    "bed_depth": "mesh2d_bldepth",
}

# Standard 50-level CMEMS/GLORYS depth axis. Using this familiar positive-down
# axis makes the generated file structurally consistent with the inspected
# original DCSM CMEMS salinity/temperature forcing. In a shallow child domain,
# levels below the seabed remain missing and are ignored by D-Flow FM.
CMEMS_DEPTHS_M = np.array([
    0.494025, 1.541375, 2.645669, 3.819495, 5.078224,
    6.440614, 7.929560, 9.572997, 11.405000, 13.467140,
    15.810070, 18.495560, 21.598820, 25.211410, 29.444730,
    34.434150, 40.344050, 47.373620, 55.764290, 65.807270,
    77.853850, 92.326070, 109.729300, 130.666000, 155.850700,
    186.125600, 222.475200, 266.040300, 318.127400, 380.213000,
    453.937700, 541.088900, 643.566800, 763.333100, 902.339300,
    1062.440000, 1245.291000, 1452.251000, 1684.284000, 1941.893000,
    2225.078000, 2533.336000, 2865.702000, 3220.820000, 3597.032000,
    3992.484000, 4405.224000, 4833.291000, 5274.784000, 5727.917000,
], dtype=np.float64)


# =============================================================================
# HELPERS
# =============================================================================

FILL_VALUE = np.float32(-999.0)


def require_variables(ds: xr.Dataset, names: List[str], filename: Path) -> None:
    missing = [name for name in names if name not in ds.variables]
    if missing:
        raise KeyError(
            "Missing variable(s) in {}: {}\nAvailable variables: {}".format(
                filename, missing, list(ds.variables)
            )
        )


def decoded_times(ds: xr.Dataset) -> np.ndarray:
    """Return a datetime64[ns] time vector, with a clear error if unavailable."""
    if "time" not in ds:
        raise KeyError("Dataset has no time coordinate")
    values = np.asarray(ds["time"].values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise TypeError(
            "Time was not decoded as datetime64. Open the map with "
            "decode_times=True and ensure cftime/netCDF4 is installed."
        )
    return values.astype("datetime64[ns]")


def select_time_index(ds: xr.Dataset, filename: Path) -> Tuple[int, float]:
    times = decoded_times(ds)
    target_ns = TARGET_TIME.astype("datetime64[ns]")
    delta_seconds = np.abs(
        (times - target_ns).astype("timedelta64[ns]").astype(np.int64)
    ) / 1.0e9
    index = int(np.argmin(delta_seconds))
    difference = float(delta_seconds[index])
    if difference > MAX_TIME_DIFFERENCE_SECONDS:
        raise ValueError(
            "{} has no record at {}. Nearest is {} (difference {:.1f} s).".format(
                filename.name, str(TARGET_TIME), str(times[index]), difference
            )
        )
    return index, difference


def as_float_array(da: xr.DataArray) -> np.ndarray:
    """Load values as float64 and convert fill values to NaN."""
    values = np.asarray(da.values, dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    # NetCDF default fill values (for example 9.969209968e36) are not always
    # exposed through attrs by every xarray/backend combination.
    values[np.abs(values) > 1.0e30] = np.nan
    fill = da.attrs.get("_FillValue", None)
    if fill is not None:
        values[np.isclose(values, float(fill))] = np.nan
    # Some Delft3D files use exactly -999 as a conventional fill value even
    # when the backend does not expose _FillValue. Do not use a broad
    # ``<= -900`` test here: mesh2d_layer_z legitimately reaches several
    # thousand metres below datum.
    values[np.isclose(values, -999.0, rtol=0.0, atol=1.0e-10)] = np.nan
    return values


def read_child_grid() -> Dict[str, np.ndarray]:
    with xr.open_dataset(CHILD_GRID, decode_times=False, mask_and_scale=True) as ds:
        require_variables(ds, ["mesh2d_face_x", "mesh2d_face_y"], CHILD_GRID)
        x_rd = as_float_array(ds["mesh2d_face_x"]).squeeze()
        y_rd = as_float_array(ds["mesh2d_face_y"]).squeeze()

    if x_rd.ndim != 1 or y_rd.shape != x_rd.shape:
        raise ValueError("Unexpected child face-coordinate shapes")
    if not np.all(np.isfinite(x_rd)) or not np.all(np.isfinite(y_rd)):
        raise ValueError("Child face coordinates contain invalid values")

    transformer = Transformer.from_crs(
        CHILD_CRS, GEOGRAPHIC_CRS, always_xy=True
    )
    longitude, latitude = transformer.transform(x_rd, y_rd)
    return {
        "x_rd": x_rd,
        "y_rd": y_rd,
        "longitude": np.asarray(longitude, dtype=np.float64),
        "latitude": np.asarray(latitude, dtype=np.float64),
    }


def inspect_vertical_coordinate(first_map: Path) -> Dict[str, np.ndarray]:
    """Read and validate the hybrid sigma-z coordinate definition.

    DCSM has fixed z levels below ``depth_c`` and sigma levels above it. The
    fill values in layer_z/layer_sigma are intentional and complementary.
    Physical sigma-layer elevations depend on local water level and bed depth,
    so they are reconstructed later for every face.
    """
    with xr.open_dataset(first_map, decode_times=True, mask_and_scale=True) as ds:
        require_variables(
            ds,
            [
                VARS["layer_z"], VARS["layer_sigma"],
                VARS["sigma_z_depth"], VARS["salinity"],
            ],
            first_map,
        )
        layer_z = as_float_array(ds[VARS["layer_z"]]).squeeze()
        layer_sigma = as_float_array(ds[VARS["layer_sigma"]]).squeeze()
        depth_c = float(np.asarray(ds[VARS["sigma_z_depth"]].values).squeeze())
        n_layers = ds[VARS["salinity"]].sizes.get("mesh2d_nLayers")

    if (
        layer_z.ndim != 1 or layer_sigma.ndim != 1
        or n_layers is None or layer_z.size != n_layers
        or layer_sigma.size != n_layers
    ):
        raise ValueError(
            "Cannot match the hybrid coordinate arrays to the DCSM layers."
        )
    z_mask = np.isfinite(layer_z)
    sigma_mask = np.isfinite(layer_sigma)
    if np.any(z_mask & sigma_mask) or np.any(~z_mask & ~sigma_mask):
        raise ValueError(
            "Each DCSM layer must be either a z-layer or a sigma-layer, "
            "but not both."
        )
    if not np.isfinite(depth_c) or depth_c <= 0.0:
        raise ValueError("Invalid mesh2d_sigmazdepth: {}".format(depth_c))
    if np.any(layer_sigma[sigma_mask] < -1.0) or np.any(layer_sigma[sigma_mask] > 0.0):
        raise ValueError("Sigma coordinates must lie between -1 and 0")

    print(
        "Hybrid vertical grid: {} z-layers, {} sigma-layers, depth_c={} m".format(
            int(z_mask.sum()), int(sigma_mask.sum()), depth_c
        )
    )
    return {
        "layer_z": layer_z,
        "layer_sigma": layer_sigma,
        "depth_c": np.array(depth_c, dtype=np.float64),
    }


def physical_layer_depths(
    waterlevel: np.ndarray,
    bed_depth: np.ndarray,
    vertical: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return layer-centre depth below the instantaneous water surface.

    Implements the CF ``ocean_sigma_z_coordinate`` formula advertised by the
    map file:

      fixed z layer: z = zlev
      sigma layer:   z = eta + sigma * (min(depth_c, depth) + eta)

    The returned depth is ``eta - z`` and is positive downward. Values at or
    below the local bed are marked missing. The DCSM scalar mask is applied
    separately as an additional safeguard.
    """
    eta = np.asarray(waterlevel, dtype=np.float64)[:, None]
    depth = np.asarray(bed_depth, dtype=np.float64)[:, None]
    layer_z = vertical["layer_z"][None, :]
    sigma = vertical["layer_sigma"][None, :]
    depth_c = float(vertical["depth_c"])

    if np.any(~np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("Selected parent faces contain invalid bed depth")

    elevation = np.full((eta.shape[0], layer_z.shape[1]), np.nan, dtype=np.float64)
    z_mask = np.isfinite(layer_z[0])
    sigma_mask = np.isfinite(sigma[0])
    elevation[:, z_mask] = layer_z[:, z_mask]
    sigma_thickness = np.minimum(depth, depth_c) + eta
    elevation[:, sigma_mask] = eta + sigma[:, sigma_mask] * sigma_thickness

    layer_depth = eta - elevation
    instantaneous_water_depth = depth + eta
    active = (
        np.isfinite(layer_depth)
        & (layer_depth >= 0.0)
        & (layer_depth < instantaneous_water_depth)
    )
    layer_depth[~active] = np.nan
    return layer_depth


def interpolate_profiles_to_depths(
    source_depth: np.ndarray,
    source_value: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    """Vertically interpolate face profiles to a common positive-down axis."""
    if source_depth.shape != source_value.shape:
        raise ValueError("Depth and scalar profile arrays do not match")
    output = np.full(
        (source_value.shape[0], target_depth.size), np.nan, dtype=np.float64
    )
    for face in range(source_value.shape[0]):
        valid = np.isfinite(source_depth[face]) & np.isfinite(source_value[face])
        if valid.sum() < 1:
            continue
        z = source_depth[face, valid]
        value = source_value[face, valid]
        order = np.argsort(z)
        z = z[order]
        value = value[order]
        z, unique_index = np.unique(z, return_index=True)
        value = value[unique_index]
        if z.size == 1:
            # Only use the single value close to its represented layer.
            close = np.abs(target_depth - z[0]) <= max(0.5, 0.25 * z[0])
            output[face, close] = value[0]
            continue
        # Hold the uppermost layer constant to the surface. Do not extrapolate
        # downward below the deepest active layer centre.
        output[face] = np.interp(
            target_depth, z, value, left=value[0], right=np.nan
        )
    return output


def read_parent_maps(
    files: List[Path],
    bounds: Tuple[float, float, float, float],
    vertical: Dict[str, np.ndarray],
    output_depths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Read the selected time and retain only DCSM profiles near the child."""
    lon_min, lon_max, lat_min, lat_max = bounds
    chunks = {
        "longitude": [],
        "latitude": [],
        "waterlevel": [],
        "velocity_x": [],
        "velocity_y": [],
        "salinity": [],
        "temperature": [],
    }  # type: Dict[str, List[np.ndarray]]

    required = [
        VARS["x"], VARS["y"], VARS["waterlevel"], VARS["velocity_x"],
        VARS["velocity_y"], VARS["salinity"], VARS["temperature"],
        VARS["layer_z"], VARS["layer_sigma"], VARS["sigma_z_depth"],
        VARS["bed_depth"], "time",
    ]

    for number, filename in enumerate(files, 1):
        print("[{0:02d}/{1:02d}] Reading {2}".format(number, len(files), filename.name))
        with xr.open_dataset(
            filename, decode_times=True, mask_and_scale=True, cache=False
        ) as ds:
            require_variables(ds, required, filename)
            time_index, _ = select_time_index(ds, filename)
            lon = as_float_array(ds[VARS["x"]]).squeeze()
            lat = as_float_array(ds[VARS["y"]]).squeeze()
            bed_all = as_float_array(ds[VARS["bed_depth"]]).squeeze()
            waterlevel_all = as_float_array(
                ds[VARS["waterlevel"]].isel(time=time_index)
            ).squeeze()
            if bed_all.shape != lon.shape or waterlevel_all.shape != lon.shape:
                raise ValueError(
                    "Face coordinates, bed depth and water level do not share "
                    "the same face dimension in {}".format(filename.name)
                )
            keep = (
                np.isfinite(lon) & np.isfinite(lat)
                & (lon >= lon_min) & (lon <= lon_max)
                & (lat >= lat_min) & (lat <= lat_max)
                # The geographic buffer also contains land/dry faces. They do
                # not define a usable 3-D water-column profile.
                & np.isfinite(bed_all) & (bed_all > 0.0)
                & np.isfinite(waterlevel_all)
                & ((bed_all + waterlevel_all) > 0.01)
            )
            indices = np.flatnonzero(keep)
            if indices.size == 0:
                continue

            def take_2d(name: str) -> np.ndarray:
                da = ds[VARS[name]].isel(time=time_index, mesh2d_nFaces=indices)
                result = as_float_array(da).squeeze()
                if result.shape != (indices.size,):
                    raise ValueError("Unexpected shape for {}: {}".format(name, result.shape))
                return result

            def take_3d(name: str) -> np.ndarray:
                da = ds[VARS[name]].isel(time=time_index, mesh2d_nFaces=indices)
                da = da.transpose("mesh2d_nFaces", "mesh2d_nLayers")
                result = as_float_array(da)
                if result.ndim != 2 or result.shape[0] != indices.size:
                    raise ValueError("Unexpected shape for {}: {}".format(name, result.shape))
                return result

            chunks["longitude"].append(lon[indices])
            chunks["latitude"].append(lat[indices])
            waterlevel = waterlevel_all[indices]
            bed_depth = bed_all[indices]
            if bed_depth.shape != (indices.size,):
                raise ValueError("Unexpected bed-depth shape: {}".format(bed_depth.shape))
            salinity_native = take_3d("salinity")
            temperature_native = take_3d("temperature")
            layer_depth = physical_layer_depths(waterlevel, bed_depth, vertical)

            # A layer is active only where both its physical coordinate and its
            # DCSM scalar are valid. Interpolate vertically before combining
            # profiles horizontally.
            salinity_common = interpolate_profiles_to_depths(
                layer_depth, salinity_native, output_depths
            )
            temperature_common = interpolate_profiles_to_depths(
                layer_depth, temperature_native, output_depths
            )

            chunks["waterlevel"].append(waterlevel)
            chunks["velocity_x"].append(take_2d("velocity_x"))
            chunks["velocity_y"].append(take_2d("velocity_y"))
            chunks["salinity"].append(salinity_common)
            chunks["temperature"].append(temperature_common)

    if not chunks["longitude"]:
        raise RuntimeError("No DCSM faces were found around the child domain")

    parent = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}

    # Remove exact duplicate partition-halo locations. Keep the first copy.
    coordinate_key = np.round(
        np.column_stack((parent["longitude"], parent["latitude"])), 10
    )
    _, unique_indices = np.unique(coordinate_key, axis=0, return_index=True)
    unique_indices.sort()
    parent = {key: value[unique_indices] for key, value in parent.items()}
    print("Parent profiles retained: {:,}".format(parent["longitude"].size))
    return parent


def lonlat_metric(longitude: np.ndarray, latitude: np.ndarray, lat0: float) -> np.ndarray:
    """Map lon/lat degrees to a local metric plane suitable for neighbour search."""
    earth_radius = 6371000.0
    x = np.deg2rad(longitude) * earth_radius * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(latitude) * earth_radius
    return np.column_stack((x, y))


def neighbour_map(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = float(np.nanmean(target_lat))
    source_xy = lonlat_metric(source_lon, source_lat, lat0)
    target_xy = lonlat_metric(target_lon, target_lat, lat0)
    tree = cKDTree(source_xy)
    k = min(IDW_NEIGHBOURS, source_xy.shape[0])
    distance, index = tree.query(target_xy, k=k)
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)


def idw(values: np.ndarray, distance: np.ndarray, index: np.ndarray) -> np.ndarray:
    """NaN-aware inverse-distance interpolation of 1-D or profile data."""
    selected = values[index]
    weights = 1.0 / np.maximum(distance, 0.01) ** IDW_POWER
    if selected.ndim == 3:
        weights = weights[:, :, None]
    valid = np.isfinite(selected)
    numerator = np.sum(np.where(valid, selected * weights, 0.0), axis=1)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator
    result[denominator == 0.0] = np.nan
    return result


def write_xyz(path: Path, x: np.ndarray, y: np.ndarray, values: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if not np.all(valid):
        raise ValueError("{} contains {} invalid child values".format(path.name, np.sum(~valid)))
    data = np.column_stack((x, y, values))
    np.savetxt(path, data, fmt=["%.3f", "%.3f", "%.10g"], delimiter=" ")


def make_regular_axis(minimum: float, maximum: float, spacing: float) -> np.ndarray:
    start = np.floor(minimum / spacing) * spacing
    stop = np.ceil(maximum / spacing) * spacing
    count = int(round((stop - start) / spacing)) + 1
    return start + np.arange(count, dtype=np.float64) * spacing


def write_nudge_netcdf(
    path: Path,
    parent: Dict[str, np.ndarray],
    child: Dict[str, np.ndarray],
    depth: np.ndarray,
) -> None:
    """
    Write salinity and temperature on a regular RD New grid.

    D-Flow FM requires projection_x_coordinate and
    projection_y_coordinate for a Cartesian child model.
    """

    # Regular RD New grid covering the child domain.
    x = make_regular_axis(
        float(child["x_rd"].min()),
        float(child["x_rd"].max()),
        NUDGE_RESOLUTION_METRES,
    )
    y = make_regular_axis(
        float(child["y_rd"].min()),
        float(child["y_rd"].max()),
        NUDGE_RESOLUTION_METRES,
    )

    x_grid, y_grid = np.meshgrid(x, y)

    # Convert target RD coordinates to WGS84 only for finding neighbouring
    # DCSM cells. Output coordinates remain RD New.
    to_wgs84 = Transformer.from_crs(
        CHILD_CRS,
        GEOGRAPHIC_CRS,
        always_xy=True,
    )

    target_lon, target_lat = to_wgs84.transform(
        x_grid.ravel(),
        y_grid.ravel(),
    )

    distance, index = neighbour_map(
        parent["longitude"],
        parent["latitude"],
        np.asarray(target_lon),
        np.asarray(target_lat),
    )

    salinity = idw(
        parent["salinity"],
        distance,
        index,
    )
    temperature = idw(
        parent["temperature"],
        distance,
        index,
    )

    # idw() returns (horizontal_points, depth).
    shape = (y.size, x.size, depth.size)

    salinity = (
        salinity.reshape(shape)
        .transpose(2, 0, 1)
    )
    temperature = (
        temperature.reshape(shape)
        .transpose(2, 0, 1)
    )

    # Two identical time records bracket initialization.
    salinity_4d = np.stack(
        (salinity, salinity),
        axis=0,
    ).astype(np.float32)

    temperature_4d = np.stack(
        (temperature, temperature),
        axis=0,
    ).astype(np.float32)

    times = np.array(
        [0.0, NUDGE_SECOND_RECORD_HOURS],
        dtype=np.float64,
    )

    date_text = pd.Timestamp(TARGET_TIME).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    ds = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "y", "x"),
                temperature_4d,
            ),
            "so": (
                ("time", "depth", "y", "x"),
                salinity_4d,
            ),
            "crs": xr.DataArray(
                np.int32(0),
                attrs={
                    "grid_mapping_name":
                        "oblique_stereographic",
                    "epsg_code":
                        "EPSG:28992",
                    "spatial_ref":
                        "EPSG:28992",
                    "longitude_of_projection_origin":
                        5.38763888888889,
                    "latitude_of_projection_origin":
                        52.1561605555556,
                    "scale_factor_at_projection_origin":
                        0.9999079,
                    "false_easting":
                        155000.0,
                    "false_northing":
                        463000.0,
                    "semi_major_axis":
                        6377397.155,
                    "inverse_flattening":
                        299.1528128,
                },
            ),
        },
        coords={
            "time": ("time", times),
            "depth": (
                "depth",
                depth.astype(np.float64),
            ),
            "x": (
                "x",
                x.astype(np.float64),
            ),
            "y": (
                "y",
                y.astype(np.float64),
            ),
        },
        attrs={
            "title":
                "DCSM-derived initial salinity and temperature",
            "source":
                "Interpolated from partitioned DCSM map files",
            "target_time":
                str(TARGET_TIME),
            "horizontal_crs":
                "Amersfoort / RD New (EPSG:28992)",
        },
    )

    ds["x"].attrs.update({
        "standard_name": "projection_x_coordinate",
        "long_name": "RD New easting",
        "units": "m",
        "axis": "X",
    })

    ds["y"].attrs.update({
        "standard_name": "projection_y_coordinate",
        "long_name": "RD New northing",
        "units": "m",
        "axis": "Y",
    })

    ds["depth"].attrs.update({
        "standard_name": "depth",
        "long_name": "Depth below water surface",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    })

    ds["time"].attrs.update({
        "standard_name": "time",
        "long_name": "Time",
        "units": "hours since {}".format(date_text),
        "calendar": "gregorian",
        "axis": "T",
    })

    ds["thetao"].attrs.update({
        "standard_name":
            "sea_water_potential_temperature",
        "long_name":
            "Temperature",
        "units":
            "degrees_C",
        "unit_long":
            "Degrees Celsius",
        "grid_mapping":
            "crs",
        "coordinates":
            "time depth y x",
    })

    ds["so"].attrs.update({
        "standard_name":
            "sea_water_salinity",
        "long_name":
            "Salinity",
        "units":
            "1e-3",
        "unit_long":
            "Practical Salinity Unit",
        "grid_mapping":
            "crs",
        "coordinates":
            "time depth y x",
    })

    encoding = {
        "thetao": {
            "dtype": "float32",
            "_FillValue": FILL_VALUE,
            "zlib": True,
            "complevel": 4,
        },
        "so": {
            "dtype": "float32",
            "_FillValue": FILL_VALUE,
            "zlib": True,
            "complevel": 4,
        },
        "time": {"dtype": "float64"},
        "depth": {"dtype": "float64"},
        "x": {"dtype": "float64"},
        "y": {"dtype": "float64"},
        "crs": {"dtype": "int32"},
    }

    ds.to_netcdf(
        path,
        engine="netcdf4",
        format="NETCDF4_CLASSIC",
        encoding=encoding,
    )
    ds.close()


def write_ext_snippet(
    path: Path, wl_name: str, ux_name: str, uy_name: str, nudge_name: str
) -> None:
    text = """# Merge these blocks into the OLD-style external forcing file used by FlowFM.mdu.
# Keep paths relative to the MDU directory. Verify FILETYPE/METHOD against the
# D-Flow FM version used in your container before the full production run.

QUANTITY     = initialwaterlevel
FILENAME     = {wl}
FILETYPE     = 7
METHOD       = 5
OPERAND      = O

QUANTITY     = initialvelocityx
FILENAME     = {ux}
FILETYPE     = 7
METHOD       = 5
OPERAND      = O

QUANTITY     = initialvelocityy
FILENAME     = {uy}
FILETYPE     = 7
METHOD       = 5
OPERAND      = O

QUANTITY     = nudge_salinity_temperature
FILENAME     = {nudge}
FILETYPE     = 11
METHOD       = 3
OPERAND      = O
""".format(wl=wl_name, ux=ux_name, uy=uy_name, nudge=nudge_name)
    path.write_text(text, encoding="utf-8")


def finite_range(name: str, values: np.ndarray) -> str:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return "{}: NO VALID VALUES".format(name)
    return "{}: {:.8g} to {:.8g}".format(name, valid.min(), valid.max())


# =============================================================================
# MAIN ANALYSIS / OUTPUT
# =============================================================================

def main() -> None:
    files = sorted(MAP_DIRECTORY.glob(MAP_PATTERN))
    if not files:
        raise FileNotFoundError("No DCSM maps match {}".format(MAP_DIRECTORY / MAP_PATTERN))
    if not CHILD_GRID.is_file():
        raise FileNotFoundError("Child grid not found: {}".format(CHILD_GRID))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("Found {} DCSM map files".format(len(files)))
    if len(files) != 32:
        warnings.warn("Expected 32 DCSM partitions; found {}".format(len(files)))

    child = read_child_grid()
    bounds = (
        float(child["longitude"].min()) - SOURCE_BUFFER_DEGREES,
        float(child["longitude"].max()) + SOURCE_BUFFER_DEGREES,
        float(child["latitude"].min()) - SOURCE_BUFFER_DEGREES,
        float(child["latitude"].max()) + SOURCE_BUFFER_DEGREES,
    )
    vertical = inspect_vertical_coordinate(files[0])
    depth = CMEMS_DEPTHS_M.copy()
    parent = read_parent_maps(files, bounds, vertical, depth)

    distance, index = neighbour_map(
        parent["longitude"], parent["latitude"],
        child["longitude"], child["latitude"],
    )
    waterlevel = idw(parent["waterlevel"], distance, index)
    velocity_x = idw(parent["velocity_x"], distance, index)
    velocity_y = idw(parent["velocity_y"], distance, index)

    stamp = pd.Timestamp(TARGET_TIME).strftime("%Y%m%d_%H%M%S")
    wl_path = OUTPUT_DIRECTORY / ("initial_waterlevel_{}.xyz".format(stamp))
    ux_path = OUTPUT_DIRECTORY / ("initial_velocity_x_{}.xyz".format(stamp))
    uy_path = OUTPUT_DIRECTORY / ("initial_velocity_y_{}.xyz".format(stamp))
    nudge_path = OUTPUT_DIRECTORY / (
        "initial_salinity_temperature_{}.nc".format(stamp)
    )
    ext_path = OUTPUT_DIRECTORY / "initial_fields.ext"
    diagnostics_path = OUTPUT_DIRECTORY / "interpolation_diagnostics.csv"

    write_xyz(wl_path, child["x_rd"], child["y_rd"], waterlevel)
    write_xyz(ux_path, child["x_rd"], child["y_rd"], velocity_x)
    write_xyz(uy_path, child["x_rd"], child["y_rd"], velocity_y)
    write_nudge_netcdf(nudge_path, parent, child, depth)
    write_ext_snippet(
        ext_path, wl_path.name, ux_path.name, uy_path.name, nudge_path.name
    )

    diagnostics = pd.DataFrame({
        "child_face_index_zero_based": np.arange(child["x_rd"].size),
        "x_rd_m": child["x_rd"],
        "y_rd_m": child["y_rd"],
        "longitude": child["longitude"],
        "latitude": child["latitude"],
        "nearest_dcsm_distance_m": distance[:, 0],
        "initial_waterlevel_m": waterlevel,
        "initial_velocity_x_m_s": velocity_x,
        "initial_velocity_y_m_s": velocity_y,
    })
    diagnostics.to_csv(diagnostics_path, index=False)

    print("\nValidation summary")
    print(finite_range("Water level (m)", waterlevel))
    print(finite_range("Velocity x (m/s)", velocity_x))
    print(finite_range("Velocity y (m/s)", velocity_y))
    print("Nearest DCSM distance: median {:.1f} m, maximum {:.1f} m".format(
        float(np.median(distance[:, 0])), float(np.max(distance[:, 0]))
    ))
    print("DCSM depth levels: {} ({:.3f} to {:.3f} m)".format(
        depth.size, float(depth.min()), float(depth.max())
    ))
    print("\nCreated files in {}:".format(OUTPUT_DIRECTORY))
    for output in [wl_path, ux_path, uy_path, nudge_path, ext_path, diagnostics_path]:
        print("  {}".format(output.name))
    print("\nSet IniWithNudge = 2 in the child MDU for initialization from the nudge file.")
    print("Do not also prescribe uniform initial salinity/temperature for this test.")


if __name__ == "__main__":
    main()