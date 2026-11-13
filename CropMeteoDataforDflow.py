# -*- coding: utf-8 -*-
"""
Crop/reproject every DCSM ERA5 forcing file required by Temperature=5.

Outputs (under OUTPUT_DIR):
  ERA5_child_mslp_u10n_v10n_chnk_rhoairalt_20121201_20130501.nc
  ERA5_child_d2m_t2m_tcc_20121201_20130501.nc
  ERA5_child_ssr_strd_20121201_20130501.nc
  ERA5_child_mer_mtpr_20121201_20130501.nc

The source variable names and attributes are preserved. Geographic wind
components are rotated into the EPSG:28992 projected x/y directions.

Compatible with Python 3.8+.

Install dependencies in the environment used to run this script:
  conda install -c conda-forge numpy pandas xarray netcdf4 scipy pyproj
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Geod, Transformer
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

SOURCE_DIR = Path(
    r"D:"
)

FLOW_GRID_FILE = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\Nested_grid\flowgrid.nc"
)

OUTPUT_DIR = Path(
    r"C:\dev\Python\DCSMmodelToNestedModel\MeteoFile_child\meteo"
)

START_TIME = pd.Timestamp("2012-12-01 00:00:00")
END_TIME = pd.Timestamp("2013-05-01 00:00:00")

SOURCE_CRS = CRS.from_epsg(4326)
TARGET_CRS = CRS.from_epsg(28992)

# The script estimates a suitable spacing, bounded by these values.
MINIMUM_TARGET_SPACING_M = 5_000.0
MAXIMUM_TARGET_SPACING_M = 20_000.0
MANUAL_TARGET_SPACING_M = 10_000.0
AUTOMATIC_GRID_SPACING = True
BUFFER_M = 20_000.0
MAXIMUM_MESH_SAMPLE_POINTS = 100_000

SOURCE_TIME_NAME = "time"
SOURCE_X_DIM = "x"
SOURCE_Y_DIM = "y"
SOURCE_LON_NAME = "X"
SOURCE_LAT_NAME = "Y"

COMPRESSION_LEVEL = 4
FILL_VALUE = np.float32(9.96921e36)


JOBS = [
    {
        "label": "pressure, wind, Charnock and air density",
        "source_token": "mslp_u10n_v10n_chnk_rhoairalt",
        "output_name": (
            "ERA5_child_mslp_u10n_v10n_chnk_rhoairalt_"
            "20121201_20130501.nc"
        ),
        "rotate_wind": True,
        "required_groups": {
            "pressure": ["air_pressure_fixed_height", "air_pressure", "mslp"],
            "eastward wind": ["eastward_wind", "u10n", "u10"],
            "northward wind": ["northward_wind", "v10n", "v10"],
            "Charnock": ["C", "charnock", "chnk"],
        },
    },
    {
        "label": "dew point, air temperature and cloudiness",
        "source_token": "d2m_t2m_tcc",
        "output_name": "ERA5_child_d2m_t2m_tcc_20121201_20130501.nc",
        "rotate_wind": False,
        "required_groups": {
            "dew point": ["dew_point_temperature", "dewpoint_temperature", "d2m"],
            "air temperature": ["air_temperature", "t2m"],
            "cloudiness": ["cloud_area_fraction", "cloudiness", "tcc"],
        },
    },
    {
        "label": "shortwave and longwave radiation",
        "source_token": "ssr_strd",
        "output_name": "ERA5_child_ssr_strd_20121201_20130501.nc",
        "rotate_wind": False,
        "required_groups": {
            "solar radiation": ["ssr"],
            "longwave radiation": ["strd"],
        },
    },
    {
        "label": "evaporation and precipitation",
        "source_token": "mer_mtpr",
        "output_name": "ERA5_child_mer_mtpr_20121201_20130501.nc",
        "rotate_wind": False,
        "required_groups": {
            "evaporation": ["mer"],
            "precipitation": ["mtpr"],
        },
    },
]


FLOW_FACE_X_CANDIDATES = [
    "mesh2d_face_x", "mesh2d_flowelem_xcc", "FlowElem_xcc",
    "flowelem_xcc", "mesh2d_face_x_coordinate",
]
FLOW_FACE_Y_CANDIDATES = [
    "mesh2d_face_y", "mesh2d_flowelem_ycc", "FlowElem_ycc",
    "flowelem_ycc", "mesh2d_face_y_coordinate",
]
FLOW_NODE_X_CANDIDATES = ["mesh2d_node_x", "NetNode_x", "netnode_x"]
FLOW_NODE_Y_CANDIDATES = ["mesh2d_node_y", "NetNode_y", "netnode_y"]


def first_existing(dataset: xr.Dataset, candidates: List[str]) -> str:
    for name in candidates:
        if name in dataset.variables or name in dataset.coords:
            return name
    raise KeyError(
        "None of {0} exists. Available names: {1}".format(
            candidates, sorted(dataset.variables)
        )
    )


def resolve_source_file(token: str) -> Path:
    matches = sorted(SOURCE_DIR.glob("*{0}*.nc".format(token)))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one NetCDF containing {0!r} in {1}; found {2}: {3}"
            .format(token, SOURCE_DIR, len(matches), [p.name for p in matches])
        )
    return matches[0]


def read_flow_points() -> Tuple[np.ndarray, np.ndarray]:
    if not FLOW_GRID_FILE.is_file():
        raise FileNotFoundError(FLOW_GRID_FILE)
    with xr.open_dataset(FLOW_GRID_FILE, decode_times=False, engine="netcdf4") as ds:
        try:
            xn = first_existing(ds, FLOW_FACE_X_CANDIDATES)
            yn = first_existing(ds, FLOW_FACE_Y_CANDIDATES)
        except KeyError:
            xn = first_existing(ds, FLOW_NODE_X_CANDIDATES)
            yn = first_existing(ds, FLOW_NODE_Y_CANDIDATES)
        x = np.asarray(ds[xn].values, float).reshape(-1)
        y = np.asarray(ds[yn].values, float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise ValueError("Nested grid has fewer than two valid coordinates")
    return x[valid], y[valid]


def estimate_mesh_spacing(x: np.ndarray, y: np.ndarray) -> float:
    points = np.column_stack([x, y])
    if len(points) > MAXIMUM_MESH_SAMPLE_POINTS:
        ii = np.linspace(0, len(points) - 1, MAXIMUM_MESH_SAMPLE_POINTS, dtype=int)
        points = points[ii]
    distance, _ = cKDTree(points).query(points, k=2)
    nearest = distance[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest))


def estimate_era5_spacing(ds: xr.Dataset, centre_x: float, centre_y: float) -> float:
    lon = np.asarray(ds[SOURCE_LON_NAME].values, float)
    lat = np.asarray(ds[SOURCE_LAT_NAME].values, float)
    to_ll = Transformer.from_crs(TARGET_CRS, SOURCE_CRS, always_xy=True)
    lon0, lat0 = to_ll.transform(centre_x, centre_y)
    ix = max(0, min(int(np.argmin(abs(lon - lon0))), len(lon) - 2))
    iy = max(0, min(int(np.argmin(abs(lat - lat0))), len(lat) - 2))
    geod = Geod(ellps="WGS84")
    _, _, dx = geod.inv(lon[ix], lat[iy], lon[ix + 1], lat[iy])
    _, _, dy = geod.inv(lon[ix], lat[iy], lon[ix], lat[iy + 1])
    return min(abs(float(dx)), abs(float(dy)))


def nice_spacing(value: float) -> float:
    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    normalized = value / scale
    for candidate in [1.0, 2.0, 2.5, 5.0, 10.0]:
        if normalized <= candidate:
            return float(candidate * scale)
    return float(10.0 * scale)


def build_target_grid(reference: xr.Dataset):
    gx, gy = read_flow_points()
    xmin, xmax = float(gx.min()), float(gx.max())
    ymin, ymax = float(gy.min()), float(gy.max())
    if AUTOMATIC_GRID_SPACING:
        mesh_spacing = estimate_mesh_spacing(gx, gy)
        era5_spacing = estimate_era5_spacing(
            reference, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        )
        raw = min(2.0 * mesh_spacing, 0.5 * era5_spacing)
        spacing = nice_spacing(
            min(MAXIMUM_TARGET_SPACING_M, max(MINIMUM_TARGET_SPACING_M, raw))
        )
        spacing = min(spacing, MAXIMUM_TARGET_SPACING_M)
    else:
        spacing = float(MANUAL_TARGET_SPACING_M)
    buffer_used = max(BUFFER_M, 2.0 * spacing)
    xmin = np.floor((xmin - buffer_used) / spacing) * spacing
    xmax = np.ceil((xmax + buffer_used) / spacing) * spacing
    ymin = np.floor((ymin - buffer_used) / spacing) * spacing
    ymax = np.ceil((ymax + buffer_used) / spacing) * spacing
    x = np.arange(xmin, xmax + 0.5 * spacing, spacing)
    y = np.arange(ymin, ymax + 0.5 * spacing, spacing)
    print("Projected target grid: {0} x {1}, spacing={2:.0f} m".format(
        len(x), len(y), spacing
    ))
    return x, y, spacing, buffer_used


def prepare_variable(source: xr.Dataset, name: str) -> xr.DataArray:
    da = source[name].assign_coords(
        longitude=(SOURCE_X_DIM, np.asarray(source[SOURCE_LON_NAME].values, float)),
        latitude=(SOURCE_Y_DIM, np.asarray(source[SOURCE_LAT_NAME].values, float)),
    ).swap_dims({SOURCE_X_DIM: "longitude", SOURCE_Y_DIM: "latitude"})
    if np.any(np.diff(da.longitude.values) < 0):
        da = da.sortby("longitude")
    if np.any(np.diff(da.latitude.values) < 0):
        da = da.sortby("latitude")
    for dim in list(da.dims):
        if dim not in {SOURCE_TIME_NAME, "longitude", "latitude"}:
            if da.sizes[dim] != 1:
                raise ValueError(
                    "{0} has unsupported dimension {1}={2}".format(
                        name, dim, da.sizes[dim]
                    )
                )
            da = da.isel({dim: 0}, drop=True)
    expected = {SOURCE_TIME_NAME, "longitude", "latitude"}
    if set(da.dims) != expected:
        raise ValueError("Unexpected dimensions for {0}: {1}".format(name, da.dims))
    return da.transpose(SOURCE_TIME_NAME, "latitude", "longitude")


def select_bracketed(da: xr.DataArray) -> xr.DataArray:
    times = pd.DatetimeIndex(pd.to_datetime(da[SOURCE_TIME_NAME].values))
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Source time must be increasing and unique")
    left = int(times.searchsorted(START_TIME, side="left"))
    right = int(times.searchsorted(END_TIME, side="right"))
    if left >= len(times) or times[left] != START_TIME:
        left -= 1
    if not (right > 0 and times[right - 1] == END_TIME):
        right += 1
    left, right = max(0, left), min(len(times), right)
    selected = da.isel({SOURCE_TIME_NAME: slice(left, right)})
    out_times = pd.DatetimeIndex(pd.to_datetime(selected[SOURCE_TIME_NAME].values))
    if len(out_times) == 0 or out_times[0] > START_TIME or out_times[-1] < END_TIME:
        raise ValueError(
            "{0} does not bracket {1} through {2}".format(
                da.name, START_TIME, END_TIME
            )
        )
    return selected


def projected_basis(lon, lat, xx, yy):
    geod = Geod(ellps="WGS84")
    to_xy = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    distance = np.full(lon.shape, 100.0)
    lon_e, lat_e, _ = geod.fwd(lon, lat, np.full(lon.shape, 90.0), distance)
    lon_n, lat_n, _ = geod.fwd(lon, lat, np.full(lon.shape, 0.0), distance)
    xe, ye = to_xy.transform(lon_e, lat_e)
    xn, yn = to_xy.transform(lon_n, lat_n)
    ex, ey = np.asarray(xe) - xx, np.asarray(ye) - yy
    nx, ny = np.asarray(xn) - xx, np.asarray(yn) - yy
    el, nl = np.hypot(ex, ey), np.hypot(nx, ny)
    return ex / el, ey / el, nx / nl, ny / nl


def interpolate(da: xr.DataArray, lon, lat) -> np.ndarray:
    lon_da = xr.DataArray(lon, dims=("y", "x"))
    lat_da = xr.DataArray(lat, dims=("y", "x"))
    out = da.interp(longitude=lon_da, latitude=lat_da, method="linear").load()
    values = np.asarray(out.values, float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Interpolation of {0} produced non-finite values. Increase source "
            "coverage or reduce BUFFER_M.".format(da.name)
        )
    return values


def make_crs_variable() -> xr.DataArray:
    attrs = TARGET_CRS.to_cf()
    attrs.update({
        "spatial_ref": TARGET_CRS.to_wkt(),
        "crs_wkt": TARGET_CRS.to_wkt(),
        "epsg_code": "EPSG:28992",
        "EPSG_code": "EPSG:28992",
    })
    return xr.DataArray(np.int32(28992), attrs=attrs)


def find_group_name(ds: xr.Dataset, candidates: List[str], label: str) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(
        "Missing {0}. Tried {1}; data variables are {2}".format(
            label, candidates, list(ds.data_vars)
        )
    )


def spatial_time_variables(ds: xr.Dataset) -> List[str]:
    result = []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if SOURCE_TIME_NAME in dims and SOURCE_X_DIM in dims and SOURCE_Y_DIM in dims:
            result.append(name)
    return result


def convert_one(job, source_path, x, y, lon, lat, basis):
    print("\n--- {0}".format(job["label"]))
    print("Source: {0}".format(source_path))
    output_path = OUTPUT_DIR / job["output_name"]

    with xr.open_dataset(source_path, decode_times=True, engine="netcdf4") as source:
        selected_names = {}
        for label, candidates in job["required_groups"].items():
            selected_names[label] = find_group_name(source, candidates, label)
        names = spatial_time_variables(source)
        if not names:
            raise ValueError("No time/y/x variables found in {0}".format(source_path))

        arrays = {}
        attrs = {}
        output_times = None
        for name in names:
            print("  Interpolating {0}".format(name))
            da = select_bracketed(prepare_variable(source, name))
            times = pd.DatetimeIndex(pd.to_datetime(da[SOURCE_TIME_NAME].values))
            if output_times is None:
                output_times = times
            elif not output_times.equals(times):
                raise ValueError("Time coordinates differ within {0}".format(source_path))
            arrays[name] = interpolate(da, lon, lat)
            attrs[name] = dict(source[name].attrs)

        if job["rotate_wind"]:
            uname = selected_names["eastward wind"]
            vname = selected_names["northward wind"]
            east_x, east_y, north_x, north_y = basis
            u0, v0 = arrays[uname].copy(), arrays[vname].copy()
            arrays[uname] = u0 * east_x[None, :, :] + v0 * north_x[None, :, :]
            arrays[vname] = u0 * east_y[None, :, :] + v0 * north_y[None, :, :]
            error = np.max(np.abs(np.hypot(u0, v0) - np.hypot(arrays[uname], arrays[vname])))
            if error > 1.0e-4:
                raise RuntimeError("Wind rotation changed speed by {0}".format(error))
            attrs[uname]["long_name"] = "wind component along projected x axis"
            attrs[vname]["long_name"] = "wind component along projected y axis"

        # Preserve pressure name but ensure Pa if the source is hPa/mbar.
        if "pressure" in selected_names:
            pname = selected_names["pressure"]
            unit = str(attrs[pname].get("units", "")).lower().strip()
            if unit in {"hpa", "mbar", "millibar", "millibars"}:
                arrays[pname] *= 100.0
                attrs[pname]["units"] = "Pa"
            elif unit not in {"pa", "pascal", "pascals"}:
                raise ValueError("Unsupported pressure unit {0!r}".format(unit))

    coords = {
        "time": ("time", output_times.values.astype("datetime64[ns]"), {
            "standard_name": "time", "axis": "T"
        }),
        "x": ("x", x, {
            "standard_name": "projection_x_coordinate", "units": "m", "axis": "X"
        }),
        "y": ("y", y, {
            "standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"
        }),
    }
    data_vars = {"spatial_ref": make_crs_variable()}
    for name, values in arrays.items():
        var_attrs = attrs[name]
        var_attrs["coordinates"] = "y x"
        var_attrs["grid_mapping"] = "spatial_ref"
        data_vars[name] = (("time", "y", "x"), values.astype(np.float32), var_attrs)

    out = xr.Dataset(coords=coords, data_vars=data_vars, attrs={
        "Conventions": "CF-1.8",
        "title": "DCSM ERA5 forcing cropped/reprojected for nested D-Flow FM",
        "source": str(source_path),
        "target_crs": "EPSG:28992",
        "requested_start": str(START_TIME),
        "requested_end": str(END_TIME),
    })
    time_units = "seconds since {0} +00:00".format(
        output_times[0].strftime("%Y-%m-%d %H:%M:%S")
    )
    encoding = {
        "time": {"dtype": "float64", "units": time_units, "calendar": "gregorian"},
        "x": {"dtype": "float64", "_FillValue": None},
        "y": {"dtype": "float64", "_FillValue": None},
        "spatial_ref": {"dtype": "int32", "_FillValue": None},
    }
    for name in arrays:
        encoding[name] = {
            "dtype": "float32", "zlib": True, "complevel": COMPRESSION_LEVEL,
            "shuffle": True, "_FillValue": FILL_VALUE,
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    out.close()

    with xr.open_dataset(output_path, decode_times=True, engine="netcdf4") as check:
        for name in arrays:
            if name not in check:
                raise RuntimeError("Output is missing {0}".format(name))
        print("Written: {0}".format(output_path))
        print("Variables: {0}".format(list(check.data_vars)))
        print("Time: {0} through {1}".format(
            pd.Timestamp(check.time.values[0]), pd.Timestamp(check.time.values[-1])
        ))


def write_ext_file():
    ext = """QUANTITY     = airpressure_windx_windy_charnock
FILENAME     = meteo/ERA5_child_mslp_u10n_v10n_chnk_rhoairalt_20121201_20130501.nc
FILETYPE     = 11
METHOD       = 3
OPERAND      = O

QUANTITY     = dewpoint_airtemperature_cloudiness
FILENAME     = meteo/ERA5_child_d2m_t2m_tcc_20121201_20130501.nc
FILETYPE     = 11
METHOD       = 3
OPERAND      = O

QUANTITY     = solarradiation
FILENAME     = meteo/ERA5_child_ssr_strd_20121201_20130501.nc
VARNAME      = ssr
FILETYPE     = 11
METHOD       = 3
OPERAND      = O

QUANTITY     = longwaveradiation
FILENAME     = meteo/ERA5_child_ssr_strd_20121201_20130501.nc
VARNAME      = strd
FILETYPE     = 11
METHOD       = 3
OPERAND      = O

QUANTITY     = rainfall_rate
FILENAME     = meteo/ERA5_child_mer_mtpr_20121201_20130501.nc
VARNAME      = mtpr
FILETYPE     = 11
METHOD       = 3
OPERAND      = O

QUANTITY     = rainfall_rate
FILENAME     = meteo/ERA5_child_mer_mtpr_20121201_20130501.nc
VARNAME      = mer
FILETYPE     = 11
METHOD       = 3
OPERAND      = +
"""
    ext_path = OUTPUT_DIR.parent / "FlowFM_meteo.ext"
    with ext_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(ext)
    print("Written EXT file: {0}".format(ext_path))


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)
    sources = [(job, resolve_source_file(job["source_token"])) for job in JOBS]
    with xr.open_dataset(sources[0][1], decode_times=True, engine="netcdf4") as reference:
        x, y, spacing, buffer_used = build_target_grid(reference)
    xx, yy = np.meshgrid(x, y)
    to_ll = Transformer.from_crs(TARGET_CRS, SOURCE_CRS, always_xy=True)
    lon, lat = to_ll.transform(xx, yy)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    basis = projected_basis(lon, lat, xx, yy)

    print("Requested period: {0} through {1}".format(START_TIME, END_TIME))
    print("Output directory: {0}".format(OUTPUT_DIR))
    for job, source_path in sources:
        convert_one(job, source_path, x, y, lon, lat, basis)
    write_ext_file()
    print("\nAll four forcing files were created successfully.")


if __name__ == "__main__":
    main()
