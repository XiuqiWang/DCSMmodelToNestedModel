# -*- coding: utf-8 -*-
"""
Extract water level and ux,uy advection velocity at the observation points from the DCSM model to be used as boundary conditions for the child model. The ux,uy are displayed at each sigma layer in the child flow model as 3d conditions. 

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ==================================================================
# USER SETTINGS
# ==================================================================

nc_path = Path("DCSM-FM_0_5nm_restarted_his.nc")

out_wl = Path("nested_waterlevel.bc")
out_uv = Path("nested_uxuyadvectionvelocity.bc")

time_origin = pd.Timestamp("2013-12-05 00:00:00")
time_unit = f"seconds since {time_origin:%Y-%m-%d %H:%M:%S}"

# Five child sigma-layer centres, ordered BED -> SURFACE.
#
# sigma = -1 at the bed
# sigma =  0 at the free surface
#
# These correspond to 10, 30, 50, 70, 90 percent upward from the bed.
sigma_centres = np.array(
    [-0.9, -0.7, -0.5, -0.3, -0.1],
    dtype=float,
)

vertical_positions_percent_from_bed = np.array(
    [10.0, 30.0, 50.0, 70.0, 90.0],
    dtype=float,
)

# Replace this placeholder with the actual CHILD-model bed elevation at
# every boundary support point, in the same vertical datum used below.
default_child_bedlevel = -10.0

# Example datum correction:
# parent MSL2013 elevation + 1.95 m -> NAP
#
# Confirm this correction independently before production use.
waterlevel_offset_parent_to_child_datum = 1.95


# ==================================================================
# HELPERS
# ==================================================================

def decode_station_names(ds: xr.Dataset) -> list[str]:
    """Return clean observation-station names."""
    if "station_name" not in ds:
        raise KeyError("Dataset does not contain 'station_name'.")

    names: list[str] = []

    for value in ds["station_name"].values:
        if isinstance(value, bytes):
            name = value.decode("utf-8").strip()
        else:
            name = str(value).strip()

        if not name:
            raise ValueError("An empty station name was found.")

        names.append(name)

    if len(names) != len(set(names)):
        duplicates = sorted(
            {name for name in names if names.count(name) > 1}
        )
        raise ValueError(
            f"Station names are not unique: {duplicates}"
        )

    return names


def require_variable(
    ds: xr.Dataset,
    variable: str,
    required_dims: tuple[str, ...],
) -> xr.DataArray:
    """Validate that a required variable and dimensions exist."""
    if variable not in ds:
        raise KeyError(
            f"Required variable {variable!r} is absent from the dataset."
        )

    da = ds[variable]

    missing_dims = [
        dim for dim in required_dims
        if dim not in da.dims
    ]

    if missing_dims:
        raise ValueError(
            f"{variable!r} is missing dimensions {missing_dims}. "
            f"Found dimensions: {da.dims}"
        )

    return da


def validate_velocity_metadata(ds: xr.Dataset) -> None:
    """
    Confirm that x_velocity/y_velocity are horizontal x/y components.
    """
    expected = {
        "x_velocity": (
            "x-component",
            "sea_water_x_velocity",
        ),
        "y_velocity": (
            "y-component",
            "sea_water_y_velocity",
        ),
    }

    for variable, expected_terms in expected.items():
        if variable not in ds:
            raise KeyError(f"{variable!r} is not in the dataset.")

        da = ds[variable]

        description = " ".join(
            str(da.attrs.get(key, ""))
            for key in (
                "long_name",
                "standard_name",
                "description",
            )
        ).lower()

        if not any(term.lower() in description for term in expected_terms):
            raise ValueError(
                f"The metadata for {variable!r} does not clearly identify "
                "it as the expected horizontal component.\n"
                f"Attributes: {dict(da.attrs)}"
            )

        units = str(da.attrs.get("units", "")).replace(" ", "").lower()

        if units not in {"ms-1", "m/s", "ms^-1"}:
            raise ValueError(
                f"Unexpected units for {variable!r}: "
                f"{da.attrs.get('units')!r}"
            )


def interp_profile(
    z_parent: np.ndarray,
    value_parent: np.ndarray,
    z_target: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one parent vertical profile to target elevations.

    Values outside the valid parent range are held at the nearest valid
    endpoint. Change this policy if extrapolation is undesirable.
    """
    z_parent = np.asarray(z_parent, dtype=float)
    value_parent = np.asarray(value_parent, dtype=float)
    z_target = np.asarray(z_target, dtype=float)

    valid = (
        np.isfinite(z_parent)
        & np.isfinite(value_parent)
    )

    if valid.sum() < 2:
        return np.full(
            z_target.shape,
            np.nan,
            dtype=float,
        )

    z = z_parent[valid]
    values = value_parent[valid]

    order = np.argsort(z)
    z = z[order]
    values = values[order]

    # Remove duplicate vertical coordinates because np.interp expects
    # increasing coordinate values.
    z_unique, unique_indices = np.unique(
        z,
        return_index=True,
    )
    values_unique = values[unique_indices]

    if z_unique.size < 2:
        return np.full(
            z_target.shape,
            np.nan,
            dtype=float,
        )

    return np.interp(
        z_target,
        z_unique,
        values_unique,
        left=values_unique[0],
        right=values_unique[-1],
    )


def write_waterlevel_header(
    file_object,
    station: str,
) -> None:
    """Write one D-Flow FM water-level forcing header."""
    file_object.write("[Forcing]\n")
    file_object.write(f"name              = {station}\n")
    file_object.write("function          = timeseries\n")
    file_object.write("timeInterpolation = linear\n")
    file_object.write("quantity          = time\n")
    file_object.write(f"unit              = {time_unit}\n")
    file_object.write("quantity          = waterlevelbnd\n")
    file_object.write("unit              = m\n")


def write_uxuy_t3d_header(
    file_object,
    station: str,
    vertical_positions: np.ndarray,
) -> None:
    """
    Write a T3D ux/uy advection-velocity forcing header.

    Data-column order:
        time,
        ux(z1), uy(z1),
        ux(z2), uy(z2),
        ...
    """
    positions_text = " ".join(
        f"{position:g}"
        for position in vertical_positions
    )

    file_object.write("[Forcing]\n")
    file_object.write(f"name              = {station}\n")
    file_object.write("function          = t3d\n")
    file_object.write("timeInterpolation = linear\n")
    file_object.write("vertPositionType  = percBed\n")
    file_object.write(
        f"vertPositions     = {positions_text}\n"
    )
    file_object.write("vertInterpolation = linear\n")

    file_object.write("quantity          = time\n")
    file_object.write(f"unit              = {time_unit}\n")

    file_object.write(
        "vector            = "
        "uxuyadvectionvelocitybnd:ux,uy\n"
    )

    for vertical_index in range(1, len(vertical_positions) + 1):
        file_object.write("quantity          = ux\n")
        file_object.write("unit              = m/s\n")
        file_object.write(
            f"vertPositionIndex = {vertical_index}\n"
        )

        file_object.write("quantity          = uy\n")
        file_object.write("unit              = m/s\n")
        file_object.write(
            f"vertPositionIndex = {vertical_index}\n"
        )


def format_time_seconds(value: float) -> str:
    """Format elapsed seconds without introducing round-off noise."""
    rounded = round(float(value))

    if not np.isclose(value, rounded, atol=1e-6):
        raise ValueError(
            f"Time value {value} is not an integer number of seconds."
        )

    return f"{rounded:12d}"


# ==================================================================
# OPEN AND VALIDATE THE PARENT HISTORY FILE
# ==================================================================

with xr.open_dataset(nc_path) as ds:

    station_names = decode_station_names(ds)
    validate_velocity_metadata(ds)

    waterlevel_da = require_variable(
        ds,
        "waterlevel",
        ("time", "station"),
    )
    ux_da = require_variable(
        ds,
        "x_velocity",
        ("time", "station", "laydim"),
    )
    uy_da = require_variable(
        ds,
        "y_velocity",
        ("time", "station", "laydim"),
    )
    z_da = require_variable(
        ds,
        "zcoordinate_c",
        ("time", "station", "laydim"),
    )

    # Explicitly arrange all 3D arrays in one known dimension order.
    ux_da = ux_da.transpose("time", "station", "laydim")
    uy_da = uy_da.transpose("time", "station", "laydim")
    z_da = z_da.transpose("time", "station", "laydim")

    waterlevel_da = waterlevel_da.transpose(
        "time",
        "station",
    )

    time_values = pd.DatetimeIndex(
        pd.to_datetime(ds["time"].values)
    )

    time_seconds = (
        (time_values - time_origin)
        / pd.Timedelta(seconds=1)
    ).to_numpy(dtype=float)

    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError(
            "Dataset times must be strictly increasing."
        )

    if time_seconds[0] < 0:
        raise ValueError(
            f"The first input time {time_values[0]} is earlier than "
            f"the BC reference time {time_origin}."
        )

    # Actual CHILD-model bed elevation at each support point.
    # Replace this dictionary with values read from the child grid.
    child_bedlevel = {
        station: default_child_bedlevel
        for station in station_names
    }

    n_times = len(time_seconds)
    n_layers = len(sigma_centres)

    with out_wl.open("w", encoding="utf-8", newline="\n") as f_wl, \
         out_uv.open("w", encoding="utf-8", newline="\n") as f_uv:

        for station_index, station in enumerate(station_names):

            waterlevel_parent = (
                waterlevel_da
                .isel(station=station_index)
                .values
                .astype(float)
            )

            ux_parent = (
                ux_da
                .isel(station=station_index)
                .values
                .astype(float)
            )

            uy_parent = (
                uy_da
                .isel(station=station_index)
                .values
                .astype(float)
            )

            z_parent = (
                z_da
                .isel(station=station_index)
                .values
                .astype(float)
            )

            if ux_parent.shape != uy_parent.shape:
                raise ValueError(
                    f"ux/uy shapes differ at {station}: "
                    f"{ux_parent.shape} versus {uy_parent.shape}"
                )

            if ux_parent.shape != z_parent.shape:
                raise ValueError(
                    f"Velocity and vertical-coordinate shapes differ "
                    f"at {station}."
                )

            if ux_parent.shape[0] != n_times:
                raise ValueError(
                    f"Unexpected number of times at {station}."
                )

            # ------------------------------------------------------
            # Convert all elevations into the child vertical datum
            # ------------------------------------------------------
            datum_offset = waterlevel_offset_parent_to_child_datum

            waterlevel_child_datum = waterlevel_parent + datum_offset
            z_parent_child_datum = z_parent + datum_offset
            zb_child = float(child_bedlevel[station])

            # ------------------------------------------------------
            # Interpolate eastward/northward parent velocities to
            # five child sigma-layer centres
            # ------------------------------------------------------
            ux_sigma = np.full(
                (n_times, n_layers),
                np.nan,
                dtype=float,
            )

            uy_sigma = np.full(
                (n_times, n_layers),
                np.nan,
                dtype=float,
            )

            for time_index in range(n_times):
                eta = waterlevel_child_datum[time_index]

                if not np.isfinite(eta):
                    continue

                water_depth = eta - zb_child

                if water_depth <= 0.0:
                    continue

                z_sigma = eta + sigma_centres * water_depth

                ux_sigma[time_index, :] = interp_profile(
                    z_parent_child_datum[time_index, :],
                    ux_parent[time_index, :],
                    z_sigma,
                )

                uy_sigma[time_index, :] = interp_profile(
                    z_parent_child_datum[time_index, :],
                    uy_parent[time_index, :],
                    z_sigma,
                )

            if np.isnan(ux_sigma).any() or np.isnan(uy_sigma).any():
                bad_count = int(
                    np.isnan(ux_sigma).sum()
                    + np.isnan(uy_sigma).sum()
                )

                raise ValueError(
                    f"{station}: {bad_count} interpolated velocity "
                    "values are NaN. Check wet/dry conditions, parent "
                    "profiles and child bed elevation."
                )

            # ------------------------------------------------------
            # Write water-level forcing
            # ------------------------------------------------------
            write_waterlevel_header(f_wl, station)

            for time_value, waterlevel_value in zip(
                time_seconds,
                waterlevel_child_datum,
            ):
                f_wl.write(
                    f"{format_time_seconds(time_value)}"
                    f"{waterlevel_value:14.5f}\n"
                )

            f_wl.write("\n")

            # ------------------------------------------------------
            # Write T3D ux/uy advection-velocity forcing
            # ------------------------------------------------------
            write_uxuy_t3d_header(
                f_uv,
                station,
                vertical_positions_percent_from_bed,
            )

            for time_index, time_value in enumerate(time_seconds):
                row_values = []

                for layer_index in range(n_layers):
                    row_values.append(
                        ux_sigma[time_index, layer_index]
                    )
                    row_values.append(
                        uy_sigma[time_index, layer_index]
                    )

                f_uv.write(format_time_seconds(time_value))

                for value in row_values:
                    f_uv.write(f"{value:14.5f}")

                f_uv.write("\n")

            f_uv.write("\n")


print("Finished writing:")
print(f"  {out_wl.resolve()}")
print(f"  {out_uv.resolve()}")

# import xarray as xr
# import numpy as np
# import pandas as pd
# from pathlib import Path

# nc_path = r"DCSM-FM_0_5nm_restarted_his.nc"

# out_wl = "nested_waterlevel.bc"
# out_un = "nested_normalvelocity.bc"
# out_ut = "nested_tangentialvelocity.bc"

# ds = xr.open_dataset(nc_path)

# station_names = [
#     name.decode("utf-8").strip() if isinstance(name, bytes) else str(name).strip()
#     for name in ds["station_name"].values
# ]

# # ------------------------------------------------------------------
# # Time: seconds since target origin
# # ------------------------------------------------------------------
# time_values = pd.to_datetime(ds["time"].values)

# time_origin = pd.Timestamp("2013-12-05 00:00:00")
# time_seconds = (time_values - time_origin) / pd.Timedelta(seconds=1)
# time_unit = "seconds since 2013-12-05 00:00:00"

# # ------------------------------------------------------------------
# # 5 sigma-layer centres, surface to bed
# # ------------------------------------------------------------------
# sigma_centres = np.array([-0.1, -0.3, -0.5, -0.7, -0.9])
# n_layers = len(sigma_centres)

# # TODO: replace by actual child-model bed levels at each boundary point
# bedlevel = {station: -10.0 for station in station_names}

# # Datum correction if needed
# waterlevel_offset_msl_to_nap = 1.95  # m, replace by your estimated value

# MSL2013_TO_NAP = {
#     station: waterlevel_offset_msl_to_nap
#     for station in station_names
# }

# def get_station_xy(ds):
#     """Try common coordinate variable names in D-Flow FM history files."""
#     x_candidates = ["station_x_coordinate", "station_x", "x_station", "station_xcoord"]
#     y_candidates = ["station_y_coordinate", "station_y", "y_station", "station_ycoord"]

#     xname = next((v for v in x_candidates if v in ds.variables), None)
#     yname = next((v for v in y_candidates if v in ds.variables), None)

#     if xname is None or yname is None:
#         raise KeyError(
#             "Could not find station x/y coordinates in his.nc. "
#             "Check ds variables and update x_candidates/y_candidates."
#         )

#     return ds[xname].values, ds[yname].values


# def boundary_vectors(x, y):
#     """
#     Compute tangential and normal unit vectors along ordered boundary support points.

#     Tangential direction follows station order:
#         offshore_0001 -> offshore_0002 -> ...

#     Normal direction is tangent rotated clockwise:
#         n = (ty, -tx)

#     If normal velocity sign is opposite of what you expect, multiply normal by -1.
#     """
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)

#     tx = np.zeros_like(x)
#     ty = np.zeros_like(y)

#     for i in range(len(x)):
#         if i == 0:
#             dx = x[i + 1] - x[i]
#             dy = y[i + 1] - y[i]
#         elif i == len(x) - 1:
#             dx = x[i] - x[i - 1]
#             dy = y[i] - y[i - 1]
#         else:
#             dx = x[i + 1] - x[i - 1]
#             dy = y[i + 1] - y[i - 1]

#         length = np.hypot(dx, dy)
#         tx[i] = dx / length
#         ty[i] = dy / length

#     # Clockwise normal
#     nx = ty.copy()
#     ny = -tx.copy()

#     return tx, ty, nx, ny


# def interp_profile(z_parent, val_parent, z_target):
#     """Interpolate one vertical profile, ignoring NaNs."""
#     mask = np.isfinite(z_parent) & np.isfinite(val_parent)

#     if np.sum(mask) < 2:
#         return np.full_like(z_target, np.nan, dtype=float)

#     z = z_parent[mask]
#     v = val_parent[mask]

#     order = np.argsort(z)
#     z = z[order]
#     v = v[order]

#     return np.interp(z_target, z, v, left=v[0], right=v[-1])


# def write_header_waterlevel(f, station):
#     f.write("[forcing]\n")
#     f.write(f"name              = {station}\n")
#     f.write("function          = timeseries\n")
#     f.write("timeInterpolation = linear\n")
#     f.write("quantity          = time\n")
#     f.write(f"unit              = {time_unit}\n")
#     f.write("quantity          = waterlevelbnd\n")
#     f.write("unit              = m\n")


# def write_header_velocity(f, station, quantity):
#     f.write("[forcing]\n")
#     f.write(f"name              = {station}\n")
#     f.write("function          = timeseries\n")
#     f.write("timeInterpolation = linear\n")
#     f.write("quantity          = time\n")
#     f.write(f"unit              = {time_unit}\n")
#     f.write(f"quantity          = {quantity}\n")
#     f.write("unit              = m/s\n")
#     f.write("verticalPositionType = index\n")
#     f.write("verticalProfileType  = uniform\n")


# # ------------------------------------------------------------------
# # Get boundary directions from station coordinates
# # ------------------------------------------------------------------
# x_station, y_station = get_station_xy(ds)

# # ------------------------------------------------------------
# # Compute boundary directions separately for each boundary group
# # ------------------------------------------------------------
# boundary_groups = ["north", "south", "offshore"]

# tx_all = np.full(len(station_names), np.nan)
# ty_all = np.full(len(station_names), np.nan)
# nx_all = np.full(len(station_names), np.nan)
# ny_all = np.full(len(station_names), np.nan)

# for boundary in boundary_groups:

#     indices = [
#         i for i, name in enumerate(station_names)
#         if name.lower().startswith(boundary)
#     ]

#     if len(indices) == 0:
#         print(f"No stations found for boundary: {boundary}")
#         continue

#     x_b = x_station[indices]
#     y_b = y_station[indices]

#     tx_b, ty_b, nx_b, ny_b = boundary_vectors(x_b, y_b)

#     tx_all[indices] = tx_b
#     ty_all[indices] = ty_b
#     nx_all[indices] = nx_b
#     ny_all[indices] = ny_b

# for boundary in ["north", "south", "offshore"]:
#     print("\n", boundary)
#     for i, name in enumerate(station_names):
#         if name.lower().startswith(boundary):
#             print(
#                 name,
#                 "t=(", tx_all[i], ty_all[i], ")",
#                 "n=(", nx_all[i], ny_all[i], ")",
#             )

# with open(out_wl, "w") as f_wl, \
#      open(out_un, "w") as f_un, \
#      open(out_ut, "w") as f_ut:

#     for idx, station in enumerate(station_names):

#         wl_msl2013 = ds["waterlevel"].isel(station=idx).values
#         wl = wl_msl2013 + MSL2013_TO_NAP[station]

#         u_parent = ds["x_velocity"].isel(station=idx).values
#         v_parent = ds["y_velocity"].isel(station=idx).values
#         z_parent = ds["zcoordinate_c"].isel(station=idx).values

#         zb = bedlevel[station]

#         u_sigma = np.zeros((len(time_seconds), n_layers))
#         v_sigma = np.zeros((len(time_seconds), n_layers))

#         for it, eta in enumerate(wl):
#             z_sigma = eta + sigma_centres * (eta - zb)

#             u_sigma[it, :] = interp_profile(
#                 z_parent[it, :],
#                 u_parent[it, :],
#                 z_sigma
#             )

#             v_sigma[it, :] = interp_profile(
#                 z_parent[it, :],
#                 v_parent[it, :],
#                 z_sigma
#             )

#         # Project to normal and tangential components
#         normal_sigma = u_sigma * nx_all[idx] + v_sigma * ny_all[idx]
#         tangential_sigma = u_sigma * tx_all[idx] + v_sigma * ty_all[idx]

#         # ----------------------------------------------------------
#         # Water level file
#         # ----------------------------------------------------------
#         write_header_waterlevel(f_wl, station)

#         for t, value in zip(time_seconds, wl):
#             f_wl.write(f"{t:12.0f}\t{value:12.5f}\n")

#         f_wl.write("\n")

#         # ----------------------------------------------------------
#         # Normal velocity file
#         # ----------------------------------------------------------
#         write_header_velocity(f_un, station, "normalvelocitybnd")

#         for it, t in enumerate(time_seconds):
#             for k in range(n_layers):
#                 f_un.write(
#                     f"{t:12.0f}\t{k + 1:d}\t{normal_sigma[it, k]:12.5f}\n"
#                 )

#         f_un.write("\n")

#         # ----------------------------------------------------------
#         # Tangential velocity file
#         # ----------------------------------------------------------
#         write_header_velocity(f_ut, station, "tangentialvelocitybnd")

#         for it, t in enumerate(time_seconds):
#             for k in range(n_layers):
#                 f_ut.write(
#                     f"{t:12.0f}\t{k + 1:d}\t{tangential_sigma[it, k]:12.5f}\n"
#                 )

#         f_ut.write("\n")

# print(f"Finished writing:")
# print(f"  {out_wl}")
# print(f"  {out_un}")
# print(f"  {out_ut}")