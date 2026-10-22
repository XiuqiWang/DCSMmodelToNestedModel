# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:44:21 2026

@author: WangX3
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
flow_only_his = r"WLonly/FlowFM_his.nc"
flow_wave_his = r"flow_wave/FlowFM_his.nc"

production_start = pd.Timestamp("2013-12-06 00:00:00")
production_end   = pd.Timestamp("2013-12-06 03:00:00")

directions = ["north", "south", "offshore"]

# 1 = surface layer, 5 = bottom layer
plot_layer = 5

# If True, average over vertical layers instead of using plot_layer
use_depth_average = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def decode_names(values):
    return [
        v.decode("utf-8").strip() if isinstance(v, bytes) else str(v).strip()
        for v in values
    ]


def get_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"Could not find any of these variables: {candidates}")


def select_velocity(da, use_depth_average=True, layer=1):
    """
    Returns velocity as array with shape (time, station).

    Handles common D-Flow FM HIS velocity dimensions:
        (time, station)
        (time, station, laydim)
        (time, laydim, station)
    """
    dims = da.dims

    if "laydim" not in dims:
        return da.values

    if use_depth_average:
        return da.mean(dim="laydim", skipna=True).values

    # selected layer
    if "laydim" in dims:
        return da.isel(laydim=layer - 1).values

    raise ValueError(f"Unexpected dimensions: {dims}")


def read_his_velocity(path, production_start, production_end,
                      use_depth_average=True, layer=1):
    ds = xr.open_dataset(path)

    station_names = decode_names(ds["station_name"].values)

    time_all = pd.to_datetime(ds["time"].values)
    mask = (time_all >= production_start) & (time_all <= production_end)

    ds_sel = ds.isel(time=mask)
    time = pd.to_datetime(ds_sel["time"].values).to_numpy()

    u_da = get_var(ds_sel, ["x_velocity", "ux", "u"])
    v_da = get_var(ds_sel, ["y_velocity", "uy", "v"])

    u = select_velocity(
        u_da,
        use_depth_average=use_depth_average,
        layer=layer,
    )

    v = select_velocity(
        v_da,
        use_depth_average=use_depth_average,
        layer=layer,
    )

    speed = np.sqrt(u**2 + v**2)

    return ds, station_names, time, u, v, speed


def names_containing(names, keyword):
    keyword = keyword.lower()
    return [
        (i, name)
        for i, name in enumerate(names)
        if keyword in name.lower()
    ]


def common_station_indices(names_a, names_b):
    """
    Match stations by name between two HIS files.
    """
    b_lookup = {name: i for i, name in enumerate(names_b)}

    matches = []
    for i_a, name in enumerate(names_a):
        if name in b_lookup:
            matches.append((i_a, b_lookup[name], name))

    return matches


def plot_quantity_by_boundary(
    quantity_flow,
    quantity_wave,
    time_flow,
    time_wave,
    station_names_flow,
    station_names_wave,
    quantity_name,
    ylabel,
    directions,
):
    matches = common_station_indices(station_names_flow, station_names_wave)

    fig, axes = plt.subplots(
        len(directions),
        1,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, direction in zip(axes, directions):
        for i_flow, i_wave, name in matches:
            if direction.lower() not in name.lower():
                continue

            # keep normal transect points out of the south boundary subplot
            if direction == "south" and "normal" in name.lower():
                continue

            ax.plot(
                time_flow,
                quantity_flow[:, i_flow],
                "-",
                linewidth=2,
                label=f"flow-only: {name}",
            )

            ax.plot(
                time_wave,
                quantity_wave[:, i_wave],
                "--",
                linewidth=2,
                label=f"flow+wave: {name}",
            )

        ax.set_ylabel(ylabel)
        ax.set_title(f"{direction.capitalize()} boundary: {quantity_name}")
        ax.grid(True)
        ax.legend(fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


def plot_difference_by_boundary(
    quantity_flow,
    quantity_wave,
    time_flow,
    station_names_flow,
    station_names_wave,
    quantity_name,
    ylabel,
    directions,
):
    """
    Assumes the two runs have the same output times.
    If not, interpolate before using this.
    """
    matches = common_station_indices(station_names_flow, station_names_wave)

    fig, axes = plt.subplots(
        len(directions),
        1,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, direction in zip(axes, directions):
        for i_flow, i_wave, name in matches:
            if direction.lower() not in name.lower():
                continue

            if direction == "south" and "normal" in name.lower():
                continue

            diff = quantity_wave[:, i_wave] - quantity_flow[:, i_flow]

            ax.plot(
                time_flow,
                diff,
                linewidth=2,
                label=f"{name}",
            )

        ax.axhline(0, color="k", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{direction.capitalize()} boundary: "
            f"flow+wave minus flow-only {quantity_name}"
        )
        ax.grid(True)
        ax.legend(fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Read both HIS files
# ------------------------------------------------------------
ds_flow, station_flow, time_flow, u_flow, v_flow, speed_flow = read_his_velocity(
    flow_only_his,
    production_start,
    production_end,
    use_depth_average=use_depth_average,
    layer=plot_layer,
)

ds_wave, station_wave, time_wave, u_wave, v_wave, speed_wave = read_his_velocity(
    flow_wave_his,
    production_start,
    production_end,
    use_depth_average=use_depth_average,
    layer=plot_layer,
)


# ------------------------------------------------------------
# Check time consistency
# ------------------------------------------------------------
if len(time_flow) != len(time_wave) or not np.all(time_flow == time_wave):
    print("Warning: flow-only and flow+wave HIS files have different times.")
    print("This script can still plot both, but difference plots need interpolation.")
    do_difference_plots = False
else:
    do_difference_plots = True


# ------------------------------------------------------------
# Figure 1: velocity magnitude
# ------------------------------------------------------------
plot_quantity_by_boundary(
    speed_flow,
    speed_wave,
    time_flow,
    time_wave,
    station_flow,
    station_wave,
    quantity_name="velocity magnitude",
    ylabel="Velocity magnitude (m/s)",
    directions=directions,
)


# ------------------------------------------------------------
# Figure 2: x velocity
# ------------------------------------------------------------
plot_quantity_by_boundary(
    u_flow,
    u_wave,
    time_flow,
    time_wave,
    station_flow,
    station_wave,
    quantity_name="x velocity",
    ylabel="x velocity (m/s)",
    directions=directions,
)


# ------------------------------------------------------------
# Figure 3: y velocity
# ------------------------------------------------------------
plot_quantity_by_boundary(
    v_flow,
    v_wave,
    time_flow,
    time_wave,
    station_flow,
    station_wave,
    quantity_name="y velocity",
    ylabel="y velocity (m/s)",
    directions=directions,
)


# ------------------------------------------------------------
# Figure 4: velocity magnitude difference
# ------------------------------------------------------------
if do_difference_plots:
    plot_difference_by_boundary(
        speed_flow,
        speed_wave,
        time_flow,
        station_flow,
        station_wave,
        quantity_name="velocity magnitude",
        ylabel="Difference (m/s)",
        directions=directions,
    )


# ------------------------------------------------------------
# Optional: south-normal transect comparison
# ------------------------------------------------------------
normal_south_matches = [
    (i_flow, i_wave, name)
    for i_flow, i_wave, name in common_station_indices(station_flow, station_wave)
    if "observationpointnormal" in name.lower()
]

fig, ax = plt.subplots(figsize=(13, 5))

for i_flow, i_wave, name in normal_south_matches:
    ax.plot(
        time_flow,
        speed_flow[:, i_flow],
        "-",
        linewidth=2,
        label=f"flow-only: {name}",
    )

    ax.plot(
        time_wave,
        speed_wave[:, i_wave],
        "--",
        linewidth=2,
        label=f"flow+wave: {name}",
    )

ax.set_xlabel("Time")
ax.set_ylabel("Velocity magnitude (m/s)")
ax.set_title("South-normal transect: velocity magnitude")
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Print maximum differences
# ------------------------------------------------------------
if do_difference_plots:
    print("\nMaximum absolute difference in velocity magnitude:")
    for i_flow, i_wave, name in common_station_indices(station_flow, station_wave):
        diff = speed_wave[:, i_wave] - speed_flow[:, i_flow]
        print(
            f"{name:40s} "
            f"max |Δspeed| = {np.nanmax(np.abs(diff)):.4f} m/s"
        )


ds_flow.close()
ds_wave.close()