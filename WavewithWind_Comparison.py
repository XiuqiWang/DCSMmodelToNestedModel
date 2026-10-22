# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:53:06 2026

@author: WangX3
"""

import xarray as xr
from pathlib import Path
from typing import Hashable, Sequence

import matplotlib.pyplot as plt
import numpy as np

# for label, path in {
#     "A_direct_wave_wind": "C:/dev/DCSM-FM/TestMeteo_flowwave_meteotowave/wave/wavm-Waves.nc",
#     "B_flow_wind_extended": "C:/dev/DCSM-FM/TestMeteo_flowwave_meteotowavefromflow/wave/wavm-Waves.nc",
#     "C_no_wind": "C:/dev/DCSM-FM/TestMeteo_flowwave_nowindwave/wave/wavm-Waves.nc",
# }.items():
#     ds = xr.open_dataset(path)
#     print(f"\n--- {label} ---")
#     print(ds)


def compute_domain_hs_statistics(
    nc_file: str | Path,
    hs_var: str,
    time_dim: str = "time",
    spatial_dims: Sequence[Hashable] | None = None,
    valid_min: float = 0.0,
) -> xr.Dataset:
    """
    Compute full-domain mean and maximum significant wave height at each time.

    Parameters
    ----------
    nc_file
        Path to a D-Waves/SWAN NetCDF output file.
    hs_var
        Name of the significant-wave-height variable in the NetCDF file.
        Common possibilities include 'hs', 'Hsig', 'hsign', or similar.
    time_dim
        Name of the time dimension.
    spatial_dims
        Spatial dimensions over which to calculate statistics. When None,
        all dimensions of hs_var except time_dim are used.
    valid_min
        Values below this threshold are excluded. A value of 0 excludes
        negative fill or invalid wave heights while retaining calm cells.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - hs_mean: spatial mean Hs at every time
        - hs_max: spatial maximum Hs at every time
    """
    nc_file = Path(nc_file)

    if not nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    with xr.open_dataset(nc_file) as ds:
        if hs_var not in ds:
            available = ", ".join(ds.data_vars)
            raise KeyError(
                f"Variable {hs_var!r} was not found in {nc_file}.\n"
                f"Available data variables are: {available}"
            )

        hs = ds[hs_var]

        if time_dim not in hs.dims:
            raise ValueError(
                f"Time dimension {time_dim!r} is not present in "
                f"{hs_var!r}. Dimensions are {hs.dims}."
            )

        if spatial_dims is None:
            spatial_dims = [dim for dim in hs.dims if dim != time_dim]
        else:
            spatial_dims = list(spatial_dims)

        if not spatial_dims:
            raise ValueError("No spatial dimensions were identified.")

        missing_dims = [dim for dim in spatial_dims if dim not in hs.dims]
        if missing_dims:
            raise ValueError(
                f"Spatial dimensions not found in {hs_var!r}: {missing_dims}"
            )

        # Exclude NaNs, fill values decoded as NaN, and negative invalid values.
        hs_valid = hs.where(np.isfinite(hs) & (hs >= valid_min))

        result = xr.Dataset(
            {
                "hs_mean": hs_valid.mean(
                    dim=spatial_dims,
                    skipna=True,
                    keep_attrs=False,
                ),
                "hs_max": hs_valid.max(
                    dim=spatial_dims,
                    skipna=True,
                    keep_attrs=False,
                ),
            }
        ).load()

    result["hs_mean"].attrs = {
        "long_name": "Full-domain mean significant wave height",
        "units": hs.attrs.get("units", "m"),
    }
    result["hs_max"].attrs = {
        "long_name": "Full-domain maximum significant wave height",
        "units": hs.attrs.get("units", "m"),
    }

    return result


def plot_hs_statistics(
    case_statistics: dict[str, xr.Dataset],
    time_dim: str = "time",
    output_file: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot domain-mean and domain-maximum Hs for multiple cases.

    Parameters
    ----------
    case_statistics
        Mapping from case labels to outputs returned by
        compute_domain_hs_statistics().
    time_dim
        Name of the time coordinate.
    output_file
        Optional path for saving the figure.

    Returns
    -------
    fig, axes
        Matplotlib figure and two axes.
    """
    if not case_statistics:
        raise ValueError("No case statistics were supplied.")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )

    for case_name, statistics in case_statistics.items():
        if time_dim not in statistics.coords:
            raise ValueError(
                f"{case_name!r} has no {time_dim!r} coordinate."
            )

        time = statistics[time_dim]

        axes[0].plot(
            time,
            statistics["hs_mean"],
            label=case_name,
            linewidth=1.8,
        )
        axes[1].plot(
            time,
            statistics["hs_max"],
            label=case_name,
            linewidth=1.8,
        )

    axes[0].set_title("Full-domain mean significant wave height")
    axes[0].set_ylabel(r"Mean $H_s$ (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Full-domain maximum significant wave height")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"Maximum $H_s$ (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig, axes

# Check your files with:
# print(xr.open_dataset("case_1/swan-wave.nc"))

HS_VARIABLE = "hsign"  # Replace with the actual Hs variable name.

case_files = {
    "1: Wind supplied to D-Waves": (
        "C:/dev/DCSM-FM/TestMeteo_flowwave_meteotowave/wave/wavm-Waves.nc"
    ),
    "2: FLOW wind extended to D-Waves": (
        "C:/dev/DCSM-FM/TestMeteo_flowwave_meteotowavefromflow/wave/wavm-Waves.nc"
    ),
    "3: No wind": (
        "C:/dev/DCSM-FM/TestMeteo_flowwave_nowindwave/wave/wavm-Waves.nc"
    ),
}

case_statistics = {
    case_name: compute_domain_hs_statistics(
        nc_file=nc_file,
        hs_var=HS_VARIABLE,
        time_dim="time",
    )
    for case_name, nc_file in case_files.items()
}

fig, axes = plot_hs_statistics(
    case_statistics,
    time_dim="time",
    output_file="hs_domain_statistics_comparison.png",
)

plt.show()