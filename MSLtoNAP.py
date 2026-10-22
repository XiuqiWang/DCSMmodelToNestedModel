import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
his_nc = r"DCSM-FM_0_5nm_restarted_his.nc"
station_name = "south_0006"

# NAP-based observed/downloaded station water level file
# Expected columns: time, waterlevel
time_nap = np.array([
    '2013-12-06T00:00:00.000000000',
    '2013-12-06T00:10:00.000000000',
    '2013-12-06T00:20:00.000000000',
    '2013-12-06T00:30:00.000000000',
    '2013-12-06T00:40:00.000000000',
    '2013-12-06T00:50:00.000000000',
    '2013-12-06T01:00:00.000000000',
    '2013-12-06T01:10:00.000000000',
    '2013-12-06T01:20:00.000000000',
    '2013-12-06T01:30:00.000000000',
    '2013-12-06T01:40:00.000000000',
    '2013-12-06T01:50:00.000000000',
    '2013-12-06T02:00:00.000000000',
    '2013-12-06T02:10:00.000000000',
    '2013-12-06T02:20:00.000000000',
    '2013-12-06T02:30:00.000000000',
    '2013-12-06T02:40:00.000000000',
    '2013-12-06T02:50:00.000000000',
    '2013-12-06T03:00:00.000000000'
], dtype='datetime64[ns]')
wl_nap = [
    1.72,
    1.74,
    1.58,
    1.71,
    1.85,
    1.77,
    1.85,
    1.76,
    1.93,
    1.94,
    2.07,
    1.94,
    2.13,
    2.09,
    2.09,
    2.10,
    2.13,
    2.26,
    2.48,
]

start_time = pd.Timestamp("2013-12-06 00:00:00")
end_time   = pd.Timestamp("2013-12-06 03:00:00")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def decode_names(values):
    return [
        v.decode("utf-8").strip() if isinstance(v, bytes) else str(v).strip()
        for v in values
    ]


# ------------------------------------------------------------
# Read DCSM his.nc water level at south_0003, referenced to MSL
# ------------------------------------------------------------
ds = xr.open_dataset(his_nc)

station_names = decode_names(ds["station_name"].values)

if station_name not in station_names:
    raise ValueError(
        f"{station_name} not found. Available stations:\n"
        + "\n".join(station_names)
    )

idx = station_names.index(station_name)

time_dcsm = pd.to_datetime(ds["time"].values)
wl_dcsm_msl = ds["waterlevel"].isel(station=idx).values

mask_dcsm = (time_dcsm >= start_time) & (time_dcsm <= end_time)

time_dcsm = np.asarray(time_dcsm[mask_dcsm])
wl_dcsm_msl = np.asarray(wl_dcsm_msl[mask_dcsm])

mean_dcsm_msl = np.nanmean(wl_dcsm_msl)

ds.close()


# ------------------------------------------------------------
# Read NAP-based downloaded station water level
# ------------------------------------------------------------
mean_nap = np.nanmean(wl_nap)

# ------------------------------------------------------------
# Compute datum offset
# ------------------------------------------------------------
offset_msl_to_nap = mean_nap - mean_dcsm_msl

print(f"Mean DCSM WL at {station_name} over MSL: {mean_dcsm_msl:.4f} m")
print(f"Mean downloaded station WL over NAP:    {mean_nap:.4f} m")
print(f"Offset MSL -> NAP:                     {offset_msl_to_nap:.4f} m")

print()
print("Apply correction as:")
print("    WL_DCSM_NAP = WL_DCSM_MSL + offset_msl_to_nap")


# ------------------------------------------------------------
# Optional plot: before/after correction
# ------------------------------------------------------------
wl_dcsm_nap = wl_dcsm_msl + offset_msl_to_nap

plt.figure(figsize=(12, 5))

plt.plot(
    time_dcsm,
    wl_dcsm_msl,
    label=f"DCSM {station_name}, original MSL",
    linewidth=2,
)

plt.plot(
    time_dcsm,
    wl_dcsm_nap,
    label=f"DCSM {station_name}, corrected to NAP",
    linewidth=2,
)

plt.plot(
    time_nap,
    wl_nap,
    "k--",
    label="Haringvliet point 10, NAP",
    linewidth=2,
)

plt.xlabel("Time")
plt.ylabel("Water level (m)")
plt.title("MSL-to-NAP offset estimate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()