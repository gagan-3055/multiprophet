"""
Daily Dual-Account ERA5-Land Downloader to Combined CSV
======================================================
Downloads ERA5-Land day-by-day using TWO Copernicus CDS accounts concurrently.
Synchronization Barrier: Does NOT advance to the next day until BOTH accounts
have successfully completed downloading and merging the previous day's features.

Workflow per Day:
  1. Account 1 (Primary ~/.cdsapirc) downloads 6 Core Features:
       - 2m_temperature
       - snow_depth
       - snow_density
       - temperature_of_snow_layer
       - snowfall
       - total_precipitation
  2. Account 2 (API key eaa6d554-ca76-4b4a-8094-f85cdb55ea34) downloads remaining 5 Features:
       - 2m_dewpoint_temperature
       - snow_cover
       - snowmelt
       - snow_evaporation
       - surface_solar_radiation_downwards
  3. Wait for BOTH accounts to finish the current day.
  4. Merge both datasets into a single pandas DataFrame across (time, lat, lon).
  5. Append the combined rows directly into the CSV file.
  6. Only then advance to the next day!
"""

import os
import sys
import time
import argparse
import tempfile
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import cdsapi
import pandas as pd
import xarray as xr

# Default Account credentials
DEFAULT_KEY1 = "234f8372-f3eb-4c0f-8f5e-c53cb612c563"
DEFAULT_KEY2 = "eaa6d554-ca76-4b4a-8094-f85cdb55ea34"
CDS_URL = "https://cds.climate.copernicus.eu/api"

# Feature Group 1 (Account 1 - 6 Core Snow & Weather Features)
GROUP_1_FEATURES = [
    "2m_temperature",
    "snow_depth",
    "snow_density",
    "temperature_of_snow_layer",
    "snowfall",
    "total_precipitation",
]

# Feature Group 2 (Account 2 - 5 Remaining Surface & Radiation Features)
GROUP_2_FEATURES = [
    "2m_dewpoint_temperature",
    "snow_cover",
    "snowmelt",
    "snow_evaporation",
    "surface_solar_radiation_downwards",
]

# Default coordinates: Gulmarg / Pir Panjal avalanche area
DEFAULT_AREA = [34.30, 74.15, 33.85, 74.65]

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# Aggregation rules for daily data
MEAN_VARS = {
    "t2m", "2m_temperature",
    "d2m", "2m_dewpoint_temperature",
    "tsn", "temperature_of_snow_layer",
    "sde", "snow_depth",
    "rsn", "snow_density",
    "snowc", "snow_cover",
}

ACCUM_VARS = {
    "tp", "total_precipitation",
    "sf", "snowfall",
    "smlt", "snowmelt",
    "es", "snow_evaporation",
    "ssrd", "surface_solar_radiation_downwards",
}

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe console print."""
    with print_lock:
        print(*args, **kwargs)


def download_single_account(
    account_label: str,
    client: cdsapi.Client,
    date_str: str,
    variables: list[str],
    target_nc_path: str,
    area: list[float],
):
    """Downloads one feature group for a specific single date."""
    year, month, day = date_str.split("-")
    request = {
        "variable": variables,
        "year": [year],
        "month": [month],
        "day": [day],
        "time": ALL_HOURS,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    safe_print(f"[{account_label}] Submitting request for {date_str} ({len(variables)} variables)...")
    try:
        client.retrieve("reanalysis-era5-land", request, target_nc_path)
        safe_print(f"[{account_label}] Download completed for {date_str} -> {os.path.basename(target_nc_path)}")
        return target_nc_path
    except Exception as e:
        safe_print(f"[{account_label}] [ERROR] Failed on {date_str}: {e}")
        raise


def get_existing_dates_in_csv(csv_path: str) -> set[str]:
    """Scan existing CSV to find dates already downloaded, allowing safe resuming."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    try:
        df_sample = pd.read_csv(csv_path, nrows=5)
        time_col = next((c for c in ["date", "valid_time", "time"] if c in df_sample.columns), None)
        if time_col:
            df_col = pd.read_csv(csv_path, usecols=[time_col])
            dates = pd.to_datetime(df_col[time_col]).dt.strftime("%Y-%m-%d").unique()
            return set(dates)
    except Exception as e:
        safe_print(f"[RESUME CHECK] Could not inspect existing CSV ({e}). Starting fresh.")
    return set()


def merge_day_to_csv(
    nc_part1: str,
    nc_part2: str,
    csv_path: str,
    date_str: str,
    daily: bool = True,
    area_mean: bool = True,
    area_str: str = "[34.30, 74.15, 33.85, 74.65]",
):
    """
    Opens both daily NetCDFs with xarray, merges variables, aggregates
    to daily values (mean for state vars, max accumulation for fluxes),
    and averages over the bounding box area into 1 row per day.
    """
    with xr.open_dataset(nc_part1) as ds1, xr.open_dataset(nc_part2) as ds2:
        ds_combined = xr.merge([ds1, ds2], compat="override")

        if daily:
            time_dim = "valid_time" if "valid_time" in ds_combined.dims else ("time" if "time" in ds_combined.dims else None)
            present_vars = list(ds_combined.data_vars.keys())
            present_mean = [v for v in present_vars if v in MEAN_VARS]
            present_accum = [v for v in present_vars if v in ACCUM_VARS]
            other_vars = [v for v in present_vars if v not in present_mean and v not in present_accum]

            parts = []
            if present_mean:
                parts.append(ds_combined[present_mean].mean(dim=time_dim))
            if present_accum:
                # In ERA5-Land, accumulations increment from 00:00 to 24:00; max gives the 24h total
                parts.append(ds_combined[present_accum].max(dim=time_dim))
            if other_vars:
                parts.append(ds_combined[other_vars].mean(dim=time_dim))

            ds_daily = xr.merge(parts, compat="override")
            df = ds_daily.to_dataframe().reset_index()

            for col in ["number", "expver", "valid_time", "time"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            if area_mean:
                # Spatially average all grid cells in the bounding box -> exactly 1 row per day
                df_mean = df.mean(numeric_only=True).to_frame().T
                df_mean.drop(columns=["latitude", "longitude"], inplace=True, errors="ignore")
                df_mean.insert(0, "area", area_str)
                df_mean.insert(0, "date", date_str)
                df = df_mean
            else:
                df.insert(0, "date", date_str)
                coord_cols = ["date", "latitude", "longitude"]
                data_cols = [c for c in df.columns if c not in coord_cols]
                df = df[coord_cols + sorted(data_cols)]
        else:
            df = ds_combined.to_dataframe().reset_index()
            if "valid_time" in df.columns and "time" not in df.columns:
                df.rename(columns={"valid_time": "time"}, inplace=True)

            coord_cols = [c for c in ["time", "latitude", "longitude"] if c in df.columns]
            data_cols = [c for c in df.columns if c not in coord_cols]
            df = df[coord_cols + sorted(data_cols)]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, mode="a", index=False, header=write_header)
    return len(df)


def generate_date_list(years: list[int], months: list[int], start_date: str = None, end_date: str = None):
    """Generate list of 'YYYY-MM-DD' strings to process."""
    import calendar
    if start_date and end_date:
        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        cur = d_start
        dates = []
        while cur <= d_end:
            dates.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return dates

    dates = []
    for y in sorted(years):
        for m in sorted(months):
            _, num_days = calendar.monthrange(y, m)
            for d in range(1, num_days + 1):
                dates.append(f"{y}-{m:02d}-{d:02d}")
    return dates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Daily Dual-Account ERA5-Land Downloader to Combined CSV (strict day-by-day barrier)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="era5_land_gulmarg_daily.csv",
        help="Target combined CSV filepath. Default: era5_land_gulmarg_daily.csv",
    )
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Keep 24 hourly rows per day instead of aggregating to daily values (Default: daily aggregation).",
    )
    parser.add_argument(
        "--keep-coords",
        action="store_true",
        help="Keep all 25 individual grid coordinate rows per day instead of averaging across the bounding box (Default: averages area into 1 row/day).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023],
        help="Years to download. Default: 2023",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[1],
        help="Months to download. Default: 1 (January)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional exact start date (YYYY-MM-DD), e.g. 2023-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional exact end date (YYYY-MM-DD), e.g. 2023-01-07",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=DEFAULT_AREA,
        help="Bounding box [N, W, S, E]. Default: 34.30 74.15 33.85 74.65 (Gulmarg / Pir Panjal)",
    )
    parser.add_argument(
        "--key1",
        type=str,
        default=DEFAULT_KEY1,
        help=f"API key for Account 1. Default: {DEFAULT_KEY1}",
    )
    parser.add_argument(
        "--key2",
        type=str,
        default=DEFAULT_KEY2,
        help=f"API key for Account 2. Default: {DEFAULT_KEY2}",
    )
    parser.add_argument(
        "--keep-temp-nc",
        action="store_true",
        help="Keep the downloaded intermediate .nc files instead of deleting after appending to CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine date list
    all_dates = generate_date_list(
        years=args.years,
        months=args.months,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("==========================================================================")
    print("DAILY DUAL-ACCOUNT ERA5-LAND -> COMBINED CSV")
    print("==========================================================================")
    print(f"Target CSV File : {os.path.abspath(args.csv_file)}")
    print(f"Coordinates      : N={args.area[0]}, W={args.area[1]}, S={args.area[2]}, E={args.area[3]}")
    print(f"Total Days       : {len(all_dates)} days (from {all_dates[0]} to {all_dates[-1]})")
    print(f"Output Mode      : {'Hourly (24 rows/coord/day)' if args.hourly else 'Daily Aggregated (1 row/coord/day)'}")
    print(f"Account 1 (Core 6)      : Key {args.key1[:8]}...{args.key1[-6:]}")
    print(f"Account 2 (Remaining 5) : Key {args.key2[:8]}...{args.key2[-6:]}")
    print("Synchronization  : Strict daily barrier (advances ONLY when both accounts finish)")
    print("==========================================================================\n")

    # Check for existing dates in CSV to resume seamlessly
    existing_dates = get_existing_dates_in_csv(args.csv_file)
    if existing_dates:
        print(f"[RESUME] Found {len(existing_dates)} date(s) already recorded in '{args.csv_file}'.")

    # Initialize CDS Clients
    try:
        client1 = cdsapi.Client(url=CDS_URL, key=args.key1)
        print(f"[INIT] Client 1 (Account 1) initialized with key {args.key1[:8]}...")
    except Exception as err:
        print(f"[FATAL] Failed to initialize Client 1: {err}", file=sys.stderr)
        sys.exit(1)

    try:
        client2 = cdsapi.Client(url=CDS_URL, key=args.key2)
        print(f"[INIT] Client 2 (Account 2) initialized with key {args.key2[:8]}...")
    except Exception as err:
        print(f"[FATAL] Failed to initialize Client 2: {err}", file=sys.stderr)
        sys.exit(1)

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv_file)), "temp_daily_nc")
    os.makedirs(temp_dir, exist_ok=True)

    completed_count = 0
    total_dates = len(all_dates)

    for idx, date_str in enumerate(all_dates, 1):
        if date_str in existing_dates:
            print(f"[{idx}/{total_dates}] [SKIP] {date_str} already in CSV.")
            completed_count += 1
            continue

        print(f"\n==========================================================================")
        print(f"[{idx}/{total_dates}] PROCESSING DATE: {date_str}")
        print(f"==========================================================================")

        nc_part1 = os.path.join(temp_dir, f"era5_part1_{date_str}.nc")
        nc_part2 = os.path.join(temp_dir, f"era5_part2_{date_str}.nc")

        # Execute both accounts in parallel threads for this exact day
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_acc1 = executor.submit(
                download_single_account,
                account_label="Account 1 (Core 6)",
                client=client1,
                date_str=date_str,
                variables=GROUP_1_FEATURES,
                target_nc_path=nc_part1,
                area=args.area,
            )

            future_acc2 = executor.submit(
                download_single_account,
                account_label="Account 2 (Remaining 5)",
                client=client2,
                date_str=date_str,
                variables=GROUP_2_FEATURES,
                target_nc_path=nc_part2,
                area=args.area,
            )

            # SYNCHRONIZATION BARRIER:
            # Wait for BOTH Account 1 and Account 2 to finish the current day.
            acc1_ok = False
            acc2_ok = False

            try:
                future_acc1.result()
                acc1_ok = True
            except Exception as e:
                print(f"[BARRIER BLOCK] Account 1 failed on {date_str}: {e}")

            try:
                future_acc2.result()
                acc2_ok = True
            except Exception as e:
                print(f"[BARRIER BLOCK] Account 2 failed on {date_str}: {e}")

            # Do not advance to next day if either account failed!
            if not (acc1_ok and acc2_ok):
                print(f"\n[CRITICAL ERROR] Stopping pipeline. Day {date_str} could not be completed by both accounts.")
                print(f"Account 1 status: {'SUCCESS' if acc1_ok else 'FAILED'}")
                print(f"Account 2 status: {'SUCCESS' if acc2_ok else 'FAILED'}")
                print("Resolve network/server issue and re-run. Previously saved days remain safe in CSV.")
                sys.exit(1)

        # Merge both NetCDF files into the combined CSV file
        try:
            num_rows = merge_day_to_csv(
                nc_part1,
                nc_part2,
                args.csv_file,
                date_str=date_str,
                daily=not args.hourly,
                area_mean=not args.keep_coords,
                area_str=f"[{args.area[0]:.2f}, {args.area[1]:.2f}, {args.area[2]:.2f}, {args.area[3]:.2f}]",
            )
            mode_lbl = "hourly" if args.hourly else "daily"
            print(f"[CSV APPEND] Successfully appended {num_rows} {mode_lbl} row(s) for {date_str} to '{args.csv_file}'.")
            completed_count += 1
        except Exception as e:
            print(f"[MERGE ERROR] Failed to merge {date_str} into CSV: {e}")
            sys.exit(1)

        # Cleanup intermediate daily .nc files unless user asked to keep them
        if not args.keep_temp_nc:
            try:
                if os.path.exists(nc_part1):
                    os.remove(nc_part1)
                if os.path.exists(nc_part2):
                    os.remove(nc_part2)
            except Exception:
                pass

        print(f"[BARRIER PASSED] Day {date_str} complete and saved. Advancing to next day...")

    # Cleanup temp directory if empty
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass

    print("\n==========================================================================")
    print(f"[FINISHED] All {completed_count}/{total_dates} days processed successfully!")
    print(f"Combined CSV location: {os.path.abspath(args.csv_file)}")
    print(f"File size: {os.path.getsize(args.csv_file) / (1024*1024):.2f} MB")
    print("==========================================================================")


if __name__ == "__main__":
    main()
