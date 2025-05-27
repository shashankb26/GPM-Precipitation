import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
import requests
from tqdm import tqdm
import cftime
import warnings
from tenacity import retry, stop_after_attempt, wait_fixed

def gpm_url_generator(lon_w, lon_e, lat_s, lat_n, date_begin, date_end):
    urls = []
    for date in pd.date_range(date_begin, date_end):
        end = date + timedelta(hours=23, minutes=59, seconds=59)
        for i, t in enumerate(pd.date_range(date, end, freq='0.5h')):
            doi = t.strftime('%j')
            y, m, d = t.strftime('%Y'), t.strftime('%m'), t.strftime('%d')
            p1 = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3/GPM_3IMERGHHL.07'
            p2 = f'/{y}/{doi}/3B-HHR-L.MS.MRG.3IMERG.{y}{m}{d}-'
            p3 = f"S{t.strftime('%H%M%S')}-E{(t + timedelta(minutes=29, seconds=59)).strftime('%H%M%S')}.{i * 30:04d}.V07B.HDF5.nc4"
            urls.append(f"{p1}{p2}{p3}")
    return urls

EARTHDATA_TOKEN = "Enter your Token"

end_date = datetime(2025, 5, 23)  
start_date = datetime(2025, 4, 15)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

output_base_dir = 'D:/Downloads/Grid_Rainfall/process_daily_rainfall/'
os.makedirs(output_base_dir, exist_ok=True)

headers = {"Authorization": f"Bearer {EARTHDATA_TOKEN}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_file(url, save_path, headers):
    r = requests.get(url, headers=headers, stream=True)
    if r.status_code != 200:
        raise Exception(f"Download failed: {url} (HTTP {r.status_code})")
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

grid_data_path = 'D:/Downloads/Grid_Rainfall/Grid_Data.csv'
df = pd.read_csv(grid_data_path)
if df.empty:
    raise ValueError("CSV is empty")

df = df.rename(columns={'ObjectID': 'grid_id', 'Latitude': 'lat', 'Longitude': 'lon'})
if not all(col in df for col in ['grid_id', 'lat', 'lon']):
    raise ValueError(f"CSV missing columns: {df.columns}")

df = df[(df['lat'].between(20, 30)) & (df['lon'].between(87, 98))]

def daily_rainfall_grid_vectorized(dataset, lats, lons, date, grid_size=0.1, precip_var='precipitation'):
    start = pd.Timestamp(date)
    end = start + timedelta(hours=23, minutes=59, seconds=59)
    if isinstance(dataset.indexes.get('time', []), xr.CFTimeIndex):
        dataset['time'] = dataset.indexes['time'].to_datetimeindex()
    
    rainfall = []
    for lat, lon in zip(lats, lons):
        try:
            # Select data for the grid point and full day
            slice_data = dataset.sel(
                time=slice(start, end),
                lat=slice(lat - grid_size, lat + grid_size),
                lon=slice(lon - grid_size, lon + grid_size)
            )
            if len(slice_data['time']) < 48:
                print(f"Warning: Only {len(slice_data['time'])} time steps found for lat={lat}, lon={lon}, date={date}")
            val = slice_data[precip_var].sum(dim='time').mean(dim=['lat', 'lon'])
            rainfall.append(float(val) if not np.isnan(val) else 0.0)
        except Exception as e:
            print(f"Error at lat={lat}, lon={lon}, date={date}: {e}")
            rainfall.append(0.0)
    return rainfall

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    date_dir_str = date.strftime('%Y%m%d')
    
    output_path = os.path.join(output_base_dir, f'daily_rainfall_{date_dir_str}.xlsx')
    if os.path.exists(output_path):
        print(f"Skipping {date_str}: {output_path} exists")
        continue

    base_data_dir = f"D:/Downloads/Grid_Rainfall/{date_dir_str}/"
    try:
        os.makedirs(base_data_dir, exist_ok=True)
    except PermissionError as e:
        print(f"Cannot create {base_data_dir} for {date_str}: {e}")
        continue

    url_list = gpm_url_generator(lon_w=87, lon_e=98, lat_s=20, lat_n=30, date_begin=date_str, date_end=date_str)

    for url in tqdm(url_list, desc=f"Downloading {date_str}"):
        filename = url.split("/")[-1]
        save_path = os.path.join(base_data_dir, filename)
        if os.path.exists(save_path):
            continue
        try:
            download_file(url, save_path, headers)
        except Exception as e:
            print(f"Download error {filename} for {date_str}: {e}")

    hdf5_files = glob.glob(os.path.join(base_data_dir, '*.HDF5*'))
    print(f"Found files for {date_str}: {hdf5_files}")
    if not hdf5_files:
        print(f"No HDF5 files for {date_str}. Skipping")
        continue

    valid_files = []
    for file in hdf5_files:
        try:
            with xr.open_dataset(file) as ds:
                if 'precipitation' in ds.variables:
                    valid_files.append(file)
                else:
                    print(f"Skipping invalid file {file}: 'precipitation' variable missing")
        except Exception as e:
            print(f"Skipping invalid file {file}: {e}")
    
    if not valid_files:
        print(f"No valid HDF5 files for {date_str}. Skipping")
        continue

    try:
        dataset = xr.open_mfdataset(valid_files, combine='by_coords')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if isinstance(dataset.indexes.get('time', []), xr.CFTimeIndex):
                dataset['time'] = dataset.indexes['time'].to_datetimeindex()
        print(f"Time steps for {date_str}: {len(dataset['time'])}")
    except Exception as e:
        print(f"Load error for {date_str}: {e}")
        continue

    rainfall_vals = daily_rainfall_grid_vectorized(dataset, df['lat'], df['lon'], date_str, precip_var='precipitation')

    combined_df = pd.DataFrame({
        'ObjectID': df['grid_id'].astype(int),
        'Latitude': df['lat'],
        'Longitude': df['lon'],
        'Total_Rainfall (mm)': rainfall_vals,
        'measurement_date': date_str
    })

    try:
        combined_df.to_excel(output_path, index=False)
        print(f"Saved {output_path} for {date_str}")
    except PermissionError as e:
        print(f"Cannot save {output_path} for {date_str}: {e}")
        continue
    except Exception as e:
        print(f"Save error {output_path} for {date_str}: {e}")
        continue
