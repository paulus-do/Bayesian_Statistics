import pandas as pd


def load_and_prepare_data(file_path="df_equator.csv"):
    """
    Load the 1D equator data.

    The returned list contains one vector per year. Each vector holds the
    temperature means on the filtered longitude interval [30, 60).
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["lon", "year"])

    # Filter longitude range
    df = df[(df["lon"] >= 30) & (df["lon"] < 60)]

    # Pivot the dataframe to have years as rows and longitudes as columns
    pivot_df = df.pivot(index="year", columns="lon", values="mean")

    # Create the desired structure: list of numpy arrays
    # Each array represents temperatures across all longitudes for a specific year
    data = [pivot_df.iloc[i].values for i in range(len(pivot_df))]

    return data


def load_and_prepare_data_2D(file_path="data/df_equator_2D.csv"):
    """
    Load the 2D spatial temperature data.

    The returned list contains one latitude-by-longitude array per year on
    latitude range [33, 53] and longitude range [2, 22].
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["year", "lat", "lon"])

    # Filter latitude and longitude ranges
    df = df[
        (df["lat"] >= 33)
        & (df["lat"] <= 53)
        & (df["lon"] >= 2)
        & (df["lon"] <= 22)
    ]

    # Get unique coordinates to understand the grid structure
    lats = sorted(df["lat"].unique())
    lons = sorted(df["lon"].unique())
    years = sorted(df["year"].unique())

    print(f"Grid dimensions: {len(lats)} latitudes × {len(lons)} longitudes")
    print(f"Number of years: {len(years)}")
    print(f"Latitude range: {min(lats)} to {max(lats)}")
    print(f"Longitude range: {min(lons)} to {max(lons)}")

    # Create the 2D structure: list of 2D arrays (lat × lon) for each year
    data = []

    for year in years:
        year_data = df[df["year"] == year]

        # Pivot to create 2D grid: rows=latitude, columns=longitude
        pivot_2d = year_data.pivot(index="lat", columns="lon", values="mean")

        # Reindex to ensure consistent shape across all years
        # This fills in any missing lat/lon combinations with NaN
        pivot_2d = pivot_2d.reindex(index=lats, columns=lons)

        data.append(pivot_2d.values)

    return data
