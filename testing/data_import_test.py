import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import numpy as np
from src.data_import import load_and_prepare_data, load_and_prepare_data_2D
from mpl_toolkits.mplot3d import Axes3D

##############################################################
# 1 D plot of datapoints (lat=equator)
def plot_longitude_means_individual(data_list, years=None):
    """
    Plot each longitude-year pair individually, mimicking the original behavior.
    """
    plt.figure(figsize=(10, 6))
    
    data_array = np.array(data_list)
    n_longitudes = data_array.shape[1]
    n_times = data_array.shape[0]
    
    # Create longitude values
    longitudes = np.linspace(30, 60, n_longitudes, endpoint=False)
    
    if years is None:
        years = list(range(n_times))
    
    # Create color mapping
    cmap = plt.get_cmap('tab20')
    year_colors = {year: cmap(i % 20) for i, year in enumerate(years)}
    
    # Plot each longitude and time point individually
    for lon_idx, lon in enumerate(longitudes):
        for time_idx, year in enumerate(years):
            temp = data_array[time_idx, lon_idx]
            # Only add label for first longitude to avoid duplicate legend entries
            label = str(year) if lon_idx == 0 else ""
            plt.scatter(lon, temp, 
                       color=year_colors[year], alpha=0.7, s=20)
    
    plt.title("Mean Temperature per Year for Each Longitude", fontsize=24, fontweight='bold')
    plt.xlabel("Longitude", fontsize=18)
    plt.ylabel("Mean Temperature", fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Create clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label:  # Only consider non-empty labels
            unique[label] = handle
    
    # Sort years numerically
    sorted_years = sorted(unique.keys(), key=int)
    unique_handles = [unique[year] for year in sorted_years]
    
    #plt.legend(unique_handles, sorted_years, title="Year", loc='best')
    
    plt.tight_layout()
    plt.show()
###############################################################
# 2 D plot of datapoints  (These 2 functions are ok, not the most useful but ok)
def plot_temperature_3d(data, lat_range=(33, 53), lon_range=(2, 22)):
    """
    Create a 3D surface plot of temperature data similar in style to the 2D plot.
    
    Args:
        data (list): List of 2D numpy arrays with shape (latitude, longitude)
        lat_range (tuple): Latitude range (min, max)
        lon_range (tuple): Longitude range (min, max)
    """
    # Create coordinate grids
    lats = np.linspace(lat_range[0], lat_range[1], data[0].shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], data[0].shape[1])
    Lons, Lats = np.meshgrid(lons, lats)
    
    # Create figure with similar style to the 2D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each year as a semi-transparent surface
    years = range(1983, 1983 + len(data))
    
    for i, (year_data, year) in enumerate(zip(data, years)):
        # Use a colormap similar to the 2D plot
        surf = ax.plot_surface(Lons, Lats, year_data, 
                              alpha=0.7,  # Semi-transparent
                              cmap='viridis',
                              linewidth=0,
                              antialiased=True,
                              label=str(year))
    
    # Set labels and title with similar style
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_zlabel('Mean Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Mean Temperature Distribution Over Latitude and Longitude', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks similar to 2D plot
    ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
    ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1, 
                 label='Mean Temperature')
    
    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    return fig, ax


def plot_temperature_3d_single_year(data, year_index=0, lat_range=(33, 53), lon_range=(2, 22)):
    """
    Create a 3D surface plot for a single year.
    
    Args:
        data (list): List of 2D numpy arrays
        year_index (int): Index of the year to plot
        lat_range (tuple): Latitude range (min, max)
        lon_range (tuple): Longitude range (min, max)
    """
    year_data = data[year_index]
    year = 1983 + year_index
    
    # Create coordinate grids
    lats = np.linspace(lat_range[0], lat_range[1], year_data.shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], year_data.shape[1])
    Lons, Lats = np.meshgrid(lons, lats)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(Lons, Lats, year_data,
                          cmap='viridis',
                          alpha=0.9,
                          linewidth=0.5,
                          edgecolor='black',
                          antialiased=True)
    
    # Customize the plot
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_zlabel('Mean Temperature', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean Temperature Distribution - Year {year}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks
    ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
    ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1,
                 label='Mean Temperature')
    
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    
    return fig, ax
###############################################################

def plot_temperature_3d_scatter(data, lat_range=(33, 53), lon_range=(2, 22)):
    """
    Create a 3D scatter plot of temperature data with color coding by year.
    
    Args:
        data (list): List of 2D numpy arrays with shape (latitude, longitude)
        lat_range (tuple): Latitude range (min, max)
        lon_range (tuple): Longitude range (min, max)
    """
    # Create coordinate grids
    lats = np.linspace(lat_range[0], lat_range[1], data[0].shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], data[0].shape[1])
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate colors for each year
    years = range(1983, 1983 + len(data))
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    # Plot each year as scatter points
    for i, (year_data, year, color) in enumerate(zip(data, years, colors)):
        # Create meshgrid for this year's data
        Lons, Lats = np.meshgrid(lons, lats)
        
        # Flatten the arrays for scatter plot
        x_flat = Lons.flatten()
        y_flat = Lats.flatten()
        z_flat = year_data.flatten()
        
        # Remove NaN values if any
        mask = ~np.isnan(z_flat)
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        z_flat = z_flat[mask]
        
        # Plot scatter points for this year
        scatter = ax.scatter(x_flat, y_flat, z_flat,
                           c=[color] * len(x_flat),  # Same color for all points in this year
                           s=30,                    # Point size
                           alpha=0.7,               # Transparency
                           edgecolors='black',      # Black edges for better visibility
                           linewidth=0.3,
                           label=f'{year}')
    
    # Customize the plot
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_zlabel('Mean Temperature', fontsize=12, fontweight='bold')
    ax.set_title('Mean Temperature - 3D Scatter Plot by Year', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set ticks
    ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
    ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              title='Year', fontsize=10)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig, ax


def plot_temperature_3d_interactive(data, lat_range=(33, 53), lon_range=(2, 22)):
    """
    Interactive 3D scatter plot with year slider.
    """
    from matplotlib.widgets import Slider
    
    # Create coordinate grids
    lats = np.linspace(lat_range[0], lat_range[1], data[0].shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], data[0].shape[1])
    Lons, Lats = np.meshgrid(lons, lats)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)  # Make room for slider
    
    years = range(1983, 1983 + len(data))
    
    # Initial plot
    year_data = data[0]
    x_flat = Lons.flatten()
    y_flat = Lats.flatten()
    z_flat = year_data.flatten()
    mask = ~np.isnan(z_flat)
    
    scatter = ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask],
                        c=z_flat[mask], cmap='viridis', s=40, alpha=0.8)
    
    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_zlabel('Mean Temperature', fontweight='bold')
    ax.set_title(f'Mean Temperature - Year {years[0]}', fontweight='bold')
    ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
    ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
    ax.view_init(elev=20, azim=45)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    year_slider = Slider(ax_slider, 'Year', 0, len(data)-1, valinit=0, valstep=1)
    
    def update_slider(val):
        year_idx = int(year_slider.val)
        year_data = data[year_idx]
        
        ax.clear()
        x_flat = Lons.flatten()
        y_flat = Lats.flatten()
        z_flat = year_data.flatten()
        mask = ~np.isnan(z_flat)
        
        scatter = ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask],
                            c=z_flat[mask], cmap='viridis', s=40, alpha=0.8)
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.set_zlabel('Mean Temperature', fontweight='bold')
        ax.set_title(f'Mean Temperature - Year {years[year_idx]}', fontweight='bold')
        ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
        ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
        ax.view_init(elev=20, azim=45)
        fig.canvas.draw_idle()
    
    year_slider.on_changed(update_slider)
    plt.show()
    
    return fig, ax, year_slider


def plot_temperature_3d_scatter_animated_simple(data, lat_range=(33, 53), lon_range=(2, 22)):
    """
    Simple animated 3D scatter plot that displays immediately.
    """
    from matplotlib.animation import FuncAnimation
    
    # Create coordinate grids
    lats = np.linspace(lat_range[0], lat_range[1], data[0].shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], data[0].shape[1])
    Lons, Lats = np.meshgrid(lons, lats)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    years = range(1983, 1983 + len(data))
    
    def update(frame):
        ax.clear()
        year_data = data[frame]
        year = years[frame]
        
        # Flatten arrays
        x_flat = Lons.flatten()
        y_flat = Lats.flatten()
        z_flat = year_data.flatten()
        
        mask = ~np.isnan(z_flat)
        
        # Plot scatter
        scatter = ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask],
                           c=z_flat[mask], cmap='viridis', s=40, alpha=0.8)
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.set_zlabel('Mean Temperature', fontweight='bold')
        ax.set_title(f'Mean Temperature - Year {year}', fontweight='bold')
        
        ax.set_xticks(np.arange(lon_range[0], lon_range[1] + 1, 5))
        ax.set_yticks(np.arange(lat_range[0], lat_range[1] + 1, 5))
        
        # Set consistent axis limits
        if len(data) > 0:
            all_temps = np.concatenate([d[~np.isnan(d)] for d in data])
            ax.set_zlim(np.min(all_temps) - 1, np.max(all_temps) + 1)
        
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        
        return scatter,
    
    # Create and immediately display animation
    ani = FuncAnimation(fig, update, frames=len(data), interval=1000, blit=False, repeat=True)
    plt.show()
    
    return ani

if __name__ == "__main__":
    data = load_and_prepare_data()
    data_2D = load_and_prepare_data_2D()

    # First, let's debug the data structure
    print(f"Data type: {type(data)}")
    print(f"Data length: {len(data)}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        print(f"First element shape: {data[0].shape}")
        print(f"First few longitudes sample: {data[0][:5]}")  # First 5 values
    
    # Test with the individual plotting version (more similar to original)
    plot_longitude_means_individual(data)     #--->2D plot
    plot_temperature_3d_scatter(data_2D)
    plot_temperature_3d_scatter_animated_simple(data_2D)      #-----> 3D interactive (useful)
    plt.show()
    print('Plot generated successfully!')