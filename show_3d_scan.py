import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Function to read a dataset and return it as a NumPy array
def read_dataset(file_path, group_name, dataset_name):
    with h5py.File(file_path, 'r') as hdf:
        dataset = hdf[f'{group_name}/{dataset_name}']
        data = np.array(dataset)
    return data

# Function to downsample the data
def downsample_data(data, factor):
    return data[::factor, ::factor, ::factor]

# Example usage
file_path =  'D:\\Datasets\\FDG-PET-CT-Lesions\\HDF5\\FDG-PET-CT-Lesions.hdf5'
full_dataset_path = 'PETCT_ff1451316e/03-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-22165'  # Update this to your full dataset path
dataset_name = 'ct' 

# Read the dataset 
data = read_dataset(file_path, full_dataset_path, dataset_name)

# Downsample the data to make it more manageable
downsampling_factor = 1  # Adjust this factor to control downsampling
data_downsampled = downsample_data(data, downsampling_factor)

# Create a 3D plot
fig = go.Figure(data=go.Volume(
    x=np.linspace(0, 1, data_downsampled.shape[0]),
    y=np.linspace(0, 1, data_downsampled.shape[1]),
    z=np.linspace(0, 1, data_downsampled.shape[2]),
    value=data_downsampled.flatten(),
    opacity=0.1,  # Adjust this for better visibility
    surface_count=20,  # Adjust this for better resolution
    colorscale='Gray'
))

fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=data_downsampled.shape[2] / data_downsampled.shape[0]))

# Show the plot directly
fig.show(renderer="browser")

# Optionally save the plot as an HTML file
pio.write_html(fig, file='3d_volume_plot_downsampled.html', auto_open=True)