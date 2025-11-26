import nibabel as nib
import numpy as np

# Load the NIfTI file
nii_file = nib.load('./CT_Electrodes.nii.gz')

# Get the data array
data = nii_file.get_fdata()

print('=== NIfTI File Information ===')
print(f'Shape: {data.shape}')
print(f'Data type: {data.dtype}')
print(f'Min value: {data.min()}')
print(f'Max value: {data.max()}')
print(f'Mean value: {data.mean():.2f}')
print(f'Unique values count: {len(np.unique(data))}')
print()

print('=== Header Information ===')
print(f'Affine matrix shape: {nii_file.affine.shape}')
print(f'Voxel dimensions: {nii_file.header.get_zooms()}')
print(f'Data dimensions: {nii_file.header.get_data_shape()}')
print()

# Check if it looks like a volume or segmentation
unique_vals = np.unique(data)
print(f'=== Data Analysis ===')
print(f'Number of unique values: {len(unique_vals)}')
if len(unique_vals) <= 10:
    print(f'Unique values: {unique_vals}')
    print('This appears to be a SEGMENTATION mask (few discrete values)')
else:
    print(f'Value range: [{data.min()}, {data.max()}]')
    print('This appears to be a VOLUME/CT scan (continuous values)')
