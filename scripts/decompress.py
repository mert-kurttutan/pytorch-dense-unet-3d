import nibabel as nib
import numpy as np
import os

# Load the compressed NIfTI file
print("Loading CT_Electrodes.nii.gz...")
nii = nib.load('CT_Electrodes.nii.gz')
data = nii.get_fdata()

# Analyze the file type
unique_vals = np.unique(data)
print(f'\nFile Analysis:')
print(f'Shape: {data.shape}')
print(f'Unique values count: {len(unique_vals)}')
print(f'Min: {data.min()}, Max: {data.max()}')

if len(unique_vals) <= 10:
    print(f'Unique values: {unique_vals}')
    print('Type: SEGMENTATION mask')
    file_type = 'segmentation'
else:
    print('Type: VOLUME/CT scan')
    file_type = 'volume'

# Determine output filename
output_filename = f'{file_type}-0.nii'
output_path = os.path.join('data', 'test', output_filename)

# Save as uncompressed .nii file
print(f'\nDecompressing and saving to: {output_path}')
nib.save(nii, output_path)

print(f'Done! File saved as {output_path}')
print(f'\nNote: The dataset loader expects BOTH volume and segmentation files.')
print(f'You will need to provide the matching pair file as well.')
