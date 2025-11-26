import nibabel as nib
import numpy as np

# Load the segmentation file
seg = nib.load('data/test/segmentation-0.nii')
data = seg.get_fdata()

print("Segmentation file analysis:")
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min value: {data.min()}")
print(f"Max value: {data.max()}")

unique_values = np.unique(data)
print(f"\nUnique label values: {unique_values}")
print(f"Count of each label:")
for val in unique_values:
    count = np.sum(data == val)
    percentage = (count / data.size) * 100
    print(f"  Label {int(val)}: {count} voxels ({percentage:.2f}%)")

print("\n" + "="*50)
print("Expected labels for this model:")
print("  0 = Background")
print("  1 = Liver")
print("  2 = Tumor/Lesion")
print("="*50)
