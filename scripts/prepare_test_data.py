import nibabel as nib
import os

# Output directory
output_dir = 'data/test'
os.makedirs(output_dir, exist_ok=True)

print("Processing sample files for data/test...")
print("-" * 50)

# Load and process volume file
print("\n1. Processing sample.nii.gz -> volume-0.nii")
volume_nii = nib.load('sample.nii.gz')
volume_data = volume_nii.get_fdata()
print(f"   Volume shape: {volume_data.shape}")
print(f"   Volume range: [{volume_data.min():.2f}, {volume_data.max():.2f}]")

# Save as uncompressed .nii
volume_output = os.path.join(output_dir, 'volume-0.nii')
nib.save(volume_nii, volume_output)
print(f"   Saved to: {volume_output}")

# Load and process segmentation file
print("\n2. Processing sample_label.nii.gz -> segmentation-0.nii")
seg_nii = nib.load('sample_label.nii.gz')
seg_data = seg_nii.get_fdata()
print(f"   Segmentation shape: {seg_data.shape}")
print(f"   Unique labels: {sorted(set(seg_data.flatten().astype(int)))}")

# Save as uncompressed .nii
seg_output = os.path.join(output_dir, 'segmentation-0.nii')
nib.save(seg_nii, seg_output)
print(f"   Saved to: {seg_output}")

print("\n" + "-" * 50)
print("Done! Files ready for main.py")
print("\nFiles created:")
print(f"  - {volume_output}")
print(f"  - {seg_output}")
