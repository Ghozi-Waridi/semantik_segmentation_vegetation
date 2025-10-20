import numpy as np
import os
from PIL import Image
from typing import Tuple, List
import glob
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
	handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)


class DataPatcher:
	"""Class untuk melakukan patching data gambar menjadi ukuran tertentu"""
    
	def __init__(self, patch_size: int = 128, stride: int = 128):
		"""
		Args:
			patch_size: Ukuran patch (default 128x128)
			stride: Jarak pergeseran patch (default 128, non-overlapping)
		"""
		self.patch_size = patch_size
		self.stride = stride
        
	def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
		"""
		Ekstrak patches dari gambar besar
        
		Args:
			image: Array gambar dengan shape (H, W, C) atau (H, W)
            
		Returns:
			List of patches
		"""
		patches = []
		h, w = image.shape[:2]
        
		for i in range(0, h - self.patch_size + 1, self.stride):
			for j in range(0, w - self.patch_size + 1, self.stride):
				patch = image[i:i+self.patch_size, j:j+self.patch_size]
				patches.append(patch)
                
		return patches
    
	def load_and_patch_image(self, image_path: str, normalize: bool = True) -> List[np.ndarray]:
		"""
		Load gambar dan ekstrak patches
        
		Args:
			image_path: Path ke file gambar
			normalize: Apakah akan dinormalisasi ke [0, 1]
            
		Returns:
			List of patches
		"""
		# Load image
		img = Image.open(image_path)
		img_array = np.array(img)
        
		# Normalisasi jika diminta
		if normalize and img_array.dtype == np.uint8:
			img_array = img_array.astype(np.float32) / 255.0
            
		# Ekstrak patches
		patches = self.extract_patches(img_array)
		
		logger.info(f"Loaded {image_path} and extracted {len(patches)} patches")
        
		return patches
	
	def save_patches_as_png(self, patches: List[np.ndarray], 
							output_dir: str, 
							patch_type: str = 'image',
							normalize: bool = True) -> None:
		"""
		Simpan patches sebagai file PNG individual
		
		Args:
			patches: List of patch arrays
			output_dir: Directory untuk menyimpan patches
			patch_type: Tipe patch ('image' atau 'ground')
			normalize: Apakah data sudah dinormalisasi [0,1]
		"""
		# Create output directory
		patch_dir = os.path.join(output_dir, f'{patch_type}_patches_png')
		Path(patch_dir).mkdir(parents=True, exist_ok=True)
		
		logger.info(f"Saving {len(patches)} {patch_type} patches as PNG to {patch_dir}")
		
		for idx, patch in enumerate(patches):
			# Convert to uint8 for PNG
			if normalize:
				# Assume normalized to [0, 1]
				patch_uint8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
			else:
				# Assume already in [0, 255]
				patch_uint8 = np.clip(patch, 0, 255).astype(np.uint8)
			
			# Handle different patch dimensions
			if len(patch.shape) == 2:
				# Grayscale patch
				img = Image.fromarray(patch_uint8, mode='L')
			elif len(patch.shape) == 3 and patch.shape[2] == 3:
				# RGB patch
				img = Image.fromarray(patch_uint8, mode='RGB')
			elif len(patch.shape) == 3 and patch.shape[2] == 4:
				# RGBA patch
				img = Image.fromarray(patch_uint8, mode='RGBA')
			else:
				# Try to handle other formats
				logger.warning(f"Unusual patch shape: {patch.shape}, attempting conversion")
				img = Image.fromarray(patch_uint8)
			
			# Save patch
			filename = f'{patch_type}_patch_{idx:04d}.png'
			filepath = os.path.join(patch_dir, filename)
			img.save(filepath)
			
			if (idx + 1) % 50 == 0 or idx == 0:
				logger.debug(f"Saved patch {idx + 1}/{len(patches)}")
		
		logger.info(f"Successfully saved all {len(patches)} {patch_type} patches")
    
	def process_dataset(self, 
					   image_dir: str, 
					   ground_dir: str,
					   save_dir: str = None,
					   save_png: bool = True) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Process seluruh dataset dengan patching
        
		Args:
			image_dir: Directory berisi gambar input
			ground_dir: Directory berisi ground truth
			save_dir: Directory untuk menyimpan patches (optional)
			save_png: Apakah menyimpan patches sebagai PNG (default: True)
            
		Returns:
			Tuple of (image_patches, ground_patches)
		"""
		# Get list of files
		image_files = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
		ground_files = sorted(glob.glob(os.path.join(ground_dir, '*.tif')))
        
		logger.info(f"Found {len(image_files)} image files")
		logger.info(f"Found {len(ground_files)} ground truth files")
        
		all_image_patches = []
		all_ground_patches = []
        
		for img_path, gnd_path in zip(image_files, ground_files):
			logger.info(f"Processing: {os.path.basename(img_path)}")
            
			# Extract patches dari image
			img_patches = self.load_and_patch_image(img_path, normalize=True)
            
			# Extract patches dari ground truth
			gnd_patches = self.load_and_patch_image(gnd_path, normalize=False)
            
			all_image_patches.extend(img_patches)
			all_ground_patches.extend(gnd_patches)
            
			logger.info(f"  Extracted {len(img_patches)} patches")
        
		# Convert to numpy arrays
		image_patches = np.array(all_image_patches)
		ground_patches = np.array(all_ground_patches)
        
		logger.info(f"\nTotal patches: {len(image_patches)}")
		logger.info(f"Image patches shape: {image_patches.shape}")
		logger.info(f"Ground patches shape: {ground_patches.shape}")
        
		# Save jika diminta
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			
			# Save as numpy arrays
			logger.info(f"Saving patches as NPY files...")
			np.save(os.path.join(save_dir, 'image_patches.npy'), image_patches)
			np.save(os.path.join(save_dir, 'ground_patches.npy'), ground_patches)
			logger.info(f"NPY files saved to {save_dir}")
			
			# Save as PNG files if requested
			if save_png:
				logger.info(f"Saving patches as PNG files...")
				self.save_patches_as_png(
					all_image_patches, 
					save_dir, 
					patch_type='image',
					normalize=True
				)
				self.save_patches_as_png(
					all_ground_patches,
					save_dir,
					patch_type='ground',
					normalize=False
				)
				logger.info(f"PNG files saved to {save_dir}")
        
		return image_patches, ground_patches


def prepare_segmentation_data(ground_truth: np.ndarray, 
							  num_classes: int) -> np.ndarray:
	"""
	Prepare ground truth untuk segmentasi (one-hot encoding)
    
	Args:
		ground_truth: Array dengan shape (N, H, W) atau (N, H, W, C)
		num_classes: Jumlah kelas untuk segmentasi
        
	Returns:
		One-hot encoded array dengan shape (N, H, W, num_classes)
	"""
	logger.info(f"Preparing segmentation data with {num_classes} classes")
	
	# Jika ground truth adalah RGB, convert ke grayscale label
	if len(ground_truth.shape) == 4:
		# Asumsi: menggunakan channel pertama atau konversi RGB ke label
		ground_truth = ground_truth[:, :, :, 0]
    
	# Normalisasi nilai ke range [0, num_classes-1]
	unique_values = np.unique(ground_truth)
	logger.info(f"Unique values in ground truth: {unique_values}")
    
	# Map nilai ke class indices
	gt_normalized = np.zeros_like(ground_truth, dtype=np.int32)
	for idx, val in enumerate(sorted(unique_values)):
		gt_normalized[ground_truth == val] = idx % num_classes
    
	# One-hot encoding
	n, h, w = gt_normalized.shape
	one_hot = np.zeros((n, h, w, num_classes), dtype=np.float32)
    
	for i in range(num_classes):
		one_hot[:, :, :, i] = (gt_normalized == i).astype(np.float32)
	
	logger.info(f"One-hot encoding completed. Shape: {one_hot.shape}")
    
	return one_hot


def split_data(images: np.ndarray, 
			   labels: np.ndarray,
			   train_ratio: float = 0.7,
			   val_ratio: float = 0.15,
			   test_ratio: float = 0.15,
			   shuffle: bool = True,
			   random_seed: int = 42) -> Tuple:
	"""
	Split data menjadi train, validation, dan test sets
    
	Args:
		images: Array gambar
		labels: Array labels
		train_ratio: Rasio data training
		val_ratio: Rasio data validation
		test_ratio: Rasio data testing
		shuffle: Apakah akan diacak
		random_seed: Random seed untuk reproducibility
        
	Returns:
		Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
	"""
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
		"Ratios must sum to 1.0"
	
	logger.info(f"Splitting data with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
	n_samples = len(images)
	indices = np.arange(n_samples)
    
	if shuffle:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
		logger.info(f"Data shuffled with seed={random_seed}")
    
	# Calculate split points
	train_end = int(n_samples * train_ratio)
	val_end = train_end + int(n_samples * val_ratio)
    
	# Split indices
	train_idx = indices[:train_end]
	val_idx = indices[train_end:val_end]
	test_idx = indices[val_end:]
    
	# Split data
	X_train = images[train_idx]
	y_train = labels[train_idx]
    
	X_val = images[val_idx]
	y_val = labels[val_idx]
    
	X_test = images[test_idx]
	y_test = labels[test_idx]
    
	logger.info(f"Data split completed:")
	logger.info(f"  Training: {len(X_train)} samples")
	logger.info(f"  Validation: {len(X_val)} samples")
	logger.info(f"  Testing: {len(X_test)} samples")
    
	return X_train, y_train, X_val, y_val, X_test, y_test

