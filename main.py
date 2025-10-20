import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from core.vgg16_cbam_segmentation import VGG16_CBAM_Segmentation
from core.evaluation.evaluation import categorical_crossentropy
from core.utils.model_io import save_model, load_model, list_saved_models
from core.utils.logging_config import setup_logging
from core.utils.gpu_utils import (
    print_gpu_info, 
    is_gpu_available, 
    get_backend, 
    to_gpu, 
    to_cpu,
    empty_cache,
    synchronize,
    xp
)
from data.processed.data import (
    DataPatcher,
    prepare_segmentation_data,
    split_data,
)

# Setup logging
logger = setup_logging(module_name='main', log_dir='logs')


class SegmentationTrainer:
	"""Class untuk training, validation, dan testing model segmentasi"""
    
	def __init__(self, model, learning_rate=0.001):
		self.model = model
		self.learning_rate = learning_rate
		self.train_losses = []
		self.val_losses = []
		self.train_accuracies = []
		self.val_accuracies = []
        
	def calculate_accuracy(self, y_true, y_pred):
		"""Calculate pixel-wise accuracy with GPU support"""
		y_true_labels = xp.argmax(y_true, axis=-1)
		y_pred_labels = xp.argmax(y_pred, axis=-1)
		accuracy = float(xp.mean(y_true_labels == y_pred_labels))
		return accuracy
    
	def calculate_iou(self, y_true, y_pred, num_classes):
		"""Calculate mean Intersection over Union (IoU) with GPU support"""
		y_true_labels = xp.argmax(y_true, axis=-1)
		y_pred_labels = xp.argmax(y_pred, axis=-1)
        
		ious = []
		for cls in range(num_classes):
			true_cls = (y_true_labels == cls)
			pred_cls = (y_pred_labels == cls)
            
			intersection = float(xp.sum(xp.logical_and(true_cls, pred_cls)))
			union = float(xp.sum(xp.logical_or(true_cls, pred_cls)))
            
			if union == 0:
				iou = 1.0 if intersection == 0 else 0.0
			else:
				iou = intersection / union
			ious.append(iou)
        
		return np.mean(ious)
    
	def train_epoch(self, X_train, y_train, batch_size=8):
		"""Train untuk satu epoch with GPU support"""
		n_samples = len(X_train)
		indices = np.random.permutation(n_samples)
        
		epoch_loss = 0.0
		epoch_acc = 0.0
		n_batches = 0
        
		for start_idx in tqdm(range(0, n_samples, batch_size), desc="train", leave=False):
			end_idx = min(start_idx + batch_size, n_samples)
			batch_indices = indices[start_idx:end_idx]
            
			# Transfer data to GPU
			X_batch = to_gpu(X_train[batch_indices])
			y_batch = to_gpu(y_train[batch_indices])
            
			# Forward pass
			y_pred = self.model.forward(X_batch)
            
			# Calculate loss
			loss = categorical_crossentropy(y_batch, y_pred)
            
			# Calculate accuracy
			acc = self.calculate_accuracy(y_batch, y_pred)
            
			# Backward pass
			self.model.backward(y_batch, self.learning_rate)
            
			epoch_loss += loss
			epoch_acc += acc
			n_batches += 1
			
			# Synchronize GPU operations
			synchronize()
        
		return epoch_loss / n_batches, epoch_acc / n_batches
    
	def validate(self, X_val, y_val, batch_size=8):
		"""Validasi model with GPU support"""
		n_samples = len(X_val)
        
		val_loss = 0.0
		val_acc = 0.0
		n_batches = 0
        
		for start_idx in tqdm(range(0, n_samples, batch_size), desc="val", leave=False):
			end_idx = min(start_idx + batch_size, n_samples)
            
			# Transfer data to GPU
			X_batch = to_gpu(X_val[start_idx:end_idx])
			y_batch = to_gpu(y_val[start_idx:end_idx])
            
			# Forward pass only
			y_pred = self.model.forward(X_batch)
            
			# Calculate loss
			loss = categorical_crossentropy(y_batch, y_pred)
            
			# Calculate accuracy
			acc = self.calculate_accuracy(y_batch, y_pred)
            
			val_loss += loss
			val_acc += acc
			n_batches += 1
			
			# Synchronize GPU operations
			synchronize()
        
		return val_loss / n_batches, val_acc / n_batches
    
	def train(self, X_train, y_train, X_val, y_val, 
			  epochs=50, batch_size=8, early_stopping_patience=10):
		"""Training lengkap dengan validation"""
        
		print("="*60)
		print("MEMULAI TRAINING")
		print("="*60)
		print(f"Training samples: {len(X_train)}")
		print(f"Validation samples: {len(X_val)}")
		print(f"Epochs: {epochs}")
		print(f"Batch size: {batch_size}")
		print(f"Learning rate: {self.learning_rate}")
		print("="*60)
		
		# Logging
		logger.info("="*60)
		logger.info("TRAINING STARTED")
		logger.info("="*60)
		logger.info(f"Training samples: {len(X_train)}")
		logger.info(f"Validation samples: {len(X_val)}")
		logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
		logger.info(f"Learning rate: {self.learning_rate}")
        
		best_val_loss = float('inf')
		patience_counter = 0
        
		for epoch in range(epochs):
			start_time = time.time()
            
			# Training
			train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
			# Validation
			val_loss, val_acc = self.validate(X_val, y_val, batch_size)
            
			# Save metrics
			self.train_losses.append(train_loss)
			self.val_losses.append(val_loss)
			self.train_accuracies.append(train_acc)
			self.val_accuracies.append(val_acc)
            
			epoch_time = time.time() - start_time
            
			print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.2f}s")
			print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
			print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
			
			# Log epoch info
			logger.info(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.2f}s")
			logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
			logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
			# Early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				print(f"  ✓ New best validation loss!")
				logger.info(f"  ✓ New best validation loss!")
			else:
				patience_counter += 1
				if patience_counter >= early_stopping_patience:
					print(f"\nEarly stopping triggered after {epoch+1} epochs")
					logger.info(f"Early stopping triggered after {epoch+1} epochs")
					break
        
		print("\n" + "="*60)
		print("TRAINING SELESAI")
		print("="*60)
		logger.info("TRAINING COMPLETED")
		logger.info("="*60)
    
	def test(self, X_test, y_test, batch_size=8):
		"""Testing model dan tampilkan hasil with GPU support"""
        
		print("\n" + "="*60)
		print("MEMULAI TESTING")
		print("="*60)
		print(f"Test samples: {len(X_test)}")
        
		n_samples = len(X_test)
		test_loss = 0.0
		test_acc = 0.0
		test_iou = 0.0
		n_batches = 0
        
		for start_idx in tqdm(range(0, n_samples, batch_size), desc="test", leave=False):
			end_idx = min(start_idx + batch_size, n_samples)
            
			# Transfer data to GPU
			X_batch = to_gpu(X_test[start_idx:end_idx])
			y_batch = to_gpu(y_test[start_idx:end_idx])
            
			# Forward pass
			y_pred = self.model.forward(X_batch)
            
			# Calculate metrics
			loss = categorical_crossentropy(y_batch, y_pred)
			acc = self.calculate_accuracy(y_batch, y_pred)
			iou = self.calculate_iou(y_batch, y_pred, y_batch.shape[-1])
            
			test_loss += loss
			test_acc += acc
			test_iou += iou
			n_batches += 1
			
			# Synchronize GPU operations
			synchronize()
        
		test_loss /= n_batches
		test_acc /= n_batches
		test_iou /= n_batches
        
		print(f"\nTest Results:")
		print(f"  Test Loss: {test_loss:.4f}")
		print(f"  Test Accuracy: {test_acc:.4f}")
		print(f"  Mean IoU: {test_iou:.4f}")
		print("="*60)
        
		return test_loss, test_acc, test_iou
    
	def plot_training_history(self, save_path='results/training_history.png'):
		"""Plot training history"""
		os.makedirs('results', exist_ok=True)
        
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
		# Plot loss
		ax1.plot(self.train_losses, label='Train Loss', marker='o')
		ax1.plot(self.val_losses, label='Val Loss', marker='s')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.set_title('Training and Validation Loss')
		ax1.legend()
		ax1.grid(True)
        
		# Plot accuracy
		ax2.plot(self.train_accuracies, label='Train Accuracy', marker='o')
		ax2.plot(self.val_accuracies, label='Val Accuracy', marker='s')
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Accuracy')
		ax2.set_title('Training and Validation Accuracy')
		ax2.legend()
		ax2.grid(True)
        
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"\nTraining history plot saved to: {save_path}")
		plt.close()
    
    


def main():
	"""Main function untuk menjalankan seluruh pipeline"""
    
	print("\n" + "="*60)
	print("VGG16 + CBAM SEGMENTATION PIPELINE")
	print("="*60)
    
	# =================================================================
	# 1. DATA PATCHING
	# =================================================================
	print("\n[1] DATA PATCHING")
	print("-"*60)
    
	# Konfigurasi paths
	image_dir = 'data/raw/image'
	ground_dir = 'data/raw/ground'
	processed_dir = 'data/processed/patches'
    
	# Buat data patcher
	patcher = DataPatcher(patch_size=128, stride=128)
    
	# Process dataset
	image_patches, ground_patches = patcher.process_dataset(
		image_dir=image_dir,
		ground_dir=ground_dir,
		save_dir=processed_dir
	)
    
	# =================================================================
	# 2. DATA PREPARATION
	# =================================================================
	print("\n[2] DATA PREPARATION")
	print("-"*60)
    
	# Konfigurasi
	num_classes = 6  # Sesuaikan dengan jumlah kelas di dataset Anda
    
	# Prepare ground truth (one-hot encoding)
	ground_patches_encoded = prepare_segmentation_data(ground_patches, num_classes)
    
	# Split data
	X_train, y_train, X_val, y_val, X_test, y_test = split_data(
		images=image_patches,
		labels=ground_patches_encoded,
		train_ratio=0.7,
		val_ratio=0.15,
		test_ratio=0.15,
		shuffle=True,
		random_seed=42
	)
    
	# =================================================================
	# 3. MODEL INITIALIZATION
	# =================================================================
	print("\n[3] MODEL INITIALIZATION")
	print("-"*60)
    
	input_shape = (128, 128, 3)  # Sesuaikan dengan channel gambar
	model = VGG16_CBAM_Segmentation(input_shape=input_shape, num_classes=num_classes)
	print(f"Model initialized: VGG16 + CBAM")
	print(f"Input shape: {input_shape}")
	print(f"Number of classes: {num_classes}")
    
	# =================================================================
	# 4. TRAINING
	# =================================================================
	print("\n[4] TRAINING")
	print("-"*60)
    
	trainer = SegmentationTrainer(model=model, learning_rate=0.001)
    
	trainer.train(
		X_train=X_train,
		y_train=y_train,
		X_val=X_val,
		y_val=y_val,
		epochs=50,
		batch_size=4,  # Kecilkan batch size jika memory terbatas
		early_stopping_patience=10
	)
	
	# Save trained model
	print("\n[4.1] SAVING TRAINED MODEL")
	print("-"*60)
	model_save_path = save_model(model, save_dir='models', model_name='vgg16_cbam_trained')
	print(f"Model saved to: {model_save_path}")
	logger.info(f"Model saved to: {model_save_path}")
    
	# =================================================================
	# 5. TESTING
	# =================================================================
	print("\n[5] TESTING")
	print("-"*60)
    
	test_loss, test_acc, test_iou = trainer.test(
		X_test=X_test,
		y_test=y_test,
		batch_size=4
	)
    
	# =================================================================
	# 6. VISUALIZATION & SAVE RESULTS
	# =================================================================
	print("\n[6] SAVING RESULTS")
	print("-"*60)
    
	# Plot training history
	trainer.plot_training_history(save_path='results/training_history.png')
    
	# Visualize predictions
	trainer.visualize_predictions(
		X_test=X_test,
		y_test=y_test,
		n_samples=5,
		save_path='results/predictions.png'
	)
    
	# Save final results to text file
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_file = f'results/results_{timestamp}.txt'
    
	with open(results_file, 'w') as f:
		f.write("="*60 + "\n")
		f.write("VGG16 + CBAM SEGMENTATION RESULTS\n")
		f.write("="*60 + "\n\n")
		f.write(f"Timestamp: {timestamp}\n\n")
        
		f.write("Dataset Information:\n")
		f.write(f"  Total patches: {len(image_patches)}\n")
		f.write(f"  Training samples: {len(X_train)}\n")
		f.write(f"  Validation samples: {len(X_val)}\n")
		f.write(f"  Testing samples: {len(X_test)}\n")
		f.write(f"  Patch size: 128x128\n")
		f.write(f"  Number of classes: {num_classes}\n\n")
        
		f.write("Model Configuration:\n")
		f.write(f"  Architecture: VGG16 + CBAM\n")
		f.write(f"  Input shape: {input_shape}\n")
		f.write(f"  Learning rate: {trainer.learning_rate}\n\n")
        
		f.write("Training Results:\n")
		f.write(f"  Final train loss: {trainer.train_losses[-1]:.4f}\n")
		f.write(f"  Final train accuracy: {trainer.train_accuracies[-1]:.4f}\n")
		f.write(f"  Final val loss: {trainer.val_losses[-1]:.4f}\n")
		f.write(f"  Final val accuracy: {trainer.val_accuracies[-1]:.4f}\n")
		f.write(f"  Total epochs: {len(trainer.train_losses)}\n\n")
        
		f.write("Test Results:\n")
		f.write(f"  Test loss: {test_loss:.4f}\n")
		f.write(f"  Test accuracy: {test_acc:.4f}\n")
		f.write(f"  Mean IoU: {test_iou:.4f}\n")
		f.write("="*60 + "\n")
		f.write(f"\nModel saved at: {model_save_path}\n")
		f.write(f"To load the model, use:\n")
		f.write(f"  from core.utils.model_io import load_model\n")
		f.write(f"  model = load_model(model, '{model_save_path}')\n")
    
	print(f"Results saved to: {results_file}")
	logger.info(f"Results saved to: {results_file}")
    
	print("\n" + "="*60)
	print("PIPELINE SELESAI!")
	print("="*60)
	print("\nHasil tersimpan di folder 'results/':")
	print("  - training_history.png: Grafik loss dan accuracy")
	print("  - predictions.png: Visualisasi prediksi")
	print(f"  - {results_file}: Ringkasan hasil lengkap")
	print("="*60 + "\n")


if __name__ == "__main__":
	main()



# disetiap file code tambahkan code untuk menyimpan hasil log dan gunakan tqdm untuk memberikan tampilan yang lebh profesional.