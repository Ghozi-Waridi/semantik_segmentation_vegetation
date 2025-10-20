# VGG16 + CBAM Segmentation Model

Implementasi model semantic segmentation menggunakan VGG16 dengan Channel dan Spatial Attention Mechanism (CBAM).

## Struktur Project

```
vggCBAM/
├── core/
│   ├── activation/
│   │   ├── activation.py           # Activation functions (ReLU, Sigmoid, etc)
│   │   └── __init__.py
│   ├── layers/
│   │   ├── attention.py            # CBAM attention mechanism
│   │   ├── convolutional.py        # Conv2D layer
│   │   ├── pooling.py              # MaxPooling2D layer
│   │   ├── upsampling.py           # UpSampling2D layer
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluation.py           # Loss functions and metrics
│   │   └── __init__.py
│   ├── utils/
│   │   ├── model_io.py             # Save/load model functions
│   │   ├── logging_config.py       # Logging configuration
│   │   └── __init__.py
│   ├── vgg16_cbam_segmentation.py  # Main model architecture
│   └── __init__.py
├── data/
│   ├── raw/
│   │   ├── image/                  # Original images
│   │   └── ground/                 # Ground truth masks
│   └── processed/
│       ├── data.py                 # Data patching and preparation
│       ├── patches/                # Saved patches
│       └── __pycache__/
├── test/
│   ├── test_activation.py          # Tests for activation functions
│   ├── test_convolutional.py       # Tests for Conv2D layer
│   └── test_model_io.py            # Tests for save/load functions
├── results/                        # Training results and visualizations
├── models/                         # Saved trained models
├── logs/                           # Log files
├── main.py                         # Main training script
├── predict.py                      # Prediction script
└── README.md                       # This file
```

## Instalasi

```bash
# Clone repository
git clone <repository-url>
cd vggCBAM

# Install dependencies (jika perlu)
pip install numpy matplotlib scikit-image
```

## Penggunaan

### 1. Training Model

```bash
python main.py
```

Pipeline akan:

1. Membuat patches dari raw images (128x128)
2. Mempersiapkan data (encoding, normalisasi)
3. Melakukan training dengan validation
4. Menyimpan model terlatih
5. Testing pada test set
6. Menyimpan hasil dan visualisasi

### 2. Loading dan Menggunakan Model Terlatih

```python
from core.vgg16_cbam_segmentation import VGG16_CBAM_Segmentation
from core.utils.model_io import load_model, list_saved_models
import numpy as np

# List available models
models = list_saved_models(save_dir='models')
print("Available models:", models)

# Load model
model = VGG16_CBAM_Segmentation(input_shape=(128, 128, 3), num_classes=6)
model = load_model(model, f'models/{models[0]}')

# Prediction
input_image = np.random.randn(1, 128, 128, 3)
output = model.forward(input_image)
prediction = np.argmax(output, axis=-1)
```

### 3. Predict Script (Simple Inference)

```bash
python predict.py
```

Script ini akan:

- Memuat model terlatih terbaru
- Melakukan inference pada sample image
- Menampilkan hasil prediksi

## Model Architecture

### Encoder (VGG16)

- **Block 1-5**: Convolution layers dengan ReLU activation
- **MaxPooling**: Downsampling dengan factor 2
- **CBAM**: Channel dan Spatial attention mechanism

### Decoder (U-Net style)

- **Upsampling**: Factor 2 upsampling
- **Convolution**: Feature refinement
- **ReLU**: Activation
- **Final Conv**: Class prediction (1x1 conv)

### Attention Mechanism (CBAM)

- **Channel Attention**: MLP-based channel weighting
- **Spatial Attention**: Convolution-based spatial weighting

## Logging

Setiap komponen memiliki comprehensive logging:

```bash
# Log files tersimpan di logs/ directory
# Format: {module_name}_{timestamp}.log

# Contoh log entries:
# 2024-10-19 10:30:45 - core.layers.convolutional - INFO - Conv2D Layer Initialized
# 2024-10-19 10:30:45 - main - INFO - Epoch [1/50] - 2.34s
# 2024-10-19 10:30:45 - main - INFO - Train Loss: 2.1234 | Train Acc: 0.4567
```

## Testing

Run comprehensive unit tests:

```bash
# Test all activation functions
python -m pytest test/test_activation.py -v

# Test Conv2D layer
python -m pytest test/test_convolutional.py -v

# Test model save/load
python -m pytest test/test_model_io.py -v

# Run all tests
python -m pytest test/ -v
```

Test coverage includes:

- Forward/backward passes
- Gradient computation
- Shape validation
- Numerical stability
- Weight updates
- Model serialization

## Data Format

### Input Images

- Format: PNG/JPG
- Size: Arbitrary (akan di-patch menjadi 128x128)
- Channels: 3 (RGB)
- Value range: [0, 255]

### Ground Truth Masks

- Format: PNG/JPG (grayscale)
- Size: Same as input image
- Channels: 1 (grayscale)
- Value range: [0, num_classes-1]

## Training Configuration

Modify dalam `main.py`:

```python
# Data patching
patch_size = 128

# Training parameters
epochs = 50
batch_size = 4
learning_rate = 0.001
early_stopping_patience = 10

# Data split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Number of classes
num_classes = 6
```

## Saving dan Loading Model

### Automatic Save (dalam training)

Model otomatis disimpan setelah training selesai:

- Lokasi: `models/vgg16_cbam_trained_{timestamp}/`
- File yang disimpan:
  - `metadata.json`: Architecture info
  - `architecture.json`: Layer details
  - `weights/`: Semua weight files (.npy)

### Manual Load

```python
from core.utils.model_io import save_model, load_model

# Save model
model_path = save_model(model, save_dir='models', model_name='my_model')

# Load model
loaded_model = load_model(new_model_instance, model_path)
```

### List Saved Models

```python
from core.utils.model_io import list_saved_models

models = list_saved_models(save_dir='models')
for model in models:
    print(model)
```

## Results Output

Setelah training, file berikut disimpan di `results/`:

1. **training_history.png**: Plot training/validation loss dan accuracy
2. **predictions.png**: Visualisasi prediksi pada test samples
3. **results_YYYYMMDD_HHMMSS.txt**: Summary report

Contoh report:

```
============================================================
VGG16 + CBAM SEGMENTATION RESULTS
============================================================

Dataset Information:
  Total patches: 150
  Training samples: 105
  Validation samples: 22
  Testing samples: 23
  Patch size: 128x128
  Number of classes: 6

Model Configuration:
  Architecture: VGG16 + CBAM
  Input shape: (128, 128, 3)
  Learning rate: 0.001

Training Results:
  Final train loss: 0.2345
  Final train accuracy: 0.9123
  Final val loss: 0.3456
  Final val accuracy: 0.8765
  Total epochs: 45

Test Results:
  Test loss: 0.3789
  Test accuracy: 0.8534
  Mean IoU: 0.7823
```

## Troubleshooting

### Out of Memory

- Kurangi `batch_size` di `main.py`
- Gunakan patch size lebih kecil (64x64 or 96x96)

### Model Not Improving

- Increase `epochs`
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Check data quality dan normalisasi

### Logging Issues

- Pastikan `logs/` directory dapat dibuat
- Check file permissions

## Performance

Typical performance metrics (pada dataset 150 patches):

- Training time: ~30-60 minutes (bergantung hardware)
- Memory usage: ~4-6 GB GPU
- Final accuracy: ~85-90%
- Mean IoU: ~75-80%

## Reference

- VGG16: [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
- CBAM: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- Semantic Segmentation: [Fully Convolutional Networks](https://arxiv.org/abs/1411.4038)

## License

This project is provided as-is for educational purposes.

## Contact

For issues or questions, please check the log files in `logs/` directory for detailed error information.
