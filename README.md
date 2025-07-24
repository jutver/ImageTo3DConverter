# 2D to 3D Model Converter

This project is only meant for experimental and educational purpose only.

## Overview

This is a Flask-based web application that converts 2D images into 3D voxel models using deep learning. The system employs a ResNet-based neural network to predict 3D voxel grids from input images and provides a modern web interface with real-time 3D visualization using Three.js.

## Project Structure

```
ImageTo3DConverter/
├── README.md                   # Project documentation
├── app.py                      # Main Flask application
├── inference_service.py        # ML inference service
├── main.py                     # Application entry point
├── outputs/                    # Generated 3D models
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
├── resnet_voxel_model.pth     # Trained ML model
├── static/                    # Frontend assets
│   ├── css/
│   │   └── styles.css         # Application styles
│   ├── index.html             # Main web interface
│   └── js/
│       ├── app.js             # React application logic
│       └── three-viewer.js    # 3D visualization component
├── train/                     # Model training pipeline
│   ├── checker.py             # Data validation utilities
│   ├── convert_mat_to_npy.py  # MATLAB to NumPy converter
│   ├── data/                  # Training data directory
│   └── train.py               # Main training script
└── uploads/                   # User uploaded images
```

## System Architecture

### Frontend Architecture
- **Technology**: React (via CDN) with Three.js for 3D visualization
- **Styling**: CSS with glass morphism design
- **Architecture Pattern**: Component-based React application with hooks for state management
- **3D Rendering**: Custom Three.js viewer class for model visualization with orbit controls

### Backend Architecture
- **Framework**: Flask with CORS enabled for API access
- **Architecture Pattern**: Service-oriented with separate inference service
- **Processing Model**: Asynchronous processing with status polling
- **File Handling**: Secure file upload with validation and organized storage

### Machine Learning Pipeline
- **Model**: Custom ResNet18-based encoder-decoder architecture
- **Input**: 224x224 RGB images with ImageNet normalization
- **Output**: 32x32x32 voxel grids converted to 3D mesh format
- **Inference**: GPU-accelerated when available, CPU fallback

## Key Components

### Core Application (`app.py`)
- Flask web server with file upload handling
- CORS configuration for frontend API access
- Secure filename handling and file type validation
- Asynchronous processing status management
- Static file serving for frontend assets

### Inference Service (`inference_service.py`)
- ResNet-based voxel prediction model
- Image preprocessing and normalization
- 3D mesh generation from voxel grids
- Device-agnostic processing (CUDA/CPU)
- Model loading and caching

### Frontend Components
- **Main App**: React-based file upload interface with drag-and-drop
- **3D Viewer**: Three.js-based model visualization with lighting and controls
- **UI Elements**: Progress tracking, status updates, and error handling

### Model Architecture
- **Encoder**: Modified ResNet18 (pretrained, FC layer removed)
- **Decoder**: Fully connected layers generating 32³ voxel predictions
- **Activation**: Sigmoid for voxel occupancy probability

## Model Training

The project includes a complete training pipeline in the `train/` directory for retraining or fine-tuning the ResNet-based voxel prediction model.

### Dataset
This project uses the **Pix3D dataset** for training and evaluation. Pix3D is a large-scale benchmark dataset for single-image 3D shape modeling.

- **Source**: [Pix3D GitHub Repository](https://github.com/xingyuansun/pix3d)
- **Description**: Contains 395 3D shapes across 9 categories with corresponding 2D images
- **Categories**: bed, bookcase, chair, desk, misc, sofa, table, tool, wardrobe
- **Format**: RGB images paired with 3D voxel grids and mesh models

### Training Components

#### Data Preprocessing (`convert_mat_to_npy.py`)
- Converts MATLAB `.mat` voxel files to NumPy `.npy` format for efficient loading
- Processes Pix3D dataset structure with automatic directory creation
- Handles batch conversion of voxel data with error logging

#### Training Pipeline (`train.py`)
- **Dataset Class**: Custom `Pix3DDataset` for loading image-voxel pairs
- **Model Architecture**: ResNet18 encoder with custom decoder layers
- **Loss Function**: Binary Cross-Entropy for voxel occupancy prediction
- **Optimization**: Adam optimizer with configurable learning rate
- **Validation**: Comprehensive metrics including MAE, MSE, and R² score
- **Checkpointing**: Automatic model saving with best validation loss tracking

#### Data Validation (`checker.py`)
- Validates dataset integrity and file accessibility
- Checks for missing or corrupted image-voxel pairs
- Provides dataset statistics and quality reports

### Training Configuration
- **Input Resolution**: 224×224 RGB images with ImageNet normalization
- **Voxel Resolution**: 32×32×32 binary occupancy grids
- **Batch Size**: Configurable (default: 16)
- **Learning Rate**: Configurable with scheduler support
- **Device Support**: Automatic CUDA/CPU detection

### Training Data Structure
The training pipeline expects data organized in the following structure:
```
train/data/
├── pix3d.json              # Dataset metadata and file mappings
├── img/                    # Input images organized by category
│   ├── bed/
│   ├── chair/
│   └── ...
├── voxel/                  # Original MATLAB voxel files
└── voxel_npy/             # Converted NumPy voxel files
    ├── bed/
    ├── chair/
    └── ...
```

### Usage
1. **Data Preparation**: Place Pix3D dataset in `train/data/`
2. **Convert Voxels**: Run `convert_mat_to_npy.py` to prepare voxel data
3. **Validate Data**: Use `checker.py` to verify dataset integrity
4. **Train Model**: Execute `train.py` with desired hyperparameters

## Data Flow

1. **Image Upload**: User uploads image through React frontend
2. **File Validation**: Backend validates file type and size (16MB max)
3. **Processing Initiation**: Inference service processes image asynchronously
4. **Model Prediction**: ResNet model generates voxel grid from preprocessed image
5. **Mesh Conversion**: Voxel grid converted to 3D mesh format
6. **Status Polling**: Frontend polls processing status every 2 seconds
7. **Visualization**: Completed model rendered in Three.js viewer

## External Dependencies

### Python Backend
- **Flask**: Web framework and routing
- **PyTorch**: Deep learning inference engine
- **OpenCV**: Image preprocessing and manipulation
- **Open3D**: 3D geometry processing and mesh generation
- **NumPy**: Numerical computations

### Frontend Libraries
- **React**: UI framework (CDN-based)
- **Three.js**: 3D graphics and WebGL rendering
- **Axios**: HTTP client for API communication
- **Font Awesome**: Icon library

### Model Dependencies
- **Torchvision**: Pretrained ResNet18 and image transforms
- **CUDA**: Optional GPU acceleration support

## Pre-trained Model

The trained ResNet-based voxel prediction model is available for download:

- **Download Link**: [Pre-trained Model on MEGA](https://mega.nz/folder/tqVTyRgY#t7U8RfDSE5N6wg__lpVszA)
- **File**: `resnet_voxel_model.pth`
- **Description**: ResNet18-based encoder-decoder model trained on Pix3D dataset
- **Usage**: Place the downloaded model file in the project root directory

The model file should be placed as `ImageTo3DConverter/resnet_voxel_model.pth` for the inference service to load automatically.
