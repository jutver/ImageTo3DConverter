# 2D to 3D Model Converter

## Overview

This is a Flask-based web application that converts 2D images into 3D voxel models using deep learning. The system employs a ResNet-based neural network to predict 3D voxel grids from input images and provides a modern web interface with real-time 3D visualization using Three.js.


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

## Deployment Strategy

### Development Setup
- Flask development server on port 5000
- Debug mode enabled for development
- Local file storage in uploads/ and outputs/ directories

### Production Considerations
- Environment-based secret key configuration
- ProxyFix middleware for reverse proxy deployment
- CORS enabled for cross-origin requests
- Configurable upload limits and storage paths

### File Organization
- Static assets served from /static/ directory
- Uploaded files stored in /uploads/
- Generated models stored in /outputs/
- Secure filename handling prevents directory traversal

### Security Features
- File type validation (images only)
- File size limits (16MB maximum)
- Secure filename sanitization
- Environment-based secret key management