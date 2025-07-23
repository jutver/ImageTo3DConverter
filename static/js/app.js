const { useState, useEffect, useRef } = React;

function App() {
    const [file, setFile] = useState(null);
    const [fileId, setFileId] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [processingStatus, setProcessingStatus] = useState(null);
    const [error, setError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [modelUrl, setModelUrl] = useState(null);
    
    const fileInputRef = useRef(null);
    const viewerContainerRef = useRef(null);
    const threeViewerRef = useRef(null);
    
    // Initialize Three.js viewer
    useEffect(() => {
        if (viewerContainerRef.current && !threeViewerRef.current) {
            threeViewerRef.current = new ThreeViewer(viewerContainerRef.current);
        }
        
        return () => {
            if (threeViewerRef.current) {
                threeViewerRef.current.dispose();
                threeViewerRef.current = null;
            }
        };
    }, []);
    
    // Poll processing status
    useEffect(() => {
        if (fileId && processingStatus?.status === 'processing') {
            const interval = setInterval(async () => {
                try {
                    const response = await axios.get(`/api/status/${fileId}`);
                    setProcessingStatus(response.data);
                    
                    if (response.data.status === 'completed') {
                        setModelUrl(`/api/model/${fileId}`);
                        clearInterval(interval);
                    } else if (response.data.status === 'error') {
                        setError(response.data.message);
                        clearInterval(interval);
                    }
                } catch (err) {
                    console.error('Status check failed:', err);
                    clearInterval(interval);
                }
            }, 2000);
            
            return () => clearInterval(interval);
        }
    }, [fileId, processingStatus?.status]);
    
    // Load model when URL is available
    useEffect(() => {
        if (modelUrl && threeViewerRef.current) {
            threeViewerRef.current.loadModel(modelUrl)
                .then(() => {
                    console.log('Model loaded successfully');
                })
                .catch((err) => {
                    console.error('Failed to load model:', err);
                    setError('Failed to load the generated 3D model');
                });
        }
    }, [modelUrl]);
    
    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && isValidImageFile(droppedFile)) {
            setFile(droppedFile);
            setError(null);
        } else {
            setError('Please drop a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP)');
        }
    };
    
    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };
    
    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragOver(false);
    };
    
    const handleFileSelect = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && isValidImageFile(selectedFile)) {
            setFile(selectedFile);
            setError(null);
        } else {
            setError('Please select a valid image file');
        }
    };
    
    const isValidImageFile = (file) => {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
        return validTypes.includes(file.type);
    };
    
    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    const uploadFile = async () => {
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            setError(null);
            setUploadProgress(0);
            
            const response = await axios.post('/api/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(progress);
                }
            });
            
            setFileId(response.data.file_id);
            setUploadProgress(100);
            
        } catch (err) {
            console.error('Upload failed:', err);
            setError(err.response?.data?.error || 'Upload failed');
            setUploadProgress(0);
        }
    };
    
    const processImage = async () => {
        if (!fileId) return;
        
        try {
            setError(null);
            setProcessingStatus({ status: 'processing', progress: 0, message: 'Starting...' });
            
            const response = await axios.post('/api/process', { file_id: fileId });
            console.log('Processing started:', response.data);
            
        } catch (err) {
            console.error('Processing failed:', err);
            setError(err.response?.data?.error || 'Processing failed');
            setProcessingStatus(null);
        }
    };
    
    const downloadModel = () => {
        if (fileId) {
            window.open(`/api/download/${fileId}`, '_blank');
        }
    };
    
    const resetApp = () => {
        setFile(null);
        setFileId(null);
        setUploadProgress(0);
        setProcessingStatus(null);
        setError(null);
        setModelUrl(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };
    
    const resetView = () => {
        if (threeViewerRef.current) {
            threeViewerRef.current.resetView();
        }
    };
    
    const toggleWireframe = () => {
        if (threeViewerRef.current) {
            threeViewerRef.current.setWireframe(!threeViewerRef.current.wireframe);
            threeViewerRef.current.wireframe = !threeViewerRef.current.wireframe;
        }
    };
    
    return (
        <div className="container">
            <div className="header glass">
                <h1>
                    <i className="fas fa-cube"></i> 2D to 3D Converter
                </h1>
                <p>Transform your 2D images into stunning 3D models using AI</p>
            </div>
            
            <div className="main-content">
                {/* Upload Section */}
                <div className="upload-section glass">
                    <h2 style={{color: 'white', marginBottom: '20px', textAlign: 'center'}}>
                        <i className="fas fa-upload"></i> Upload Image
                    </h2>
                    
                    {!file ? (
                        <div 
                            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <i className="fas fa-cloud-upload-alt upload-icon"></i>
                            <div className="upload-text">
                                Drag & drop your image here
                            </div>
                            <div className="upload-hint">
                                or click to browse (PNG, JPG, GIF, BMP, WebP)
                            </div>
                            <input 
                                ref={fileInputRef}
                                type="file" 
                                className="hidden"
                                accept="image/*"
                                onChange={handleFileSelect}
                            />
                        </div>
                    ) : (
                        <div className="image-preview fade-in">
                            <img 
                                src={URL.createObjectURL(file)} 
                                alt="Preview" 
                                className="preview-image"
                            />
                            <div className="image-info">
                                <strong>{file.name}</strong><br />
                                {formatFileSize(file.size)}
                            </div>
                            
                            {uploadProgress > 0 && uploadProgress < 100 && (
                                <div className="progress-container">
                                    <div className="progress-bar">
                                        <div 
                                            className="progress-fill" 
                                            style={{width: `${uploadProgress}%`}}
                                        ></div>
                                    </div>
                                    <div className="status-message">Uploading... {uploadProgress}%</div>
                                </div>
                            )}
                            
                            {/* Processing Status - Show only when processing */}
                            {processingStatus?.status === 'processing' && (
                                <div className="progress-container fade-in" style={{marginTop: '20px'}}>
                                    <div className="pulse" style={{textAlign: 'center', marginBottom: '15px'}}>
                                        <i className="fas fa-spinner fa-spin" style={{fontSize: '2rem', color: '#4facfe', marginBottom: '10px'}}></i>
                                    </div>
                                    <div className="progress-bar">
                                        <div 
                                            className="progress-fill" 
                                            style={{width: `${processingStatus.progress || 0}%`}}
                                        ></div>
                                    </div>
                                    <div className="status-message" style={{textAlign: 'center', marginTop: '10px'}}>
                                        {processingStatus.message}
                                    </div>
                                </div>
                            )}
                            
                            {/* Success Status - Show when completed */}
                            {processingStatus?.status === 'completed' && (
                                <div className="fade-in" style={{marginTop: '20px', textAlign: 'center'}}>
                                    <div style={{marginBottom: '15px'}}>
                                        <i className="fas fa-check-circle" style={{fontSize: '2rem', color: '#51cf66', marginBottom: '10px'}}></i>
                                        <div className="success-message">
                                            {processingStatus.message}
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            {/* Error Status - Show when error */}
                            {processingStatus?.status === 'error' && (
                                <div className="fade-in" style={{marginTop: '20px', textAlign: 'center'}}>
                                    <div className="error-message">
                                        <i className="fas fa-exclamation-circle"></i>
                                        {processingStatus.message}
                                    </div>
                                </div>
                            )}

                            {/* Action Buttons - Hide during processing */}
                            {processingStatus?.status !== 'processing' && (
                                <div style={{marginTop: '20px', textAlign: 'center'}}>
                                    {!fileId ? (
                                        <button 
                                            className="btn btn-primary"
                                            onClick={uploadFile}
                                            disabled={uploadProgress > 0 && uploadProgress < 100}
                                        >
                                            {uploadProgress > 0 && uploadProgress < 100 ? (
                                                <>
                                                    <div className="loading-spinner"></div>
                                                    Uploading...
                                                </>
                                            ) : (
                                                <>
                                                    <i className="fas fa-upload"></i>
                                                    Upload Image
                                                </>
                                            )}
                                        </button>
                                    ) : (
                                        <>
                                            <button 
                                                className="btn btn-primary"
                                                onClick={processImage}
                                                disabled={processingStatus?.status === 'processing'}
                                            >
                                                <i className="fas fa-magic"></i>
                                                Generate 3D Model
                                            </button>
                                            
                                            {processingStatus?.status === 'completed' && (
                                                <button 
                                                    className="btn btn-primary"
                                                    onClick={downloadModel}
                                                    style={{marginLeft: '10px'}}
                                                >
                                                    <i className="fas fa-download"></i>
                                                    Download Model
                                                </button>
                                            )}
                                        </>
                                    )}
                                    
                                    <button 
                                        className="btn btn-secondary"
                                        onClick={resetApp}
                                        style={{marginLeft: '10px'}}
                                    >
                                        <i className="fas fa-refresh"></i>
                                        Reset
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
                
            </div>
            
            {/* 3D Model Viewer Section */}
            <div className="viewer-section glass">
                <h2 style={{color: 'white', marginBottom: '20px', textAlign: 'center'}}>
                    <i className="fas fa-cube"></i> 3D Model Viewer
                </h2>
                
                <div className="viewer-container" ref={viewerContainerRef}>
                    {!modelUrl && (
                        <div className="viewer-placeholder">
                            <i className="fas fa-cube"></i>
                            <div>Your 3D model will appear here</div>
                        </div>
                    )}
                </div>
                
                {modelUrl && (
                    <div className="viewer-controls">
                        <button className="btn btn-secondary" onClick={resetView}>
                            <i className="fas fa-home"></i>
                            Reset View
                        </button>
                        <button className="btn btn-secondary" onClick={toggleWireframe}>
                            <i className="fas fa-border-all"></i>
                            Toggle Wireframe
                        </button>
                    </div>
                )}
            </div>

            
            {/* Error Display */}
            {error && (
                <div className="glass" style={{padding: '20px', marginTop: '20px'}}>
                    <div className="error-message fade-in">
                        <i className="fas fa-exclamation-circle"></i>
                        {error}
                    </div>
                </div>
            )}
        </div>
    );
}

// Render the app
ReactDOM.render(<App />, document.getElementById('root'));
