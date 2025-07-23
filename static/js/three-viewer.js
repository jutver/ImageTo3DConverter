class ThreeViewer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(5, 5, 5);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Prevent page scroll when mouse wheel is used over the 3D viewer
        this.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();
            event.stopPropagation();
        }, { passive: false });
        
        // Additional mouse event handling for better interaction isolation
        this.container.addEventListener('mouseenter', () => {
            document.body.style.overflow = 'hidden';
        });
        
        this.container.addEventListener('mouseleave', () => {
            document.body.style.overflow = 'auto';
        });
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.25;
        this.controls.enableZoom = true;
        this.controls.target.set(0, 0, 0); // Focus on center of grid
        this.controls.update();
        
        // Lighting
        this.setupLighting();
        
        // Handle resize
        window.addEventListener('resize', this.handleResize.bind(this));
        
        // Start render loop
        this.animate();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Point light
        const pointLight = new THREE.PointLight(0x4facfe, 0.5);
        pointLight.position.set(-5, 5, -5);
        this.scene.add(pointLight);
        
        // Helper grid
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x444444);
        gridHelper.material.opacity = 0.3;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
    }
    
    loadModel(objUrl) {
        return new Promise((resolve, reject) => {
            // Remove existing model
            if (this.model) {
                this.scene.remove(this.model);
                this.model = null;
            }
            
            const loader = new THREE.OBJLoader();
            
            loader.load(
                objUrl,
                (object) => {
                    // Process the loaded object
                    this.model = object;
                    
                    // Apply material to all meshes
                    const material = new THREE.MeshLambertMaterial({
                        color: 0x4facfe,
                        transparent: true,
                        opacity: 0.9
                    });
                    
                    object.traverse((child) => {
                        if (child instanceof THREE.Mesh) {
                            child.material = material;
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }
                    });
                    
                    // Scale the model first if needed
                    const box = new THREE.Box3().setFromObject(object);
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    if (maxDim > 10) {
                        const scale = 10 / maxDim;
                        object.scale.setScalar(scale);
                    }
                    
                    // Recalculate bounding box after scaling
                    const finalBox = new THREE.Box3().setFromObject(object);
                    const center = finalBox.getCenter(new THREE.Vector3());
                    
                    // Apply 90-degree rotation around Y-axis to make models stand upright
                    object.rotation.y = Math.PI / 2; // 90 degrees in radians
                    
                    // Recalculate bounding box after rotation
                    const rotatedBox = new THREE.Box3().setFromObject(object);
                    const rotatedCenter = rotatedBox.getCenter(new THREE.Vector3());
                    
                    // Position the model: center it horizontally and place bottom on grid
                    object.position.set(
                        -rotatedCenter.x,  // Center on X axis
                        -rotatedBox.min.y,  // Bottom of model touches grid (y=0)
                        -rotatedCenter.z   // Center on Z axis
                    );
                    
                    console.log('Model rotated and positioned at:', object.position);
                    console.log('Model rotation:', object.rotation);
                    console.log('Rotated model box:', rotatedBox);
                    
                    this.scene.add(object);
                    
                    // Adjust camera to fit the model
                    this.fitCameraToModel();
                    
                    resolve(object);
                },
                (progress) => {
                    console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
                },
                (error) => {
                    console.error('Error loading model:', error);
                    reject(error);
                }
            );
        });
    }
    
    fitCameraToModel() {
        if (!this.model) return;
        
        const box = new THREE.Box3().setFromObject(this.model);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        const distance = maxDim * 2;
        this.camera.position.set(distance, distance, distance);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    animate() {
        this.animationId = requestAnimationFrame(this.animate.bind(this));
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    handleResize() {
        if (!this.container || !this.camera || !this.renderer) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    resetView() {
        if (this.model) {
            this.fitCameraToModel();
        } else {
            this.camera.position.set(5, 5, 5);
            this.controls.target.set(0, 0, 0);
            this.camera.lookAt(0, 0, 0);
            this.controls.update();
        }
    }
    
    setWireframe(enabled) {
        if (!this.model) return;
        
        this.model.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.material.wireframe = enabled;
            }
        });
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        if (this.controls) {
            this.controls.dispose();
        }
        
        window.removeEventListener('resize', this.handleResize.bind(this));
    }
}

// Make ThreeViewer available globally
window.ThreeViewer = ThreeViewer;