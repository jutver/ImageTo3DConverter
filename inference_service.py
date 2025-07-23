import torch
import numpy as np
import cv2
from torchvision import transforms, models
import os
import open3d as o3d
import torch.nn as nn
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define the model class exactly as in the original inference script
class ResNetVoxel(nn.Module):
    def __init__(self, voxel_dim=32):
        super().__init__()
        self.voxel_dim = voxel_dim
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        resnet.fc = nn.Identity()
        self.encoder = resnet
        
        # Decoder to generate voxel grid
        self.decoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, voxel_dim**3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, self.voxel_dim, self.voxel_dim, self.voxel_dim)

class InferenceService:
    def __init__(self, model_path='resnet_voxel_model.pth', voxel_dim=32):
        self.model_path = model_path
        self.voxel_dim = voxel_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"InferenceService initialized with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = ResNetVoxel(voxel_dim=self.voxel_dim).to(self.device)
            
            # Check if model file exists
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Using untrained model.")
                # Create a dummy model file path in environment if not exists
                if not os.path.exists(self.model_path):
                    # For demo purposes, we'll use the untrained model
                    # In production, ensure the trained model is available
                    logger.warning("Using untrained model for demonstration. Results may not be optimal.")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def preprocess_image(self, image_path):
        """Load and preprocess the image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found or invalid at {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise e
    
    def generate_3d_model(self, image_tensor):
        """Generate 3D model from the image tensor."""
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                voxel_pred = self.model(image_tensor)
                return voxel_pred.cpu().numpy()
                
        except Exception as e:
            logger.error(f"3D model generation failed: {e}")
            raise e
    
    def save_voxel_as_obj(self, voxel_data, output_path, threshold=0.5):
        """Saves the voxel data as a 3D .obj file."""
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Get the coordinates of voxels above the threshold
            voxel_data = np.squeeze(voxel_data)
            positions = np.argwhere(voxel_data > threshold)
            
            if positions.size == 0:
                logger.warning("No voxels above the threshold. Creating a simple cube.")
                # Create a simple cube if no voxels are found
                positions = np.array([[15, 15, 15], [16, 15, 15], [15, 16, 15], [16, 16, 15],
                                    [15, 15, 16], [16, 15, 16], [15, 16, 16], [16, 16, 16]])

            # Create an Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions.astype(float))
            
            # Create a mesh from the point cloud
            mesh = o3d.geometry.TriangleMesh()
            
            # Create cubes for each voxel position
            for pos in positions:
                cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                cube.translate(pos.astype(float))
                mesh += cube
            
            # Merge close vertices and remove duplicates
            if len(mesh.vertices) > 0:
                mesh.merge_close_vertices(0.01)
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_unreferenced_vertices()
                
                # Compute normals for better rendering
                mesh.compute_vertex_normals()
            
            # Save the mesh as an .obj file
            success = o3d.io.write_triangle_mesh(output_path, mesh)
            
            if success:
                logger.info(f"3D model saved successfully to {output_path}")
                return True
            else:
                logger.error(f"Failed to save 3D model to {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save voxel as OBJ: {e}")
            return False
    
    def process_image(self, image_path, output_path):
        """Complete pipeline to process an image and generate a 3D model."""
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            logger.info("Image preprocessed successfully")
            
            # Generate 3D model
            voxel_data = self.generate_3d_model(image_tensor)
            logger.info("3D model generated successfully")
            
            # Save as .obj
            success = self.save_voxel_as_obj(voxel_data, output_path)
            
            if success:
                logger.info(f"Complete processing successful. Output: {output_path}")
                return True
            else:
                logger.error("Failed to save the generated model")
                return False
                
        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            return False
