import os
from typing import Dict, Any


class Config:
    # R2 Configuration
    R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
    R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
    R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
    R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')
    R2_REGION_NAME = os.environ.get('R2_REGION_NAME', 'auto')

    # Environment Configuration
    ENVIRONMENT = os.environ.get('FLASK_ENV', 'production')
    ENABLE_VISUALIZATION = os.environ.get('ENABLE_VISUALIZATION', '0').lower() in ('1', 'true', 'yes')

    # Audio Processing Configuration
    ALLOWED_EXTENSIONS = {'wav'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Feature Database Configuration
    FEATURE_DB_ENABLED = True
    FEATURE_DB_MAX_NEIGHBORS = 16
    FEATURE_DB_PREFER_SIMILARITY_FEATURES = True
    FEATURE_DB_PREFER_PATTERN = os.environ.get('FEATURE_DB_PREFER_PATTERN', 'similarity')

    # Feature Weights Default Values
    FEATURE_WEIGHTS = {
        "warmth": {
            "display_name": "Warmth",
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of tonal warmth in the audio"
        },
        "depth": {
            "display_name": "Depth",
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of depth and spatial characteristics"
        },
        "brightness": {
            "display_name": "Brightness",
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of brightness and clarity"
        },
        "clarity": {
            "display_name": "Clarity",
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of clarity and detail"
        },
        "punchiness": {
            "display_name": "Punchiness",   
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of punchiness and impact"
        },
        "dynamic_range": {
            "display_name": "Dynamic Range",
            "min": -3,
            "max": 3,
            "default": 0,
            "description": "Measure of dynamic range and loudness"
        }
    }

    # UMAP Configuration
    UMAP_N_NEIGHBORS = 30
    UMAP_MIN_DIST = 0.1
    UMAP_N_COMPONENTS = 10

    # Add configuration validation
    def validate(self) -> Dict[str, str]:
        """
        Validate configuration and return a dict of any issues found.
        
        Returns:
            Dict[str, str]: A dictionary of configuration problems {config_name: error_message}
        """
        issues = {}
        
        # Check required R2 settings
        if not all([self.R2_ENDPOINT_URL, self.R2_ACCESS_KEY_ID, 
                   self.R2_SECRET_ACCESS_KEY, self.R2_BUCKET_NAME]):
            issues["R2_CONFIG"] = "Missing one or more required R2 configuration values"
        
        # Feature weights
        if not self.FEATURE_WEIGHTS:
            issues["FEATURE_WEIGHTS"] = "No feature weights defined"
            
        # Feature database settings
        if self.FEATURE_DB_ENABLED and not self.FEATURE_DB_PREFER_PATTERN:
            issues["FEATURE_DB_PREFER_PATTERN"] = "Feature database prefer pattern is empty but needed"
            
        # UMAP settings - only validate if visualization is enabled
        if self.ENABLE_VISUALIZATION:
            if self.UMAP_N_NEIGHBORS < 2:
                issues["UMAP_N_NEIGHBORS"] = "UMAP n_neighbors must be at least 2"
                
            if self.UMAP_MIN_DIST <= 0 or self.UMAP_MIN_DIST > 1:
                issues["UMAP_MIN_DIST"] = "UMAP min_dist must be between 0 and 1"
                
            if self.UMAP_N_COMPONENTS < 2:
                issues["UMAP_N_COMPONENTS"] = "UMAP n_components must be at least 2"
            
        return issues
