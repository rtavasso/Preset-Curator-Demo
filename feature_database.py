import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import time
from config import Config

logger = logging.getLogger(__name__)
config = Config()

class FeatureDatabase:
    """
    A simple in-memory database for fast cosine similarity computations.
    Precomputes ALL feature vectors and similarity matrices during initialization
    for instant retrieval of similar files.
    
    Feature Format:
    The database expects features in the following format:
    {
        "file1": {
            "feature1": value1,
            "feature2": value2,
            "feature3": [array_of_values],
            ...
        },
        "file2": {
            ...
        }
    }
    
    Where values can be either scalars (float, int) or lists/arrays of numbers.
    """
    
    def __init__(self):
        self.is_initialized: bool = False
        self.feature_names: List[str] = []  # List of feature names used for vectors
        self.scalar_feature_names: List[str] = []  # List of scalar feature names
        self.filenames: List[str] = []  # List of all filenames
        self.feature_matrix: Optional[np.ndarray] = None  # Numpy array of all feature vectors
        self.original_features: Dict[str, Dict[str, Any]] = {}  # Original features dictionary
        self.filename_to_index: Dict[str, int] = {}  # Mapping from filename to index in the matrix
        self.initialization_time: float = 0  # Time taken to initialize the database
        self.precomputed_similarities: Dict[str, List[Tuple[str, float]]] = {}  # Precomputed similarities
        self.similarity_matrix: Optional[np.ndarray] = None  # Full similarity matrix between all files
        self._file_cache: Dict[str, Dict[str, Any]] = {}  # Cache for normalized filenames -> features
        self._similarity_cache: Dict[str, List[Tuple[str, float]]] = {}  # Cache for similarity queries
    
    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize a filename by removing .wav extension if present.
        
        Args:
            filename: The filename to normalize
            
        Returns:
            str: Normalized filename without .wav extension
        """
        return filename.split('.wav')[0] if filename.endswith('.wav') else filename
    
    def _select_features(self, features_dict: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Select which features to use for similarity calculation.
        Identifies scalar features and preferred features based on configuration.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            Tuple containing (selected_feature_names, all_scalar_feature_names)
        """
        # Use a sample file to identify feature names
        if not features_dict:
            return [], []
            
        sample_features = next(iter(features_dict.values()))
        
        # Identify scalar features and preferred features
        preferred_feature_names = []
        scalar_feature_names = []
        
        # Features to prefer based on configuration
        prefer_pattern = config.FEATURE_DB_PREFER_PATTERN.lower() if hasattr(config, 'FEATURE_DB_PREFER_PATTERN') else "similarity"
        
        for feature_name, value in sample_features.items():
            if not isinstance(value, list):
                scalar_feature_names.append(feature_name)
                if prefer_pattern in feature_name.lower():
                    preferred_feature_names.append(feature_name)
        
        # Select feature names based on preference
        if preferred_feature_names and getattr(config, 'FEATURE_DB_PREFER_SIMILARITY_FEATURES', True):
            selected_feature_names = preferred_feature_names
            logger.info(f"Using {len(preferred_feature_names)} preferred features matching '{prefer_pattern}'")
        else:
            selected_feature_names = scalar_feature_names
            logger.info(f"Using all {len(scalar_feature_names)} scalar features")
        
        return selected_feature_names, scalar_feature_names
    
    def _extract_feature_vectors(
        self, 
        features_dict: Dict[str, Dict[str, Any]], 
        feature_names: List[str]
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Extract feature vectors for all files using the selected features.
        
        Args:
            features_dict: Dictionary of features
            feature_names: List of feature names to extract
            
        Returns:
            Tuple containing (feature_vectors, valid_filenames)
        """
        feature_vectors = []
        valid_filenames = []
        total_files = len(features_dict)
        processed = 0
        
        # Report progress every 10% of files
        progress_step = max(1, total_files // 10)
        
        for filename, features in features_dict.items():
            # Check if this file has all required features
            has_all_features = True
            for feature_name in feature_names:
                if feature_name not in features:
                    has_all_features = False
                    break
            
            if not has_all_features:
                continue
            
            # Extract feature values
            feature_vector = []
            for feature_name in feature_names:
                value = features[feature_name]
                if isinstance(value, list):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            if feature_vector:
                feature_vectors.append(feature_vector)
                valid_filenames.append(filename)
            
            # Report progress
            processed += 1
            if processed % progress_step == 0:
                progress_percent = int(processed / total_files * 100)
                logger.debug(f"Extracting feature vectors: {progress_percent}% complete")
        
        logger.info(f"Extracted {len(feature_vectors)} valid feature vectors from {total_files} files")
        return feature_vectors, valid_filenames
    
    def _compute_similarities(
        self, 
        feature_matrix: np.ndarray, 
        filenames: List[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute similarities between all files.
        
        Args:
            feature_matrix: Matrix of feature vectors
            filenames: List of filenames
            
        Returns:
            Dictionary of precomputed similarities
        """
        # Calculate full cosine similarity matrix
        similarity_matrix = cosine_similarity(feature_matrix)
        precomputed_similarities = {}
        
        # Use batching to reduce memory pressure
        batch_size = min(1000, len(filenames))
        num_batches = (len(filenames) + batch_size - 1) // batch_size
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(filenames))
            
            logger.debug(f"Computing similarities batch {batch+1}/{num_batches}")
            
            for i in range(start_idx, end_idx):
                filename = filenames[i]
                # Get similarities for this file (exclude self-comparison)
                similarities = similarity_matrix[i]
                
                # Create a mask to exclude the current file
                mask = np.ones(len(filenames), dtype=bool)
                mask[i] = False
                
                # Get indices of top similar files
                similar_indices = np.argsort(similarities[mask])[::-1]
                masked_indices = np.arange(len(filenames))[mask]
                
                # Store the top similar files for this filename
                similar_files = []
                for j in range(len(similar_indices)):
                    idx = masked_indices[similar_indices[j]]
                    similar_files.append((filenames[idx], float(similarities[idx])))
                
                precomputed_similarities[filename] = similar_files
        
        return precomputed_similarities, similarity_matrix
    
    def initialize(self, features_dict: Dict[str, Dict[str, Any]]) -> bool:
        """
        Initialize the database with features extracted from the JSON.
        Precomputes ALL similarities between all files for instant retrieval.
        
        Args:
            features_dict: Dictionary mapping filenames to feature dictionaries
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        start_time = time.time()
        
        if not features_dict:
            logger.error("Cannot initialize feature database: empty features dictionary")
            return False
        
        try:
            # Store original features
            self.original_features = features_dict
            
            # Get all filenames
            self.filenames = list(features_dict.keys())
            
            # Create filename to index mapping
            self.filename_to_index = {filename: i for i, filename in enumerate(self.filenames)}
            
            # Select features to use for similarity calculation
            self.feature_names, self.scalar_feature_names = self._select_features(features_dict)
            
            if not self.feature_names:
                logger.error("No suitable features found for the database")
                return False
            
            # Extract feature vectors
            feature_vectors, valid_filenames = self._extract_feature_vectors(
                features_dict, self.feature_names)
            
            # Update filenames and filename_to_index to only include valid files
            self.filenames = valid_filenames
            self.filename_to_index = {filename: i for i, filename in enumerate(self.filenames)}
            
            # Convert to numpy array
            if not feature_vectors:
                logger.error("No valid feature vectors created for the database")
                return False
            
            self.feature_matrix = np.array(feature_vectors)
            
            # Precompute full similarity matrix between all files
            logger.info("Precomputing similarities between all files...")
            precompute_start = time.time()
            
            # Compute similarities
            self.precomputed_similarities, self.similarity_matrix = self._compute_similarities(
                self.feature_matrix, self.filenames)
            
            precompute_time = time.time() - precompute_start
            logger.info(f"Precomputed similarities for {len(self.filenames)} files in {precompute_time:.2f} seconds")
            
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            logger.info(f"Feature database initialized with {len(self.filenames)} files "
                        f"and {self.feature_matrix.shape[1]} features "
                        f"in {self.initialization_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing feature database: {str(e)}", exc_info=True)
            return False
    
    @lru_cache(maxsize=100)
    def get_similar_files(self, query_filename: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get k most similar files to the query file from precomputed similarities.
        No calculation needed - instant retrieval.
        
        Args:
            query_filename: Name of the query audio file
            k: Number of similar files to return
            
        Returns:
            List of (filename, similarity) tuples sorted by similarity (descending)
        """
        if not self.is_initialized:
            logger.error("Feature database not initialized")
            return []
        
        try:
            # Normalize the filename
            base_filename = self._normalize_filename(query_filename)
            
            # Check cache first
            if base_filename in self._similarity_cache:
                return self._similarity_cache[base_filename][:k]
            
            # Check if query file exists in precomputed similarities
            if base_filename not in self.precomputed_similarities:
                logger.error(f"Query file {base_filename} not found in precomputed similarities")
                return []
            
            # Get precomputed similarities for this file
            similar_files = self.precomputed_similarities[base_filename]
            
            # Store in cache for future use
            self._similarity_cache[base_filename] = similar_files
            
            # Return top k
            return similar_files[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving similar files: {str(e)}", exc_info=True)
            return []
    
    def find_similar(self, query_filename: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Legacy method for compatibility - uses precomputed similarities.
        
        Args:
            query_filename: Name of the query audio file
            k: Number of similar files to return
            
        Returns:
            List of (filename, similarity) tuples sorted by similarity (descending)
        """
        return self.get_similar_files(query_filename, k)
    
    def get_features(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get the original features for a file.
        
        Args:
            filename: Name of the audio file
            
        Returns:
            Dictionary of features for the file or None if not found
        """
        if not self.is_initialized:
            logger.error("Feature database not initialized")
            return None
        
        try:
            # Normalize the filename
            base_filename = self._normalize_filename(filename)
            
            # Check cache first
            if base_filename in self._file_cache:
                return self._file_cache[base_filename]
            
            if base_filename not in self.original_features:
                logger.error(f"Features for file {base_filename} not found")
                return None
            
            # Get features and cache them
            features = self.original_features[base_filename]
            self._file_cache[base_filename] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting features: {str(e)}")
            return None
    
    def calculate_feature_deltas(
        self, 
        query_filename: str, 
        similar_filename: str, 
        weights: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature deltas between query and similar file.
        
        Args:
            query_filename: Name of the query audio file
            similar_filename: Name of the similar audio file
            weights: Dictionary of feature weights
            
        Returns:
            Dictionary of feature deltas
        """
        if not self.is_initialized:
            logger.error("Feature database not initialized")
            return {}
        
        try:
            query_features = self.get_features(query_filename)
            similar_features = self.get_features(similar_filename)
            
            if not query_features or not similar_features:
                return {}
            
            # Calculate deltas for scalar features with weights
            deltas = {}
            
            for feature_name in self.scalar_feature_names:
                if feature_name not in weights:
                    continue
                
                if (feature_name in query_features and 
                    feature_name in similar_features and
                    not isinstance(query_features[feature_name], list) and
                    not isinstance(similar_features[feature_name], list)):
                    
                    # Create a mini dataset for standardization
                    feature_values = np.array([
                        query_features[feature_name], 
                        similar_features[feature_name]
                    ]).reshape(-1, 1)
                    
                    # Standardize the values
                    scaler = StandardScaler()
                    standardized_values = scaler.fit_transform(feature_values).flatten()
                    
                    # Calculate delta in standard deviations
                    std_delta = standardized_values[1] - standardized_values[0]
                    
                    # Convert to percentage (1 std = 100%)
                    delta_percent = std_delta * 100
                    
                    # Store delta with weight information
                    deltas[feature_name] = {
                        'delta': delta_percent,
                        'weight': weights.get(feature_name, 0)
                    }
            
            return deltas
            
        except Exception as e:
            logger.error(f"Error calculating feature deltas: {str(e)}")
            return {}
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feature database.
        
        Returns:
            Dictionary of database statistics
        """
        if not self.is_initialized:
            return {
                "initialized": False,
                "message": "Database not initialized"
            }
            
        # Calculate memory usage
        feature_matrix_bytes = 0
        similarity_matrix_bytes = 0
        
        if self.feature_matrix is not None:
            feature_matrix_bytes = self.feature_matrix.nbytes
            
        if self.similarity_matrix is not None:
            similarity_matrix_bytes = self.similarity_matrix.nbytes
            
        total_bytes = feature_matrix_bytes + similarity_matrix_bytes
        
        return {
            "initialized": True,
            "num_files": len(self.filenames),
            "num_features": len(self.feature_names),
            "num_precomputed_similarities": len(self.precomputed_similarities),
            "initialization_time": self.initialization_time,
            "initialization_seconds": f"{self.initialization_time:.2f}",
            "feature_matrix_size_mb": feature_matrix_bytes / (1024 * 1024),
            "similarity_matrix_size_mb": similarity_matrix_bytes / (1024 * 1024),
            "total_memory_mb": total_bytes / (1024 * 1024),
            "feature_names": self.feature_names,
            "scalar_feature_names": self.scalar_feature_names
        } 
        
    def clear_caches(self) -> None:
        """
        Clear all internal caches.
        """
        self._file_cache.clear()
        self._similarity_cache.clear()
        # Also clear the LRU cache of get_similar_files
        self.get_similar_files.cache_clear()
        logger.debug("Feature database caches cleared") 