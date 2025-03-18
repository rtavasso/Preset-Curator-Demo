import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap
import logging
from config import Config

config = Config()

logger = logging.getLogger(__name__)

def calculate_similarity(query_features, compare_features, weights):
    """Calculate weighted similarity between two feature sets"""
    try:
        similarity = 0
        total_weight = sum(weights.values())
        if total_weight == 0:
            # If no weights provided, use equal weights
            weights = {k: 50 for k in query_features.keys()}
            total_weight = sum(weights.values())

        for feature, weight in weights.items():
            if feature not in query_features or feature not in compare_features:
                continue

            if isinstance(query_features[feature], list):
                # For array features, use cosine similarity
                query_array = np.array(query_features[feature])
                compare_array = np.array(compare_features[feature])
                feature_sim = np.dot(query_array, compare_array) / (np.linalg.norm(query_array) * np.linalg.norm(compare_array))
            else:
                # For scalar features, use normalized absolute difference
                max_val = max(query_features[feature], compare_features[feature])
                min_val = min(query_features[feature], compare_features[feature])
                feature_sim = 1 - abs(max_val - min_val) / max_val if max_val != 0 else 1

            similarity += (weight / total_weight) * feature_sim

        return similarity

    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0

def generate_umap(features_list):
    """Generate UMAP visualization coordinates"""
    try:
        if not features_list:
            raise ValueError("Empty features list provided")

        # Convert features to matrix
        feature_matrix = []
        for features in features_list:
            feature_vector = []
            for feature_name, value in features.items():
                if isinstance(value, list):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            feature_matrix.append(feature_vector)

        feature_matrix = np.array(feature_matrix)
        if feature_matrix.size == 0:
            raise ValueError("No valid features found in data")

        # Scale features
        robust_scaler = RobustScaler()
        scaled_features = robust_scaler.fit_transform(feature_matrix)

        # Generate UMAP
        reducer = umap.UMAP(
            n_components=config.UMAP_N_COMPONENTS,
            n_neighbors=config.UMAP_N_NEIGHBORS,
            min_dist=config.UMAP_MIN_DIST,
            # random_state=42
        )
        embedding = reducer.fit_transform(scaled_features)

        return embedding.tolist()

    except Exception as e:
        logger.error(f"Error generating UMAP: {str(e)}")
        raise

def find_nearest_neighbors(query_idx, embedding, k=5):
    """Find k nearest neighbors in the UMAP embedding"""
    try:
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Invalid embedding data")

        query_point = embedding[query_idx]
        distances = []

        for i, point in enumerate(embedding):
            if i != query_idx:
                dist = np.linalg.norm(np.array(query_point) - np.array(point))
                distances.append((i, dist))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    except Exception as e:
        logger.error(f"Error finding nearest neighbors: {str(e)}")
        raise

def calculate_feature_deltas(query_features, neighbor_features, weight_features):
    """Calculate feature deltas between query and neighbor samples using StandardScaler
    
    Args:
        query_features: Dict of features for the query sample
        neighbor_features: Dict of features for the neighbor sample
        weight_features: Dict of feature weights from the sliders
        
    Returns:
        Dict of feature deltas normalized and standardized
    """
    try:
        # Extract only the features that are in the weight_features
        feature_deltas = {}
        
        # Filter active features (features with non-zero weights)
        active_features = {k: v for k, v in weight_features.items() if v != 0}
        if not active_features:
            active_features = weight_features  # Use all if none are active
        
        # Use StandardScaler for each feature
        from sklearn.preprocessing import StandardScaler
        
        for feature_name in active_features.keys():
            if feature_name not in query_features or feature_name not in neighbor_features:
                continue
                
            # Extract the feature values
            query_value = query_features[feature_name]
            neighbor_value = neighbor_features[feature_name]
            
            # Skip array features for simplicity
            if isinstance(query_value, list) or isinstance(neighbor_value, list):
                continue
                
            # Create a mini dataset for standardization
            feature_values = np.array([query_value, neighbor_value]).reshape(-1, 1)
            
            # Standardize the values
            scaler = StandardScaler()
            standardized_values = scaler.fit_transform(feature_values).flatten()
            
            # Calculate the delta in standard deviations
            std_delta = standardized_values[1] - standardized_values[0]
            
            # Convert to percentage (1 std = 100%)
            delta_percent = std_delta * 100
            
            feature_deltas[feature_name] = delta_percent
            
        return feature_deltas
        
    except Exception as e:
        logger.error(f"Error calculating feature deltas: {str(e)}")
        return {}

def rank_by_feature_weights(query_features, neighbors_data, all_features, weights):
    """Rank neighbors based on feature weights using StandardScaler
    
    Args:
        query_features: Dict of features for the query sample
        neighbors_data: List of (idx, distance) tuples from UMAP neighbors
        all_features: Dict of all audio features
        weights: Dict of feature weights from the sliders
        
    Returns:
        List of dicts with filename, similarity, distance, and feature_deltas
    """
    try:
        # Filter to only weights that are not zero
        active_weights = {k: v for k, v in weights.items() if v != 0}
        
        # If no active weights, return original order
        if not active_weights:
            results = []
            for idx, distance in neighbors_data:
                filename = list(all_features.keys())[idx]
                similarity = calculate_similarity(
                    query_features, all_features[filename], weights)
                
                # Calculate feature deltas
                feature_deltas = calculate_feature_deltas(
                    query_features, all_features[filename], weights)
                    
                results.append({
                    'filename': filename,
                    'similarity': similarity,
                    'distance': float(distance),
                    'feature_deltas': feature_deltas
                })
            return results
            
        # StandardScale the features for query and neighbors (only the features in weights)
        from sklearn.preprocessing import StandardScaler
        
        # Get filenames for all neighbors
        filenames = [list(all_features.keys())[idx] for idx, _ in neighbors_data]
        
        # Add query filename to the list for scaling
        all_filenames = filenames + [list(query_features.keys())[0] if isinstance(query_features, dict) else "query"]
        
        # Collect feature values for scaling
        scalar_features = {k: [] for k in active_weights.keys()}
        for filename in filenames:
            neighbor_features = all_features[filename]
            for feature_name in active_weights.keys():
                if feature_name in neighbor_features and not isinstance(neighbor_features[feature_name], list):
                    scalar_features.setdefault(feature_name, []).append(neighbor_features[feature_name])
        
        # Add query features to scaling
        for feature_name in active_weights.keys():
            if feature_name in query_features and not isinstance(query_features[feature_name], list):
                scalar_features.setdefault(feature_name, []).append(query_features[feature_name])
        
        # Standardize features
        scaled_features = {}
        for feature_name, values in scalar_features.items():
            if len(values) > 1:  # Need at least 2 values for scaling
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
                
                # Last value is the query
                query_scaled = scaled_values[-1]
                
                # Store scaled values for neighbors
                for i, filename in enumerate(filenames):
                    if i < len(scaled_values) - 1:  # Skip the query which is last
                        scaled_features.setdefault(filename, {})[feature_name] = scaled_values[i]
                
                # Store query scaled value
                scaled_features.setdefault("query", {})[feature_name] = query_scaled
        
        # Calculate weighted distances and feature deltas
        weighted_results = []
        
        for idx, umap_distance in neighbors_data:
            filename = list(all_features.keys())[idx]
            neighbor_features = all_features[filename]
            
            # Calculate raw similarity using unscaled features
            similarity = calculate_similarity(
                query_features, neighbor_features, weights)
                
            # Calculate feature deltas using scaled features
            feature_deltas = {}
            weighted_score = 0
            total_weight = sum(abs(w) for w in active_weights.values())
            
            for feature_name, weight in active_weights.items():
                if (feature_name in scaled_features.get("query", {}) and 
                    feature_name in scaled_features.get(filename, {})):
                    
                    query_scaled = scaled_features["query"][feature_name]
                    neighbor_scaled = scaled_features[filename][feature_name]
                    
                    # Delta in standard deviations
                    delta = neighbor_scaled - query_scaled
                    
                    # Scale to percentage for display
                    delta_percent = delta * 100  # Each standard deviation = 100%
                    feature_deltas[feature_name] = delta_percent
                    
                    # For ranking:
                    # - Positive weight means we prefer higher values
                    # - Negative weight means we prefer lower values
                    norm_weight = abs(weight) / total_weight
                    
                    if weight > 0:
                        # Higher values are better
                        weighted_score += norm_weight * delta
                    else:
                        # Lower values are better
                        weighted_score -= norm_weight * delta
            
            weighted_results.append({
                'filename': filename,
                'similarity': similarity,
                'distance': float(umap_distance),
                'weighted_score': weighted_score,
                'feature_deltas': feature_deltas
            })
            
        # Sort by weighted score (higher is better)
        weighted_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Remove the weighted_score field from results
        for result in weighted_results:
            del result['weighted_score']
            
        return weighted_results
        
    except Exception as e:
        logger.error(f"Error ranking by feature weights: {str(e)}")
        raise

def find_nearest_neighbors_by_cosine_similarity(query_filename, all_features, k=5):
    """Find nearest neighbors using cosine similarity on features containing 'similarity'
    
    Args:
        query_filename: Filename of the query audio file
        all_features: Dict of all audio features
        k: Number of neighbors to return
        
    Returns:
        List of (idx, similarity) tuples
    """
    try:
        # Get query features
        query_features = all_features[query_filename]
        
        # Extract all features containing "similarity"
        similarity_features = {}
        
        # First, identify all similarity-related features
        similarity_feature_names = []
        for feature_name in query_features.keys():
            if "similarity" in feature_name.lower():
                similarity_feature_names.append(feature_name)
        
        logger.warning(f"Found {len(similarity_feature_names)} similarity-related features: {similarity_feature_names}")
        
        if not similarity_feature_names:
            logger.warning("No similarity-related features found, using all features instead")
            # Use all scalar features if no similarity features found
            for feature_name, value in query_features.items():
                if not isinstance(value, list):
                    similarity_feature_names.append(feature_name)
            
            logger.warning(f"Using {len(similarity_feature_names)} scalar features instead")
        
        # Extract similarity features for all audio files
        filenames = list(all_features.keys())
        feature_matrix = []
        valid_filenames = []
        
        # Get query vector first
        query_vector = []
        for feature_name in similarity_feature_names:
            if feature_name in query_features:
                if isinstance(query_features[feature_name], list):
                    query_vector.extend(query_features[feature_name])
                else:
                    query_vector.append(query_features[feature_name])
        
        # If no valid features found, return empty result
        if not query_vector:
            logger.error("No valid features found for query file")
            return []
        
        query_vector = np.array(query_vector).reshape(1, -1)
        
        # Get vectors for all other files
        for i, filename in enumerate(filenames):
            if filename == query_filename:
                continue  # Skip query file
                
            feature_vector = []
            features = all_features[filename]
            
            # Check if this file has all required features
            has_all_features = True
            for feature_name in similarity_feature_names:
                if feature_name not in features:
                    has_all_features = False
                    break
            
            if not has_all_features:
                continue
                
            # Extract feature values
            for feature_name in similarity_feature_names:
                if isinstance(features[feature_name], list):
                    feature_vector.extend(features[feature_name])
                else:
                    feature_vector.append(features[feature_name])
            
            if feature_vector:
                feature_matrix.append(feature_vector)
                valid_filenames.append(filename)
        
        # Convert to numpy array
        if not feature_matrix:
            logger.error("No valid feature vectors found for comparison")
            return []
            
        feature_matrix = np.array(feature_matrix)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, feature_matrix)[0]
        
        # Create (idx, similarity) tuples
        similarity_tuples = []
        for i, similarity in enumerate(similarities):
            # Create a tuple with the index in the original all_features dict and the similarity
            orig_idx = filenames.index(valid_filenames[i])
            similarity_tuples.append((orig_idx, float(similarity)))
        
        # Sort by similarity (higher is better)
        similarity_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarity_tuples[:k]
        
    except Exception as e:
        logger.error(f"Error finding nearest neighbors by cosine similarity: {str(e)}", exc_info=True)
        return []