import os
import logging
import json
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple, Union
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, send_file, session, Response
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import NoCredentialsError, ClientError
import audio_processor as ap
from config import Config
from dotenv import load_dotenv
import numpy as np
import time
from feature_database import FeatureDatabase
import datetime
import threading
import requests

# Load environment variables from .env file
load_dotenv()

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize feature database
feature_db = FeatureDatabase()
config = Config()

# Get required environment variables with fallbacks
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')

if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, BUCKET_NAME]):
    logger.warning("Missing required R2 environment variables. Some features may not work correctly.")

# R2 Configuration
try:
    r2 = boto3.client('s3',
                    endpoint_url=R2_ENDPOINT_URL,
                    aws_access_key_id=R2_ACCESS_KEY_ID,
                    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                    config=BotoConfig(signature_version='s3v4'),
                    region_name=config.R2_REGION_NAME)
    logger.info("S3 client initialized successfully")
except (NoCredentialsError, ClientError) as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    r2 = None

FEATURES_KEY = 'features.json'
_features_cache = None
_features_cache_time = 0
CACHE_TIMEOUT = 300  # 5 minutes cache timeout


# Heartbeat functionality
def heartbeat_task(interval_minutes=5):
    """
    Background task to periodically hit the health endpoint.
    
    Args:
        interval_minutes: How often to ping in minutes
    """
    logger.info(f"Starting heartbeat service. Will ping health endpoint every {interval_minutes} minutes")
    interval_seconds = interval_minutes * 60
    
    # Use a separate logger for heartbeat messages
    heartbeat_logger = logging.getLogger("heartbeat")

    # Get base URL from environment or default to localhost
    base_url = os.environ.get("APP_BASE_URL", "http://localhost:10000")
    health_url = f"{base_url}/health"
    
    while True:
        try:
            heartbeat_logger.info(f"Pinging health endpoint: {health_url}")
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                heartbeat_logger.info(f"Health check successful (Response time: {response.elapsed.total_seconds():.2f}s)")
            else:
                heartbeat_logger.warning(f"Health check received unexpected status code: {response.status_code}")
                
        except requests.RequestException as e:
            heartbeat_logger.error(f"Failed to connect to health endpoint: {str(e)}")
        except Exception as e:
            heartbeat_logger.error(f"Unexpected error in heartbeat: {str(e)}")
            
        # Sleep until next interval
        time.sleep(interval_seconds)


def start_heartbeat(app):
    """
    Start the heartbeat background thread if enabled.
    """
    # Only start heartbeat if explicitly enabled via environment variable
    heartbeat_enabled = os.environ.get("ENABLE_HEARTBEAT", "false").lower() == "true"
    
    if heartbeat_enabled:
        # Get interval from environment variable or use default (14 minutes)
        interval = int(os.environ.get("HEARTBEAT_INTERVAL_MINUTES", 14))
        
        # Start heartbeat in a daemon thread
        logger.info(f"Starting heartbeat thread with {interval} minute interval")
        heartbeat_thread = threading.Thread(
            target=heartbeat_task,
            args=(interval,),
            daemon=True  # Thread will exit when main app exits
        )
        heartbeat_thread.start()
    else:
        logger.info("Heartbeat service is disabled. Set ENABLE_HEARTBEAT=true to enable.")


@lru_cache(maxsize=1)
def load_features() -> Optional[Dict[str, Any]]:
    """
    Load precomputed features from R2 with caching.
    
    Returns:
        Dict[str, Any]: Dictionary of audio features or None if loading fails
    """
    global _features_cache, _features_cache_time
    
    # Check if we have a valid cache
    current_time = time.time()
    if _features_cache is not None and (current_time - _features_cache_time) < CACHE_TIMEOUT:
        logger.debug("Using cached features data")
        return _features_cache
    
    if r2 is None:
        logger.error("S3 client not initialized")
        return None
        
    try:
        logger.info(f"Loading features from {FEATURES_KEY}")
        response = r2.get_object(Bucket=BUCKET_NAME, Key=FEATURES_KEY)
        features = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Loaded features for {len(features)} audio files")
        
        # Update cache
        _features_cache = features
        _features_cache_time = current_time
        
        return features
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"Features file {FEATURES_KEY} not found in bucket {BUCKET_NAME}")
        else:
            logger.error(f"Error loading features: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return None


def initialize_feature_database(retry_count: int = 1) -> bool:
    """
    Initialize the feature database with precomputed features.
    
    Args:
        retry_count: Number of times to retry initialization if it fails
        
    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    if not config.FEATURE_DB_ENABLED:
        logger.info("Feature database is disabled in configuration")
        return False
    
    for attempt in range(retry_count + 1):
        features = load_features()
        if not features:
            if attempt < retry_count:
                logger.warning(f"Failed to load features, retrying ({attempt+1}/{retry_count})...")
                time.sleep(2)  # Wait before retrying
                continue
            logger.error("Failed to load features for database initialization after retries")
            return False
        
        logger.info("Initializing feature database...")
        start_time = time.time()
        success = feature_db.initialize(features)
        
        if success:
            duration = time.time() - start_time
            logger.info(f"Feature database initialized successfully in {duration:.2f} seconds")
            return True
        
        if attempt < retry_count:
            logger.warning(f"Failed to initialize database, retrying ({attempt+1}/{retry_count})...")
            time.sleep(2)  # Wait before retrying
    
    logger.error("Failed to initialize feature database after retries")
    return False


def create_app() -> Flask:
    """
    Create and configure the Flask app.
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Set secret key for session - use a persistent key if possible
    app.secret_key = os.environ.get("SESSION_SECRET") or os.environ.get("FLASK_SECRET_KEY")
    
    # If no environment variable is set, warn but still use a random key
    if not app.secret_key:
        logger.warning("No SESSION_SECRET environment variable found. Using random secret key. Sessions will be invalidated on restart.")
        app.secret_key = os.urandom(24)
    
    # Add function to get current datetime to Jinja templates
    @app.context_processor
    def inject_now():
        return {'now': datetime.datetime.now}
    
    # Initialize the feature database during app creation
    with app.app_context():
        initialize_feature_database(retry_count=2)
        
        # Start the heartbeat background thread
        start_heartbeat(app)
    
    return app

# Create the Flask application
app = create_app()

# Audio file list cache
_audio_files_cache = None
_audio_files_cache_time = 0


def list_audio_files(use_cache: bool = True) -> List[str]:
    """
    List all WAV files in the bucket with caching.
    
    Args:
        use_cache: Whether to use cached results if available
        
    Returns:
        List[str]: List of audio file names
    """
    global _audio_files_cache, _audio_files_cache_time
    
    # Check if we have a valid cache and caching is enabled
    current_time = time.time()
    if use_cache and _audio_files_cache is not None and (current_time - _audio_files_cache_time) < CACHE_TIMEOUT:
        logger.debug("Using cached audio files list")
        return _audio_files_cache
    
    if r2 is None:
        logger.error("S3 client not initialized")
        return []
        
    try:
        # For large buckets, we should use pagination
        audio_files = []
        continuation_token = None
        
        while True:
            # Prepare parameters for list_objects_v2
            params = {'Bucket': BUCKET_NAME, 'MaxKeys': 1000}  # Get up to 1000 keys at a time
            if continuation_token:
                params['ContinuationToken'] = continuation_token
                
            response = r2.list_objects_v2(**params)
            
            # Process current batch of objects
            batch_files = [
                obj['Key'] for obj in response.get('Contents', [])
                if obj['Key'].endswith('.wav')
            ]
            audio_files.extend(batch_files)
            
            # Check if there are more objects to fetch
            if not response.get('IsTruncated'):
                break
                
            continuation_token = response.get('NextContinuationToken')
            
        logger.info(f"Found {len(audio_files)} WAV files")
        
        # Update cache
        if use_cache:
            _audio_files_cache = audio_files
            _audio_files_cache_time = current_time
            
        return audio_files
    except Exception as e:
        logger.error(f"Error listing audio files: {str(e)}")
        return []


@app.route('/')
def index() -> str:
    """
    Render the main application page.
    
    Returns:
        str: Rendered HTML template
    """
    # Use cached audio files list for better performance
    audio_files = list_audio_files(use_cache=True)
    
    # Validate config
    config_issues = config.validate()
    if config_issues:
        for issue, message in config_issues.items():
            logger.warning(f"Configuration issue: {issue} - {message}")
    
    return render_template('index.html',
                          audio_files=audio_files,
                          feature_weights=config.FEATURE_WEIGHTS)


@app.route('/audio/<filename>')
def get_audio(filename: str) -> Union[Response, Tuple[Response, int]]:
    """
    Stream audio file from R2 with support for range requests.
    
    Args:
        filename: The name of the audio file to stream
        
    Returns:
        Response: Audio file stream or error response
    """
    if r2 is None:
        logger.error("S3 client not available")
        return jsonify({'error': 'Storage service unavailable'}), 503
        
    try:
        # Safely extract base filename without extension
        base_filename = os.path.splitext(filename)[0]
        file_key = f"{base_filename}.wav"
        
        # Get file metadata first for content-length
        try:
            head = r2.head_object(Bucket=BUCKET_NAME, Key=file_key)
            file_size = head['ContentLength']
            content_type = head.get('ContentType', 'audio/wav')
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Audio file not found: {file_key}")
                return jsonify({'error': 'Audio file not found'}), 404
            raise
        
        # Handle range request if present
        ranges = request.headers.get('Range')
        if ranges:
            range_header = ranges.replace('bytes=', '')
            start, end = range_header.split('-')
            
            start = int(start) if start else 0
            end = int(end) if end else (file_size - 1)
            
            # Ensure end doesn't exceed file size
            end = min(end, file_size - 1)
            
            # Calculate the length of the content
            length = end - start + 1
            
            # Get the specified range
            response = r2.get_object(
                Bucket=BUCKET_NAME, 
                Key=file_key,
                Range=f'bytes={start}-{end}'
            )
            
            # Create partial content response
            rv = Response(
                response['Body'].read(),
                206,
                mimetype=content_type,
                direct_passthrough=True
            )
            rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
            rv.headers.add('Accept-Ranges', 'bytes')
            rv.headers.add('Content-Length', str(length))
            return rv
        
        # No range request, get the full file
        file_obj = r2.get_object(Bucket=BUCKET_NAME, Key=file_key)
        
        # Create response
        rv = send_file(
            BytesIO(file_obj['Body'].read()),
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )
        
        # Add content length header
        rv.headers.add('Content-Length', str(file_size))
        rv.headers.add('Accept-Ranges', 'bytes')
        return rv
        
    except ClientError as e:
        status_code = e.response['ResponseMetadata']['HTTPStatusCode']
        logger.error(f"S3 error ({status_code}): {str(e)}")
        return jsonify({'error': f'Storage service error: {str(e)}'}), status_code
    except Exception as e:
        logger.error(f"Error streaming audio: {str(e)}")
        return jsonify({'error': 'Server error while streaming audio'}), 500


@app.route('/audio-list')
def get_audio_list() -> Response:
    """
    Get list of available audio files.
    
    Returns:
        Response: JSON list of audio files
    """
    # Use cache unless request specifically asks for fresh data
    use_cache = request.args.get('refresh', '0') != '1'
    audio_files = list_audio_files(use_cache=use_cache)
    
    # Set cache control header to allow browser caching
    response = jsonify(audio_files)
    if use_cache:
        response.headers.add('Cache-Control', f'max-age={CACHE_TIMEOUT}')
    return response


def prepare_visualization_data(filenames: List[str], coordinates: List[List[float]]) -> Dict[str, Any]:
    """
    Prepare visualization data response.
    
    Args:
        filenames: List of audio file names
        coordinates: UMAP coordinates
        
    Returns:
        Dict[str, Any]: Formatted response data
    """
    return {
        'coordinates': coordinates,
        'filenames': filenames
    }


def prepare_similarity_data(
    query_filename: str, 
    similarities: List[Dict[str, Any]], 
    all_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare similarity data for response.
    
    Args:
        query_filename: Query audio filename
        similarities: List of similarity data
        all_features: Dictionary of all features
        
    Returns:
        Dict[str, Any]: Formatted response data
    """
    # Use only scalar features for client-side reranking
    logger.debug("Preparing scalar features for client-side reranking")
    
    if feature_db.is_initialized:
        query_features = feature_db.get_features(query_filename)
    else:
        query_features = all_features[query_filename]
        
    query_scalar_features = {
        k: v for k, v in query_features.items() 
        if not isinstance(v, list)
    }
    
    # Prepare response data
    response_data = {
        'similarities': similarities,
        'filenames': list(all_features.keys()),
        'all_features': {
            'query': query_scalar_features
        }
    }
    
    # Add neighbor features to response for client-side reranking
    for similarity_data in similarities:
        neighbor_filename = similarity_data['filename'].split('.wav')[0]
        if feature_db.is_initialized:
            neighbor_features = feature_db.get_features(neighbor_filename)
        else:
            neighbor_features = all_features[neighbor_filename]
        
        # Only include scalar features for client-side reranking
        response_data['all_features'][neighbor_filename] = {
            k: v for k, v in neighbor_features.items() 
            if not isinstance(v, list)
        }
    
    return response_data


def process_with_database(
    query_filename: str, 
    weights: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Process audio comparison using the feature database.
    
    Args:
        query_filename: Query audio filename
        weights: Feature weights
        
    Returns:
        List[Dict[str, Any]]: List of similarity data
    """
    start_time = time.time()
    logger.debug("Using precomputed similarities from feature database")
    
    # Get precomputed similar files directly from the database
    neighbors_data = feature_db.get_similar_files(
        query_filename, k=config.FEATURE_DB_MAX_NEIGHBORS)
    
    query_time = time.time() - start_time
    logger.info(f"Retrieved {len(neighbors_data)} neighbors from database in {query_time:.4f} seconds")
    
    # Format the neighbors data for response
    similarities = []
    
    # Include the feature deltas for visualization
    for neighbor_filename, similarity_score in neighbors_data:
        # Calculate feature deltas
        feature_deltas = feature_db.calculate_feature_deltas(
            query_filename, neighbor_filename, weights)
        
        similarities.append({
            'filename': f"{neighbor_filename}.wav",
            'similarity': float(similarity_score),
            'cosine_similarity': float(similarity_score),
            'feature_deltas': feature_deltas
        })
    
    return similarities


def process_without_database(
    query_filename: str, 
    all_features: Dict[str, Any],
    weights: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Process audio comparison using on-the-fly calculation.
    
    Args:
        query_filename: Query audio filename
        all_features: Dictionary of all features
        weights: Feature weights
        
    Returns:
        List[Dict[str, Any]]: List of similarity data
    """
    logger.debug("Feature database not initialized, using on-the-fly calculation")
    
    # Use cosine similarity for nearest neighbors (original method)
    neighbors = ap.find_nearest_neighbors_by_cosine_similarity(
        query_filename, all_features, k=5)
    
    logger.debug(f"Found {len(neighbors)} neighbors using cosine similarity")
    
    # Calculate similarities for the neighbors
    similarities = []
    query_features = all_features[query_filename]
    filenames = list(all_features.keys())

    for idx, similarity_score in neighbors:
        neighbor_filename = filenames[idx]
        # Use the provided similarity method for consistency with the rest of the app
        calculated_similarity = ap.calculate_similarity(
            query_features, all_features[neighbor_filename], weights)
        
        # Calculate feature deltas for initial display
        feature_deltas = ap.calculate_feature_deltas(
            query_features, all_features[neighbor_filename], weights)
        
        similarities.append({
            'filename': neighbor_filename,
            'similarity': calculated_similarity,
            'cosine_similarity': float(similarity_score),
            'feature_deltas': feature_deltas
        })
    
    return similarities


@app.route('/audio-features', methods=['POST'])
def process_audio_features() -> Response:
    """
    Process audio features for visualization and comparison.
    
    Returns:
        Response: JSON data for visualization or comparison
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing JSON data in request'}), 400
            
        logger.debug(f"Processing audio features request: {data}")
        
        query_filename = data.get('queryFile', '').split('.wav')[0] if data.get('queryFile') else None
        weights = data.get('weights', {})
        operation = data.get('operation', 'visualize')  # 'visualize' or 'compare'
        
        logger.debug(f"Processing operation: {operation}, query: {query_filename}")

        # Load all features
        all_features = load_features()
        if all_features is None:
            logger.error("Failed to load features from R2")
            return jsonify({'error': 'Audio features data not available'}), 500

        # Get list of features and filenames
        features_list = list(all_features.values())
        if not features_list:
            return jsonify({'error': 'No feature data available'}), 500
        
        filenames = list(all_features.keys())
        
        # UMAP visualization placeholder
        coordinates = []  # Empty placeholder for coordinates
        
        # For visualization requests
        if operation == 'visualize':
            logger.debug("Returning visualization data")
            response_data = prepare_visualization_data(filenames, coordinates)
            return jsonify(response_data)
        
        # For comparison requests
        if not query_filename:
            return jsonify({'error': 'Missing query filename'}), 400
            
        # Check if query file exists for comparison
        if query_filename not in all_features:
            logger.error(f"Query file {query_filename} not found in features")
            return jsonify({'error': 'Query file features not found'}), 404
            
        # Process comparison
        if feature_db.is_initialized and config.FEATURE_DB_ENABLED:
            similarities = process_with_database(query_filename, weights)
        else:
            similarities = process_without_database(query_filename, all_features, weights)
        
        # Prepare final response
        response_data = prepare_similarity_data(query_filename, similarities, all_features)
        return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Error in process_audio_features: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/compare', methods=['POST'])
def compare_audio() -> Response:
    """
    Process audio comparison request.
    
    Returns:
        Response: JSON data for comparison results
    """
    try:
        if not request.is_json:
            logger.warning(f"Invalid content type in compare request: {request.content_type}")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        if not data:
            logger.warning("Empty JSON data in compare request")
            return jsonify({'error': 'Missing JSON data'}), 400
            
        logger.debug(f"Compare request: {data}")
        
        # Add operation field to indicate this is a comparison request
        data['operation'] = 'compare'
        
        # Simply redirect to process_audio_features
        return process_audio_features()
            
    except Exception as e:
        logger.error(f"Error in compare_audio: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# Cache for database stats
_db_stats_cache = None
_db_stats_cache_time = 0


@app.route('/database-stats')
def get_database_stats() -> Response:
    """
    Get statistics about the feature database.
    
    Returns:
        Response: JSON data with database statistics
    """
    global _db_stats_cache, _db_stats_cache_time
    
    # Check if we have a valid cache
    current_time = time.time()
    if _db_stats_cache is not None and (current_time - _db_stats_cache_time) < CACHE_TIMEOUT:
        logger.debug("Using cached database stats")
        return jsonify(_db_stats_cache)
    
    try:
        if not feature_db.is_initialized:
            stats = {
                'initialized': False,
                'message': 'Feature database not initialized'
            }
        else:
            # Get the stats from the feature database
            stats = feature_db.get_database_stats()
            
            # Add additional system info
            stats['feature_weights'] = list(config.FEATURE_WEIGHTS.keys())
            stats['num_audio_files'] = len(list_audio_files(use_cache=True))
        
        # Update cache
        _db_stats_cache = stats
        _db_stats_cache_time = current_time
        
        # Add caching headers
        response = jsonify(stats)
        response.headers.add('Cache-Control', f'max-age={CACHE_TIMEOUT}')
        return response
        
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check() -> Response:
    """
    Simple health check endpoint for monitoring.
    
    Returns:
        Response: JSON status information
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'db_initialized': feature_db.is_initialized,
        'environment': config.ENVIRONMENT
    })
