from app import app

# This allows running in development mode with python main.py
if __name__ == "__main__":
    # Validate configuration before starting
    from config import Config
    import logging
    import os
    
    config = Config()
    logger = logging.getLogger(__name__)
    
    config_issues = config.validate()
    if config_issues:
        for issue, message in config_issues.items():
            logger.warning(f"Configuration issue: {issue} - {message}")
            
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 10000))
    
    # Only use debug mode if explicitly enabled
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    
    app.run(debug=debug, host="0.0.0.0", port=port)
