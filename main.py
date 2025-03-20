"""
PROPRIETARY AND CONFIDENTIAL

Copyright (c) 2025-2026. All Rights Reserved.

NOTICE: All information contained herein is, and remains the property of the owner.
The intellectual and technical concepts contained herein are proprietary and may be
covered by U.S. and Foreign Patents, patents in process, and are protected by trade
secret or copyright law. Dissemination of this information or reproduction of this
material is strictly forbidden unless prior written permission is obtained from the
owner. Access to the source code contained herein is hereby forbidden to anyone except
current employees or contractors of the owner who have executed Confidentiality and
Non-disclosure Agreements explicitly covering such access.

THE RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT
CONVEY OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO
MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

Usage of this source code is subject to a strict license agreement. Unauthorized
reproduction, modification, or distribution, in part or in whole, is strictly prohibited.
License terms available upon request.
"""

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
