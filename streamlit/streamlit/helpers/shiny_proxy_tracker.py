import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import json
import socket
import psutil

# Database setup
pd.set_option("display.max_rows", 500)
pd.options.display.max_columns = None

# Get database URI from environment
DATABASE_URI = os.getenv("DATABASE_URI")
if not DATABASE_URI:
    DATABASE_URI = "postgresql://joshjanzen:normal@ai_clone_db:5432/ai_clone"

# Create database engine
engine = create_engine(DATABASE_URI)


class ShinyProxyTracker:
    def __init__(self, engine):
        self.engine = engine
        self.ensure_table_exists()

    def ensure_table_exists(self):
        """Create the shinyproxy_user_activity table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS shinyproxy_user_activity (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            app_name VARCHAR(255),
            session_id VARCHAR(255),
            event_type VARCHAR(100) NOT NULL,
            event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            ip_address INET,
            user_agent TEXT,
            container_id VARCHAR(255),
            memory_usage_mb INTEGER,
            cpu_usage_percent DECIMAL(5,2),
            duration_seconds INTEGER,
            error_message TEXT,
            additional_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_shinyproxy_username ON shinyproxy_user_activity(username);
        CREATE INDEX IF NOT EXISTS idx_shinyproxy_timestamp ON shinyproxy_user_activity(event_timestamp);
        CREATE INDEX IF NOT EXISTS idx_shinyproxy_event_type ON shinyproxy_user_activity(event_type);
        CREATE INDEX IF NOT EXISTS idx_shinyproxy_app_name ON shinyproxy_user_activity(app_name);
        """
        
        try:
            with self.engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
                print("✅ Database table 'shinyproxy_user_activity' ensured to exist")
        except Exception as e:
            print(f"❌ Error creating table: {e}")
            print(f"Database URI: {DATABASE_URI}")
            import traceback
            print(traceback.format_exc())

    def get_shinyproxy_username(self):
        """Get username from ShinyProxy environment variables"""
        # ShinyProxy sets these environment variables in the container
        username = os.getenv("SHINYPROXY_USERNAME")
        if not username:
            # Check for other ShinyProxy-related environment variables
            username = os.getenv("SHINYPROXY_USER_EMAIL") or os.getenv("SHINYPROXY_USER")
        if not username:
            # Fallback options
            username = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
        return username

    def get_shinyproxy_metadata(self):
        """Get ShinyProxy-specific metadata from environment"""
        metadata = {
            "shinyproxy_app": os.getenv("SHINYPROXY_APP_INSTANCE"),
            "shinyproxy_container_id": os.getenv("HOSTNAME"),  # Usually container ID
            "shinyproxy_usergroups": os.getenv("SHINYPROXY_USERGROUPS"),
            "shinyproxy_app_name": os.getenv("SHINYPROXY_APP_NAME"),
        }
        # Remove None values
        return {k: v for k, v in metadata.items() if v is not None}

    def get_system_info(self):
        """Get basic system information"""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                "memory_usage_mb": int(memory_info.used / (1024 * 1024)),
                "cpu_usage_percent": cpu_percent,
            }
        except:
            return {"memory_usage_mb": None, "cpu_usage_percent": None}

    def log_activity(
        self,
        event_type,
        app_name=None,
        session_id=None,
        duration_seconds=None,
        error_message=None,
        additional_data=None,
    ):
        """
        Log user activity to the database

        Args:
            event_type: Type of event ('login', 'app_start', 'app_stop', 'logout', 'error')
            app_name: Name of the Shiny app
            session_id: Unique session identifier
            duration_seconds: Duration for app sessions
            error_message: Error message if applicable
            additional_data: Dictionary of additional metadata
        """
        try:
            # Check if any ShinyProxy environment variables exist - if not, don't log
            shinyproxy_env_vars = [
                os.getenv("SHINYPROXY_USERNAME"),
                os.getenv("SHINYPROXY_USER_EMAIL"), 
                os.getenv("SHINYPROXY_USER"),
                os.getenv("SHINYPROXY_APP_INSTANCE")
            ]
            if not any(shinyproxy_env_vars):
                print("No ShinyProxy environment variables found - skipping activity logging")
                return False
            
            # Get user info
            username = self.get_shinyproxy_username()

            # Get ShinyProxy metadata
            sp_metadata = self.get_shinyproxy_metadata()

            # Get system info
            sys_info = self.get_system_info()

            # Combine additional metadata
            metadata = {}
            if additional_data:
                metadata.update(additional_data)
            metadata.update(sp_metadata)

            # Create the data record
            activity_data = {
                "username": username,
                "app_name": app_name or sp_metadata.get("shinyproxy_app_name"),
                "session_id": session_id,
                "event_type": event_type,
                "event_timestamp": datetime.now(),
                "ip_address": self._get_client_ip(),
                "user_agent": os.getenv("HTTP_USER_AGENT"),
                "container_id": sp_metadata.get("shinyproxy_container_id"),
                "memory_usage_mb": sys_info.get("memory_usage_mb"),
                "cpu_usage_percent": sys_info.get("cpu_usage_percent"),
                "duration_seconds": duration_seconds,
                "error_message": error_message,
                "additional_metadata": json.dumps(metadata) if metadata else None,
            }

            # Convert to DataFrame
            df = pd.DataFrame([activity_data])

            # Insert into database
            df.to_sql(
                "shinyproxy_user_activity",
                con=self.engine,
                if_exists="append",
                index=False,
            )

            print(f"✅ Logged activity: {event_type} for user {username}")
            return True

        except Exception as e:
            print(f"❌ Error logging activity: {str(e)}")
            print(f"Database URI: {DATABASE_URI}")
            print(f"Activity data: {activity_data}")
            import traceback
            print(traceback.format_exc())
            return False

    def _get_client_ip(self):
        """Attempt to get client IP address"""
        # Try various environment variables that might contain client IP
        ip_vars = [
            "HTTP_X_FORWARDED_FOR",
            "HTTP_X_REAL_IP",
            "HTTP_CLIENT_IP",
            "REMOTE_ADDR",
        ]

        for var in ip_vars:
            ip = os.getenv(var)
            if ip:
                # Handle comma-separated IPs (X-Forwarded-For)
                return ip.split(",")[0].strip()

        try:
            # Fallback to local IP
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except:
            return None

    def get_user_activity(self, username=None, limit=100):
        """Retrieve user activity from database"""
        query = """
        SELECT * FROM shinyproxy_user_activity 
        """

        if username:
            query += f" WHERE username = '{username}'"

        query += " ORDER BY event_timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql(query, con=self.engine)


# Initialize tracker
tracker = ShinyProxyTracker(engine)


# Example usage functions
def track_app_start(app_name=None, session_id=None):
    """Track when a user starts an app"""
    return tracker.log_activity("app_start", app_name=app_name, session_id=session_id)


def track_app_stop(app_name=None, session_id=None, duration_seconds=None):
    """Track when a user stops an app"""
    return tracker.log_activity(
        "app_stop",
        app_name=app_name,
        session_id=session_id,
        duration_seconds=duration_seconds,
    )


def track_login():
    """Track user login"""
    return tracker.log_activity("login")


def track_logout():
    """Track user logout"""
    return tracker.log_activity("logout")


def track_error(error_message, app_name=None):
    """Track an error"""
    return tracker.log_activity("error", app_name=app_name, error_message=error_message)


# Auto-track login when module is imported (optional)
if __name__ == "__main__":
    # Example: Track login when script starts
    track_login()

    # Example: Get current user's activity
    current_user = tracker.get_shinyproxy_username()
    recent_activity = tracker.get_user_activity(username=current_user, limit=10)
    print(f"\nRecent activity for {current_user}:")
    print(recent_activity[["event_type", "event_timestamp", "app_name"]].to_string())

# Example of how to use in your Shiny app:
"""
# At the start of your Shiny app:
import uuid
session_id = str(uuid.uuid4())
track_app_start(app_name="My Shiny App", session_id=session_id)

# At the end of your Shiny app or in cleanup:
track_app_stop(app_name="My Shiny App", session_id=session_id, duration_seconds=300)

# For error tracking:
try:
    # your app code
    pass
except Exception as e:
    track_error(str(e), app_name="My Shiny App")
"""

""" table does not exist yet, but could look like this:
-- Create table for tracking ShinyProxy user activity
CREATE TABLE shinyproxy_user_activity (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    app_name VARCHAR(255),
    session_id VARCHAR(255),
    event_type VARCHAR(100) NOT NULL, -- 'login', 'app_start', 'app_stop', 'logout', 'error'
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    container_id VARCHAR(255),
    memory_usage_mb INTEGER,
    cpu_usage_percent DECIMAL(5,2),
    duration_seconds INTEGER, -- for app sessions
    error_message TEXT,
    additional_metadata JSONB, -- for flexible additional data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_shinyproxy_username ON shinyproxy_user_activity(username);
CREATE INDEX idx_shinyproxy_timestamp ON shinyproxy_user_activity(event_timestamp);
CREATE INDEX idx_shinyproxy_event_type ON shinyproxy_user_activity(event_type);
CREATE INDEX idx_shinyproxy_app_name ON shinyproxy_user_activity(app_name);

-- Create a view for easy querying of active sessions
CREATE VIEW active_shinyproxy_sessions AS
SELECT 
    username,
    app_name,
    session_id,
    event_timestamp as session_start,
    ip_address,
    container_id
FROM shinyproxy_user_activity 
WHERE event_type = 'app_start' 
AND session_id NOT IN (
    SELECT DISTINCT session_id 
    FROM shinyproxy_user_activity 
    WHERE event_type = 'app_stop' 
    AND session_id IS NOT NULL
);
"""
