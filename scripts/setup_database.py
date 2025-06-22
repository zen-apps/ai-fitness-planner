#!/usr/bin/env python3
"""
Database setup script for AI Fitness Planner
Handles demo USDA nutrition database setup
"""

import argparse
import sys
import requests
import time
from pathlib import Path

# Add the fast_api directory to Python path
sys.path.append('/app/fast_api')
sys.path.append('/app')

def download_usda_sample_data():
    """Download USDA sample data from GitHub releases"""
    url = "https://github.com/zen-apps/ai-fitness-planner/releases/download/v1.0.0/usda_sampled_5000_foods.json"
    sample_data_path = Path("/app/fast_api/app/api/nutrition_data/samples/usda_sampled_5000_foods.json")
    
    # Create directory if it doesn't exist
    sample_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    if sample_data_path.exists():
        print(f"✅ Sample data already exists at {sample_data_path}")
        return True
    
    print(f"📥 Downloading USDA sample data from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(sample_data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded sample data to {sample_data_path}")
        return True
        
    except requests.RequestException as e:
        print(f"❌ Failed to download sample data: {e}")
        return False

def wait_for_api():
    """Wait for the FastAPI server to be ready"""
    api_base = "http://fast_api_ai_fitness_planner:8000"
    max_retries = 30
    retry_delay = 2
    
    print("⏳ Waiting for FastAPI server to be ready...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_base}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ FastAPI server is ready!")
                return True
        except requests.RequestException:
            pass
        
        if attempt < max_retries - 1:
            print(f"⏳ Attempt {attempt + 1}/{max_retries} - waiting {retry_delay}s...")
            time.sleep(retry_delay)
    
    print("❌ FastAPI server did not become ready in time")
    return False

def setup_demo_database():
    """Setup demo database with sample data"""
    print("🚀 Setting up DEMO database with sample data...")
    
    # Download sample data if needed
    if not download_usda_sample_data():
        return False
    
    if not wait_for_api():
        return False
    
    api_base = "http://fast_api_ai_fitness_planner:8000"
    
    try:
        # Test MongoDB connection
        print("🔍 Testing MongoDB connection...")
        response = requests.get(f"{api_base}/v1/nutrition_setup/test_mongo_db/")
        if response.status_code != 200:
            print(f"❌ MongoDB connection failed: {response.text}")
            return False
        print("✅ MongoDB connection successful")
        
        # Import sample data
        print("📥 Importing sample USDA data...")
        response = requests.post(f"{api_base}/v1/nutrition_setup/import_sampled_data/")
        if response.status_code != 200:
            print(f"❌ Failed to import sample data: {response.text}")
            return False
        
        result = response.json()
        print(f"✅ Successfully imported {result.get('imported_count', 'N/A')} foods")
        
        # Get database stats
        print("📊 Getting database statistics...")
        response = requests.get(f"{api_base}/v1/nutrition_setup/database_stats/")
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Database contains {stats.get('total_foods', 'N/A')} foods")
        
        print("🎉 Demo database setup complete!")
        return True
        
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup AI Fitness Planner database")
    parser.add_argument(
        "--mode", 
        choices=["demo"], 
        default="demo",
        help="Setup mode: 'demo' for quick sample data (default: demo)"
    )
    
    args = parser.parse_args()
    
    print(f"🏃‍♂️ Starting {args.mode.upper()} database setup...")
    
    success = setup_demo_database()
    
    if success:
        print("✅ Database setup completed successfully!")
        print("🌐 Access your application:")
        print("   Frontend: http://localhost:8526")
        print("   API Docs: http://localhost:1015/docs")
        print("   MongoDB UI: http://localhost:8084")
        sys.exit(0)
    else:
        print("❌ Database setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()