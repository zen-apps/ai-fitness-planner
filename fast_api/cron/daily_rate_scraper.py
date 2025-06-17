#!/usr/bin/env python3
"""
Script to be run by a cron job to trigger daily rate scraping
This can be scheduled to run once per day
"""

import requests
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"/var/log/rate_scraper/daily_scraper_{datetime.now().strftime('%Y-%m-%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("daily_rate_scraper")

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/v1/rate_tools/scrape_rates/"

# Days ahead to scrape rates for
DAYS_AHEAD = 14


def trigger_scraping():
    """Trigger the rate scraping process via the API"""
    logger.info(f"Triggering rate scraping for {DAYS_AHEAD} days ahead")

    try:
        response = requests.post(API_ENDPOINT, json={"days_ahead": DAYS_AHEAD})

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Successfully triggered scraping: {result['message']}")
            return True
        else:
            logger.error(
                f"Failed to trigger scraping. Status code: {response.status_code}, Response: {response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"Error triggering scraping: {e}")
        return False


if __name__ == "__main__":
    trigger_scraping()
