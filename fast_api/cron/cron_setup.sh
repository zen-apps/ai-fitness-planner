#!/bin/bash
# Setup script for creating the cron job for daily rate scraping

# Create log directory if it doesn't exist
mkdir -p /var/log/rate_scraper
chmod 755 /var/log/rate_scraper

# Make the daily scraper script executable
chmod +x daily_rate_scraper.py

# Create a cron job entry - runs at 1:00 AM every day
CRON_ENTRY="0 1 * * * $(pwd)/daily_rate_scraper.py"

# Add the cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Cron job has been set up to run daily at 1:00 AM"
echo "You can verify with: crontab -l"