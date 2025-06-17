import os
import sys
import random
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pymongo import MongoClient
from selenium_driverless.types.by import By
from selenium_driverless import webdriver as driverless_webdriver

# Set up logging
logger = logging.getLogger(__name__)

# Create the router
rate_tools = APIRouter()


# MongoDB connection setup
def get_mongodb_connection():
    """Get MongoDB connection from environment variables"""
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise HTTPException(
            status_code=500, detail="MongoDB URI not found in environment variables"
        )

    client = MongoClient(mongo_uri)
    return client


# Pydantic models
class ZipCodePair(BaseModel):
    origin_zip: str = Field(description="Origin 3-digit zip code")
    destination_zip: str = Field(description="Destination 3-digit zip code")
    status: str = Field(description="Status of the scraping (pending, success, failed)")
    last_updated: datetime = Field(description="Last updated timestamp")


class RateScrapingResult(BaseModel):
    origin_zip: str = Field(description="Origin 3-digit zip code")
    destination_zip: str = Field(description="Destination 3-digit zip code")
    pickup_date: str = Field(description="Pickup date in YYYY-MM-DD format")
    rate: float = Field(description="Scraped shipping rate")
    currency: str = Field(description="Currency of the rate")
    status: str = Field(description="Status of the scraping (success, failed)")
    error_message: Optional[str] = Field(
        description="Error message if scraping failed", default=None
    )
    timestamp: datetime = Field(description="Timestamp of when the rate was scraped")


# Helper function to generate all 3-digit zip code combinations
def generate_zip_code_pairs():
    """Generate all combinations of 3-digit zip codes"""
    # This is a simplified version - in reality, you'd need to get the valid 3-digit zip prefixes
    # For demonstration, we'll use a range from 100-999
    valid_zips = [str(i).zfill(3) for i in range(100, 1000)]
    zip_pairs = []

    # For demo purposes, let's limit to a smaller subset to avoid creating a massive list
    # In production, you might want to handle this differently or process in batches
    sample_origins = random.sample(valid_zips, 30)  # Adjust as needed
    sample_destinations = random.sample(valid_zips, 30)  # Adjust as needed

    for origin in sample_origins:
        for destination in sample_destinations:
            if origin != destination:  # Skip same origin-destination
                zip_pairs.append(
                    {
                        "origin_zip": origin,
                        "destination_zip": destination,
                        "status": "pending",
                        "last_updated": datetime.now(),
                    }
                )

    return zip_pairs


# Initialize the database with zip code pairs if needed
@rate_tools.post("/initialize_zip_pairs/")
async def initialize_zip_pairs():
    """Initialize the database with all zip code pairs if not already present"""
    try:
        client = get_mongodb_connection()
        db = client["shipping_rates"]
        collection = db["zip_pairs"]

        # Check if we already have entries
        count = collection.count_documents({})

        if count == 0:
            # Generate all zip pairs
            zip_pairs = generate_zip_code_pairs()

            # Insert into MongoDB
            if zip_pairs:
                collection.insert_many(zip_pairs)

            return {
                "status": "success",
                "message": f"Initialized {len(zip_pairs)} zip code pairs",
            }
        else:
            return {
                "status": "info",
                "message": f"Database already contains {count} zip code pairs",
            }

    except Exception as e:
        logger.error(f"Error initializing zip pairs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Function to scrape shipping rates for a zip code pair
async def scrape_shipping_rate(origin_zip: str, destination_zip: str, pickup_date: str):
    """Scrape shipping rate from RXO.com for a specific zip code pair and date"""
    options = driverless_webdriver.ChromeOptions()
    browser_args = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-accelerated-2d-canvas",
        "--disable-gpu",
        "--window-size=1920,1080",
        "--remote-debugging-port=9222",
        "--headless",  # Run in headless mode
    ]

    for arg in browser_args:
        options.add_argument(arg)

    result = {
        "origin_zip": origin_zip,
        "destination_zip": destination_zip,
        "pickup_date": pickup_date,
        "rate": 0.0,
        "currency": "USD",
        "status": "failed",
        "error_message": None,
        "timestamp": datetime.now(),
    }

    try:
        async with driverless_webdriver.Chrome(options=options) as driver:
            target = await driver.current_target
            url = "https://rxo.com/get-a-quote/"
            logger.info(f"Navigating to {url} for {origin_zip} to {destination_zip}")
            await target.get(url, wait_load=True)

            # Wait for page to fully load
            await asyncio.sleep(5)

            # Fill form - similar to your existing code but simplified
            # Select 'Truckload' service
            try:
                truckload_radio = await target.find_element(
                    By.XPATH, "//input[@id='choice_2_3_0']", timeout=5
                )
                await truckload_radio.click()
                logger.info("Selected Truckload service")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error selecting Truckload service: {e}")
                result["error_message"] = f"Error selecting Truckload service: {e}"
                return result

            # Fill pickup location with origin zip
            try:
                pickup_input = await target.find_element(By.ID, "input_2_63", timeout=5)
                await pickup_input.clear()
                await pickup_input.send_keys(origin_zip)
                logger.info(f"Filled pickup location: {origin_zip}")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error filling pickup location: {e}")
                result["error_message"] = f"Error filling pickup location: {e}"
                return result

            # Fill delivery location with destination zip
            try:
                delivery_input = await target.find_element(
                    By.ID, "input_2_67", timeout=5
                )
                await delivery_input.clear()
                await delivery_input.send_keys(destination_zip)
                logger.info(f"Filled delivery location: {destination_zip}")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error filling delivery location: {e}")
                result["error_message"] = f"Error filling delivery location: {e}"
                return result

            # Fill pickup date
            try:
                date_input = await target.find_element(By.ID, "input_2_10", timeout=5)
                await date_input.clear()
                await date_input.send_keys(pickup_date)
                logger.info(f"Filled pickup date: {pickup_date}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error filling pickup date: {e}")
                result["error_message"] = f"Error filling pickup date: {e}"
                return result

            # Fill other required fields like commodity, etc.
            try:
                commodity_input = await target.find_element(
                    By.ID, "input_2_20", timeout=5
                )
                await commodity_input.clear()
                await commodity_input.send_keys("General Freight")
                logger.info("Filled commodity: General Freight")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error filling commodity: {e}")
                result["error_message"] = f"Error filling commodity: {e}"
                return result

            # Fill total weight
            try:
                weight_input = await target.find_element(By.ID, "input_2_35", timeout=5)
                await weight_input.clear()
                await weight_input.send_keys("30000")
                logger.info("Filled total weight: 30000")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error filling total weight: {e}")
                result["error_message"] = f"Error filling total weight: {e}"
                return result

            # Select equipment type
            try:
                dry_van_radio = await target.find_element(
                    By.ID, "choice_2_130_0", timeout=5
                )
                await dry_van_radio.click()
                logger.info("Selected Dry Van equipment")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error selecting equipment type: {e}")
                result["error_message"] = f"Error selecting equipment type: {e}"
                return result

            # Fill business email
            try:
                email_input = await target.find_element(By.ID, "input_2_116", timeout=5)
                await email_input.clear()
                await email_input.send_keys("test@example.com")
                logger.info("Filled business email: test@example.com")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error filling business email: {e}")
                result["error_message"] = f"Error filling business email: {e}"
                return result

            # Handle reCAPTCHA - would need to implement based on your solve_2captcha function
            # For now, skip this or implement a similar approach to your existing code

            # Click Next button
            try:
                next_button = await target.find_element(
                    By.ID, "gform_next_button_2_22", timeout=5
                )
                await next_button.click()
                logger.info("Clicked Next button to submit the first page")
                await asyncio.sleep(8)  # Wait for quote to load
            except Exception as e:
                logger.error(f"Error clicking Next button: {e}")
                result["error_message"] = f"Error clicking Next button: {e}"
                return result

            # Extract quote information
            try:
                quote_amount_element = await target.find_element(
                    By.XPATH,
                    "//div[contains(@class, 'tltotalrate')]/span[1]",
                    timeout=10,
                )
                quote_amount = await quote_amount_element.text
                quote_amount = quote_amount.replace("$", "").strip()

                result["rate"] = float(quote_amount)
                result["status"] = "success"
                result["error_message"] = None
                logger.info(f"Successfully scraped rate: ${quote_amount}")

            except Exception as e:
                logger.error(f"Error extracting quote information: {e}")
                result["error_message"] = f"Error extracting quote information: {e}"
                return result

            return result

    except Exception as e:
        logger.error(f"Error during scraping: {e}", exc_info=True)
        result["error_message"] = f"Error during scraping: {e}"
        return result


# Background task to process all pending zip code pairs
async def process_all_zip_pairs(days_ahead: int):
    """Process all pending zip code pairs and scrape rates for the next N days"""
    try:
        client = get_mongodb_connection()
        db = client["shipping_rates"]
        zip_pairs_collection = db["zip_pairs"]
        rates_collection = db["rates"]

        # Get all pending zip pairs
        pending_pairs = list(zip_pairs_collection.find({"status": "pending"}))
        logger.info(f"Processing {len(pending_pairs)} pending zip code pairs")

        # Generate dates to scrape
        dates_to_scrape = []
        for i in range(1, days_ahead + 1):
            future_date = datetime.now() + timedelta(days=i)
            dates_to_scrape.append(future_date.strftime("%m/%d/%Y"))

        # Process each zip pair for each date
        for zip_pair in pending_pairs:
            origin = zip_pair["origin_zip"]
            destination = zip_pair["destination_zip"]

            for pickup_date in dates_to_scrape:
                # Check if we already have a recent rate for this combination
                existing_rate = rates_collection.find_one(
                    {
                        "origin_zip": origin,
                        "destination_zip": destination,
                        "pickup_date": pickup_date,
                        "timestamp": {
                            "$gt": datetime.now() - timedelta(days=1)
                        },  # Only consider data newer than 1 day
                    }
                )

                if not existing_rate:
                    try:
                        # Scrape the rate
                        rate_result = await scrape_shipping_rate(
                            origin, destination, pickup_date
                        )

                        # Save to MongoDB
                        rates_collection.insert_one(rate_result)

                        # Update zip pair status
                        zip_pairs_collection.update_one(
                            {"_id": zip_pair["_id"]},
                            {
                                "$set": {
                                    "status": rate_result["status"],
                                    "last_updated": datetime.now(),
                                }
                            },
                        )

                        logger.info(
                            f"Processed {origin} to {destination} for {pickup_date}: {rate_result['status']}"
                        )

                        # Add a small delay to avoid overwhelming the site
                        await asyncio.sleep(2)

                    except Exception as e:
                        logger.error(f"Error processing {origin} to {destination}: {e}")

                        # Update zip pair status to failed
                        zip_pairs_collection.update_one(
                            {"_id": zip_pair["_id"]},
                            {
                                "$set": {
                                    "status": "failed",
                                    "last_updated": datetime.now(),
                                }
                            },
                        )
                else:
                    logger.info(
                        f"Skipping {origin} to {destination} for {pickup_date}: Recent data exists"
                    )

        logger.info("Completed processing all pending zip pairs")

    except Exception as e:
        logger.error(f"Error in background task: {e}", exc_info=True)


# API endpoint to trigger rate scraping for all pending zip pairs
@rate_tools.post("/scrape_rates/")
async def scrape_rates(background_tasks: BackgroundTasks, days_ahead: int = 14):
    """
    Trigger the scraping of shipping rates for all pending zip code pairs

    Args:
        days_ahead: Number of days ahead to scrape rates for (default: 14)
    """
    try:
        # Add the task to the background
        background_tasks.add_task(process_all_zip_pairs, days_ahead)

        return {
            "status": "success",
            "message": f"Started rate scraping for the next {days_ahead} days in the background",
        }

    except Exception as e:
        logger.error(f"Error triggering rate scraping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint to get rate statistics
@rate_tools.get("/rate_stats/")
async def get_rate_stats():
    """Get statistics about the rate scraping process"""
    try:
        client = get_mongodb_connection()
        db = client["shipping_rates"]
        zip_pairs_collection = db["zip_pairs"]
        rates_collection = db["rates"]

        # Get counts
        total_pairs = zip_pairs_collection.count_documents({})
        pending_pairs = zip_pairs_collection.count_documents({"status": "pending"})
        success_pairs = zip_pairs_collection.count_documents({"status": "success"})
        failed_pairs = zip_pairs_collection.count_documents({"status": "failed"})

        total_rates = rates_collection.count_documents({})
        success_rates = rates_collection.count_documents({"status": "success"})
        failed_rates = rates_collection.count_documents({"status": "failed"})

        # Get some example rate data
        recent_rates = list(
            rates_collection.find({"status": "success"}).sort("timestamp", -1).limit(5)
        )

        # Format for response
        recent_rates_formatted = []
        for rate in recent_rates:
            recent_rates_formatted.append(
                {
                    "origin_zip": rate["origin_zip"],
                    "destination_zip": rate["destination_zip"],
                    "pickup_date": rate["pickup_date"],
                    "rate": rate["rate"],
                    "timestamp": rate["timestamp"].isoformat(),
                }
            )

        return {
            "zip_pairs": {
                "total": total_pairs,
                "pending": pending_pairs,
                "success": success_pairs,
                "failed": failed_pairs,
            },
            "rates": {
                "total": total_rates,
                "success": success_rates,
                "failed": failed_rates,
            },
            "recent_successful_rates": recent_rates_formatted,
        }

    except Exception as e:
        logger.error(f"Error getting rate stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Schedule daily rate scraping
@rate_tools.post("/schedule_daily_scraping/")
async def schedule_daily_scraping():
    """
    Set up a daily schedule for rate scraping

    This endpoint doesn't actually set up the schedule (that would typically be done with a task scheduler
    like Celery or a cron job), but it returns instructions on how to do so.
    """
    return {
        "status": "info",
        "message": "Daily scraping should be set up using a cron job or task scheduler",
        "example_cron": "0 1 * * * curl -X POST http://your-api-url/v1/rate_tools/scrape_rates/",
        "example_systemd_timer": "OnCalendar=*-*-* 01:00:00",
    }
