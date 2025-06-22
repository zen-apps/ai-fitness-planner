import os
import json
import logging
from fastapi import APIRouter, HTTPException
import pymongo


# Set up logging with a more specific name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("workout_optimization")

nutrition_setup = APIRouter()


# Fixed MongoDB connection test endpoint
@nutrition_setup.get("/test_mongo_db/")
async def test_mongo_db():
    """Test MongoDB connection with improved error handling"""
    from pymongo import MongoClient, errors

    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_db_name = os.getenv("MONGO_DB_NAME")

    mongo_hosts_to_try = [
        "mongodb_ai_fitness_planner",  # Docker service name
        "localhost",  # If running locally
    ]

    for host in mongo_hosts_to_try:
        try:
            logger.info(f"Attempting to connect to MongoDB at host: {host}")

            # Fixed URI format - connect to admin database for authentication
            if host == "mongodb_ai_fitness_planner":
                port = 27017  # Internal Docker port
            else:
                port = 27019  # External mapped port

            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{host}:{port}/admin"

            logger.info(
                f"Using MongoDB URI: mongodb://{mongo_user}:***@{host}:{port}/admin"
            )

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)

            # Test the connection
            client.admin.command("ping")

            # Access your specific database
            db = client[mongo_db_name] if mongo_db_name else client["usda_nutrition"]
            collection_names = db.list_collection_names()

            # Test if we can access existing collections
            existing_collections = []
            for collection_name in collection_names[:5]:  # Limit to first 5
                try:
                    count = db[collection_name].count_documents({})
                    existing_collections.append(
                        {"name": collection_name, "document_count": count}
                    )
                except Exception as e:
                    existing_collections.append(
                        {"name": collection_name, "error": str(e)}
                    )

            client.close()

            logger.info(f"Successfully connected to MongoDB at host: {host}")
            return {
                "status": "success",
                "connected_host": host,
                "port": port,
                "database_name": mongo_db_name or "usda_nutrition",
                "total_collections": len(collection_names),
                "collections_sample": existing_collections,
                "environment_vars": {
                    "MONGO_USER": mongo_user,
                    "MONGO_PASSWORD": "***" if mongo_password else None,
                    "MONGO_DB_NAME": mongo_db_name,
                },
            }

        except errors.ServerSelectionTimeoutError as err:
            logger.warning(
                f"Server selection timeout for MongoDB at host {host}: {str(err)}"
            )
            continue
        except errors.OperationFailure as err:
            logger.warning(
                f"Authentication failed for MongoDB at host {host}: {str(err)}"
            )
            continue
        except Exception as err:
            logger.warning(
                f"Unexpected error connecting to MongoDB at host {host}: {str(err)}"
            )
            continue

    # If all hosts failed
    error_msg = (
        f"Failed to connect to MongoDB. Tried hosts: {', '.join(mongo_hosts_to_try)}"
    )
    logger.error(error_msg)
    raise HTTPException(status_code=500, detail=error_msg)


# Also add a simpler connection helper function
def get_mongo_client():
    """Get MongoDB client connection"""
    from pymongo import MongoClient

    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")

    # Try Docker service name first, then localhost
    try:
        # Docker internal connection
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mongodb_ai_fitness_planner:27017/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # Test connection
        return client
    except:
        # Fallback to localhost
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27019/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # Test connection
        return client


@nutrition_setup.get("/search_nutrition/")
async def search_nutrition(query: str = "coca cola", limit: int = 1):
    """Search nutrition data to test MongoDB functionality"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        # Search by brand owner or description
        search_query = {
            "$or": [
                {"brandOwner": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
            ]
        }

        results = list(db.branded_foods.find(search_query).limit(limit))

        # Format results for display
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "fdc_id": result.get("fdcId"),
                    "description": result.get("description"),
                    "brand_owner": result.get("brandOwner"),
                    "brand_name": result.get("brandName"),
                    "food_class": result.get("foodClass"),
                    "food_category": result.get("foodCategory"),
                    "gtin_upc": result.get("gtinUpc"),
                    "ingredients": result.get("ingredients"),
                    "serving_size": result.get("servingSize"),
                    "serving_size_unit": result.get("servingSizeUnit"),
                    "household_serving_fulltext": result.get(
                        "householdServingFullText"
                    ),
                    "modified_date": result.get("modifiedDate"),
                    "available_date": result.get("availableDate"),
                    "market_country": result.get("marketCountry"),
                    "discontinued_date": result.get("discontinuedDate"),
                    "preparation_state_code": result.get("preparationStateCode"),
                    "trade_channel": result.get("tradeChannel"),
                    "short_description": result.get("shortDescription"),
                    "nutrition_enhanced": result.get("nutrition_enhanced", {}),
                    "food_nutrients": result.get("foodNutrients", []),
                    "food_attributes": result.get("foodAttributes", []),
                    "food_attribute_types": result.get("foodAttributeTypes", []),
                    "food_version_ids": result.get("foodVersionIds", []),
                }
            )

        client.close()

        return {
            "query": query,
            "results_found": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        logger.error(f"Error searching nutrition data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search nutrition data: {str(e)}"
        )


@nutrition_setup.get("/database_stats/")
async def get_database_stats():
    """Get current database statistics"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        collections = db.list_collection_names()
        stats = {"collections": {}}

        for collection_name in collections:
            try:
                collection = db[collection_name]
                count = collection.count_documents({})

                # Get a sample document to show structure
                sample = collection.find_one()
                fields = list(sample.keys()) if sample else []

                stats["collections"][collection_name] = {
                    "document_count": count,
                    "sample_fields": fields[:10],  # First 10 fields
                    "total_fields": len(fields),
                }
            except Exception as e:
                stats["collections"][collection_name] = {"error": str(e)}

        # Get database stats
        db_stats = db.command("dbstats")

        client.close()

        return {
            "database_name": os.getenv("MONGO_DB_NAME", "usda_nutrition"),
            "total_collections": len(collections),
            "collection_details": stats["collections"],
            "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
            "storage_size_mb": round(db_stats.get("storageSize", 0) / (1024 * 1024), 2),
            "indexes": db_stats.get("indexes", 0),
        }

    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@nutrition_setup.post("/import_sampled_data/")
async def import_sampled_data(
    sample_file: str = "./fast_api/app/api/nutrition_data/samples/usda_sampled_5000_foods.json",
):
    """Import sampled USDA data into MongoDB 'branded_foods_sample' collection for quick setup"""

    try:
        # Check if file exists
        if not os.path.exists(sample_file):
            raise HTTPException(
                status_code=404, detail=f"Sample file not found: {sample_file}"
            )

        # Get MongoDB connection
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        branded_foods = db["branded_foods_sample"]

        logger.info("Creating enhanced indexes...")
        # Create indexes with error handling to avoid conflicts
        try:
            # Basic indexes
            branded_foods.create_index([("foodClass", pymongo.ASCENDING)])
            branded_foods.create_index([("brandOwner", pymongo.ASCENDING)])
            branded_foods.create_index([("foodCategory", pymongo.ASCENDING)])
            branded_foods.create_index([("gtinUpc", pymongo.ASCENDING)])

            # Create compound text index for search
            branded_foods.create_index(
                [("description", pymongo.TEXT), ("ingredients", pymongo.TEXT)],
                name="search_text_index",
            )

            # Nutrition indexes
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.primary_macro_category",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.is_high_protein",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_fat", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_carb", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_balanced", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.protein_g", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.energy_kcal", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.nutrition_density_score", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.protein_percent",
                        pymongo.DESCENDING,
                    )
                ]
            )

            logger.info("All indexes created successfully")

        except Exception as e:
            logger.warning(
                f"Some indexes may already exist or failed to create: {str(e)}"
            )
            # Continue with import even if some indexes fail

        # Load and import sampled data
        logger.info(f"Loading sampled data from {sample_file}...")

        with open(sample_file, "r") as f:
            sample_data = json.load(f)

        foods = sample_data.get("BrandedFoods", [])
        metadata = sample_data.get("metadata", {})

        logger.info(f"Importing {len(foods)} sampled foods...")

        batch_size = 1000
        total_processed = 0

        for i in range(0, len(foods), batch_size):
            batch = foods[i : i + batch_size]
            try:
                branded_foods.insert_many(batch)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed}/{len(foods)} foods...")
            except Exception as e:
                logger.error(f"Error in batch: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

        final_count = branded_foods.count_documents({})
        enhanced_final_count = branded_foods.count_documents(
            {"nutrition_enhanced": {"$exists": True}}
        )
        client.close()

        logger.info("Sampled data import completed successfully!")

        return {
            "status": "success",
            "message": "Sampled USDA data imported successfully",
            "sample_metadata": metadata,
            "total_documents_imported": total_processed,
            "enhanced_documents": enhanced_final_count,
            "final_document_count": final_count,
            "source_file": sample_file,
            "indexes_created": [
                "foodClass",
                "brandOwner",
                "foodCategory",
                "gtinUpc",
                "search_text_index (description + ingredients)",
                "nutrition_enhanced.macro_breakdown.primary_macro_category",
                "nutrition_enhanced.macro_breakdown.is_high_protein",
                "nutrition_enhanced.macro_breakdown.is_high_fat",
                "nutrition_enhanced.macro_breakdown.is_high_carb",
                "nutrition_enhanced.macro_breakdown.is_balanced",
                "nutrition_enhanced.per_100g.protein_g (desc)",
                "nutrition_enhanced.per_100g.energy_kcal (asc)",
                "nutrition_enhanced.nutrition_density_score (desc)",
                "nutrition_enhanced.macro_breakdown.protein_percent (desc)",
            ],
        }

    except Exception as e:
        logger.error(f"Error importing sampled data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to import sampled data: {str(e)}"
        )


@nutrition_setup.get("/database_availability/")
async def check_database_availability():
    """Check which databases are available (full vs sample)"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        # Check if full database exists and has data
        full_collection = db["branded_foods"]
        full_count = full_collection.count_documents({})
        full_available = full_count > 0

        # Check if sample database exists and has data
        sample_collection = db["branded_foods_sample"]
        sample_count = sample_collection.count_documents({})
        sample_available = sample_count > 0

        client.close()

        return {
            "full_database": {
                "available": full_available,
                "document_count": full_count,
                "collection_name": "branded_foods",
            },
            "sample_database": {
                "available": sample_available,
                "document_count": sample_count,
                "collection_name": "branded_foods_sample",
            },
            "recommendation": (
                "full" if full_available else "sample" if sample_available else "none"
            ),
        }

    except Exception as e:
        logger.error(f"Error checking database availability: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check database availability: {str(e)}"
        )
