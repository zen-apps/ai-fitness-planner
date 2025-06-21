import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.vectorstores import FAISS
from langchain.schema import Document
import numpy as np
import faiss
from pymongo import MongoClient
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nutrition_vector_search")

nutrition_search = APIRouter()


# Pydantic models for API
class NutritionQuery(BaseModel):
    query: str = Field(..., description="Natural language search query")
    dietary_restrictions: Optional[List[str]] = Field(
        default=[], description="Dietary restrictions like 'vegan', 'gluten-free'"
    )
    macro_goals: Optional[Dict[str, float]] = Field(
        default={},
        description="Macro targets like {'protein_min': 20, 'carbs_max': 10}",
    )
    limit: int = Field(default=10, description="Number of results to return")
    similarity_threshold: float = Field(
        default=0.5, description="Minimum similarity score (0-1)"
    )
    use_full_database: bool = Field(
        default=False, description="Use full database (branded_foods) vs sample (branded_foods_sample)"
    )


class VectorSearchResponse(BaseModel):
    query: str
    results_found: int
    results: List[Dict[str, Any]]
    search_time_ms: int


# Global variables for vector stores and embeddings
vector_store_full = None  # FAISS index for branded_foods
vector_store_sample = None  # FAISS index for branded_foods_sample
embeddings_model = None
mongo_client = None


def get_mongo_client():
    """Get MongoDB client connection"""
    global mongo_client
    if mongo_client is None:
        mongo_user = os.getenv("MONGO_USER")
        mongo_password = os.getenv("MONGO_PASSWORD")

        try:
            # Try Docker service name first
            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mongodb_ai_fitness_planner:27017/admin"
            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command("ping")
        except:
            # Fallback to localhost
            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27019/admin"
            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command("ping")

    return mongo_client


def get_embeddings_model():
    """Initialize OpenAI embeddings model"""
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Cost-effective option
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return embeddings_model


def get_vector_store(use_full_database: bool = False):
    """Get the appropriate FAISS vector store based on database selection"""
    global vector_store_full, vector_store_sample
    
    if use_full_database:
        if vector_store_full is None:
            # Try to load full database vector store
            vector_store_path = "./nutrition_faiss_index_full"
            if os.path.exists(vector_store_path):
                embeddings = get_embeddings_model()
                vector_store_full = FAISS.load_local(
                    vector_store_path, embeddings, allow_dangerous_deserialization=True
                )
            else:
                logger.warning("Full database vector index not found. Falling back to sample.")
                return get_vector_store(use_full_database=False)
        return vector_store_full
    else:
        if vector_store_sample is None:
            # Try to load sample database vector store
            vector_store_path = "./nutrition_faiss_index_sample"
            if os.path.exists(vector_store_path):
                embeddings = get_embeddings_model()
                vector_store_sample = FAISS.load_local(
                    vector_store_path, embeddings, allow_dangerous_deserialization=True
                )
            else:
                # Fallback to old path for backward compatibility
                vector_store_path = "./nutrition_faiss_index"
                if os.path.exists(vector_store_path):
                    embeddings = get_embeddings_model()
                    vector_store_sample = FAISS.load_local(
                        vector_store_path, embeddings, allow_dangerous_deserialization=True
                    )
        return vector_store_sample


def create_food_text_representation(food_item: Dict) -> str:
    """Create a comprehensive text representation of a food item for embedding"""
    parts = []

    # Basic info
    if food_item.get("description"):
        parts.append(f"Food: {food_item['description']}")

    if food_item.get("brandOwner"):
        parts.append(f"Brand: {food_item['brandOwner']}")

    if food_item.get("brandName"):
        parts.append(f"Product: {food_item['brandName']}")

    if food_item.get("foodCategory"):
        parts.append(f"Category: {food_item['foodCategory']}")

    # Ingredients
    if food_item.get("ingredients"):
        ingredients_clean = food_item["ingredients"][:500]  # Limit length
        parts.append(f"Ingredients: {ingredients_clean}")

    # Enhanced nutrition info if available
    nutrition = food_item.get("nutrition_enhanced", {})
    if nutrition:
        per_100g = nutrition.get("per_100g", {})
        if per_100g:
            parts.append(
                f"Per 100g: {per_100g.get('energy_kcal', 0)} calories, "
                f"{per_100g.get('protein_g', 0)}g protein, "
                f"{per_100g.get('total_fat_g', 0)}g fat, "
                f"{per_100g.get('carbs_g', 0)}g carbs"
            )

        macro_breakdown = nutrition.get("macro_breakdown", {})
        if macro_breakdown:
            primary_macro = macro_breakdown.get("primary_macro_category", "")
            if primary_macro != "unknown":
                parts.append(f"Primary macro: {primary_macro}")

    # Serving info
    if food_item.get("servingSize") and food_item.get("servingSizeUnit"):
        parts.append(
            f"Serving: {food_item['servingSize']}{food_item['servingSizeUnit']}"
        )

    if food_item.get("householdServingFullText"):
        parts.append(f"Serving description: {food_item['householdServingFullText']}")

    return " | ".join(parts)


@nutrition_search.post(
    "/search_nutrition_semantic/", response_model=VectorSearchResponse
)
async def search_nutrition_semantic(query_data: NutritionQuery):
    """Semantic search using FAISS vector store with collection-specific indices"""
    start_time = datetime.now()

    try:
        # Get the appropriate vector store based on database selection
        vector_store = get_vector_store(use_full_database=query_data.use_full_database)
        
        if vector_store is None:
            db_type = "full" if query_data.use_full_database else "sample"
            raise HTTPException(
                status_code=404,
                detail=f"Vector index for {db_type} database not found. Please create it first.",
            )

        # Perform semantic search
        docs_and_scores = vector_store.similarity_search_with_score(
            query_data.query, k=query_data.limit * 2  # Get more results to filter
        )

        # Filter results based on similarity threshold
        filtered_results = [
            (doc, score)
            for doc, score in docs_and_scores
            if score >= query_data.similarity_threshold
        ]

        # Apply dietary restrictions and macro goals filtering
        final_results = []
        for doc, score in filtered_results[: query_data.limit]:
            metadata = doc.metadata

            # Check dietary restrictions
            if query_data.dietary_restrictions:
                content_lower = doc.page_content.lower()
                skip = False

                for restriction in query_data.dietary_restrictions:
                    restriction_lower = restriction.lower()
                    if restriction_lower == "vegan":
                        # Simple heuristic - check for animal products
                        animal_products = [
                            "milk",
                            "cheese",
                            "butter",
                            "egg",
                            "meat",
                            "chicken",
                            "beef",
                            "pork",
                            "fish",
                        ]
                        if any(animal in content_lower for animal in animal_products):
                            skip = True
                            break
                    elif restriction_lower == "gluten-free":
                        gluten_sources = ["wheat", "barley", "rye", "gluten"]
                        if any(gluten in content_lower for gluten in gluten_sources):
                            skip = True
                            break

                if skip:
                    continue

            # Check macro goals
            if query_data.macro_goals:
                protein = metadata.get("protein_per_100g", 0)
                fat = metadata.get("fat_per_100g", 0)
                carbs = metadata.get("carbs_per_100g", 0)
                calories = metadata.get("calories_per_100g", 0)

                # Check macro constraints
                skip = False
                for goal, value in query_data.macro_goals.items():
                    if goal == "protein_min" and protein < value:
                        skip = True
                        break
                    elif goal == "protein_max" and protein > value:
                        skip = True
                        break
                    elif goal == "fat_min" and fat < value:
                        skip = True
                        break
                    elif goal == "fat_max" and fat > value:
                        skip = True
                        break
                    elif goal == "carbs_min" and carbs < value:
                        skip = True
                        break
                    elif goal == "carbs_max" and carbs > value:
                        skip = True
                        break
                    elif goal == "calories_min" and calories < value:
                        skip = True
                        break
                    elif goal == "calories_max" and calories > value:
                        skip = True
                        break

                if skip:
                    continue

            # Format result
            result = {
                "fdc_id": metadata.get("fdc_id"),
                "description": metadata.get("description"),
                "brand_owner": metadata.get("brand_owner"),
                "brand_name": metadata.get("brand_name"),
                "food_category": metadata.get("food_category"),
                "similarity_score": float(score),
                "nutrition_per_100g": {
                    "calories": metadata.get("calories_per_100g", 0),
                    "protein_g": metadata.get("protein_per_100g", 0),
                    "fat_g": metadata.get("fat_per_100g", 0),
                    "carbs_g": metadata.get("carbs_per_100g", 0),
                },
                "primary_macro_category": metadata.get("primary_macro", "unknown"),
                "is_high_protein": metadata.get("is_high_protein", False),
                "nutrition_density_score": metadata.get("nutrition_density_score", 0),
                "serving_size": metadata.get("serving_size", 0),
                "matched_content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
            }

            final_results.append(result)

            if len(final_results) >= query_data.limit:
                break

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return VectorSearchResponse(
            query=query_data.query,
            results_found=len(final_results),
            results=final_results,
            search_time_ms=int(processing_time),
        )

    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@nutrition_search.get("/search_nutrition_hybrid/")
async def search_nutrition_hybrid(
    query: str = "high protein snack",
    dietary_restrictions: str = "",  # comma-separated
    protein_min: float = 0,
    carbs_max: float = 999,
    calories_max: float = 999,
    limit: int = 10,
    semantic_weight: float = 0.7,
    use_full_database: bool = False,
):
    """Hybrid search combining traditional MongoDB queries with semantic search"""

    try:
        # Parse dietary restrictions
        restrictions = (
            [r.strip() for r in dietary_restrictions.split(",") if r.strip()]
            if dietary_restrictions
            else []
        )

        # Create query data for semantic search
        query_data = NutritionQuery(
            query=query,
            dietary_restrictions=restrictions,
            macro_goals={
                "protein_min": protein_min,
                "carbs_max": carbs_max,
                "calories_max": calories_max,
            },
            limit=limit * 2,  # Get more results for hybrid scoring
            similarity_threshold=0.5,  # Lower threshold for hybrid approach
            use_full_database=use_full_database,
        )

        # Get semantic search results
        semantic_results = await search_nutrition_semantic(query_data)

        # Get traditional MongoDB search results
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        collection_name = "branded_foods" if use_full_database else "branded_foods_sample"
        branded_foods = db[collection_name]

        # Traditional search query
        mongo_query = {
            "$and": [
                {
                    "$or": [
                        {"description": {"$regex": query, "$options": "i"}},
                        {"brandOwner": {"$regex": query, "$options": "i"}},
                        {"ingredients": {"$regex": query, "$options": "i"}},
                    ]
                },
                {"nutrition_enhanced.per_100g.protein_g": {"$gte": protein_min}},
                {"nutrition_enhanced.per_100g.carbs_g": {"$lte": carbs_max}},
                {"nutrition_enhanced.per_100g.energy_kcal": {"$lte": calories_max}},
            ]
        }

        mongo_results = list(branded_foods.find(mongo_query).limit(limit * 2))

        # Combine and score results
        hybrid_results = {}

        # Add semantic results with semantic weight
        for result in semantic_results.results:
            fdc_id = result["fdc_id"]
            hybrid_results[fdc_id] = {
                **result,
                "hybrid_score": result["similarity_score"] * semantic_weight,
                "semantic_score": result["similarity_score"],
                "traditional_score": 0,
            }

        # Add traditional results with traditional weight
        traditional_weight = 1 - semantic_weight
        for food_item in mongo_results:
            fdc_id = food_item.get("fdcId")

            # Calculate traditional relevance score based on text matches
            description = food_item.get("description", "").lower()
            brand = food_item.get("brandOwner", "").lower()
            ingredients = food_item.get("ingredients", "").lower()
            query_lower = query.lower()

            # Simple relevance scoring
            score = 0
            if query_lower in description:
                score += 0.4
            if query_lower in brand:
                score += 0.3
            if query_lower in ingredients:
                score += 0.3

            # Boost for exact matches
            query_words = query_lower.split()
            for word in query_words:
                if word in description:
                    score += 0.1
                if word in brand:
                    score += 0.05

            traditional_score = min(score, 1.0)  # Cap at 1.0

            if fdc_id in hybrid_results:
                # Update existing result
                hybrid_results[fdc_id]["hybrid_score"] += (
                    traditional_score * traditional_weight
                )
                hybrid_results[fdc_id]["traditional_score"] = traditional_score
            else:
                # Add new result from traditional search
                nutrition = food_item.get("nutrition_enhanced", {})
                per_100g = nutrition.get("per_100g", {})

                hybrid_results[fdc_id] = {
                    "fdc_id": fdc_id,
                    "description": food_item.get("description"),
                    "brand_owner": food_item.get("brandOwner"),
                    "brand_name": food_item.get("brandName"),
                    "food_category": food_item.get("foodCategory"),
                    "nutrition_per_100g": {
                        "calories": per_100g.get("energy_kcal", 0),
                        "protein_g": per_100g.get("protein_g", 0),
                        "fat_g": per_100g.get("total_fat_g", 0),
                        "carbs_g": per_100g.get("carbs_g", 0),
                    },
                    "primary_macro_category": nutrition.get("macro_breakdown", {}).get(
                        "primary_macro_category", "unknown"
                    ),
                    "is_high_protein": nutrition.get("macro_breakdown", {}).get(
                        "is_high_protein", False
                    ),
                    "nutrition_density_score": nutrition.get(
                        "nutrition_density_score", 0
                    ),
                    "serving_size": food_item.get("servingSize", 0),
                    "hybrid_score": traditional_score * traditional_weight,
                    "semantic_score": 0,
                    "traditional_score": traditional_score,
                }

        # Sort by hybrid score and return top results
        sorted_results = sorted(
            hybrid_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]

        return {
            "query": query,
            "search_type": "hybrid",
            "semantic_weight": semantic_weight,
            "traditional_weight": traditional_weight,
            "results_found": len(sorted_results),
            "results": sorted_results,
        }

    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@nutrition_search.post("/create_vector_index_full/")
async def create_vector_index_full(
    batch_size: int = 1000, max_documents: Optional[int] = None, recreate: bool = False
):
    """Create FAISS vector index from full MongoDB nutrition data (branded_foods)"""
    global vector_store_full
    return await _create_vector_index(
        collection_name="branded_foods",
        index_path="./nutrition_faiss_index_full",
        batch_size=batch_size,
        max_documents=max_documents,
        recreate=recreate
    )


@nutrition_search.post("/create_vector_index_sample/")
async def create_vector_index_sample(
    batch_size: int = 1000, max_documents: Optional[int] = None, recreate: bool = False
):
    """Create FAISS vector index from sample MongoDB nutrition data (branded_foods_sample)"""
    global vector_store_sample
    return await _create_vector_index(
        collection_name="branded_foods_sample",
        index_path="./nutrition_faiss_index_sample",
        batch_size=batch_size,
        max_documents=max_documents,
        recreate=recreate
    )


async def _create_vector_index(
    collection_name: str,
    index_path: str,
    batch_size: int = 1000,
    max_documents: Optional[int] = None,
    recreate: bool = False
):
    """Helper function to create FAISS vector index from MongoDB data"""
    start_time = datetime.now()

    try:
        # Check if vector store already exists and we don't want to recreate
        if os.path.exists(index_path) and not recreate:
            logger.info(f"Loading existing FAISS index from {index_path}...")
            embeddings = get_embeddings_model()
            vector_store = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            return {
                "status": "success",
                "message": f"Loaded existing FAISS index for {collection_name}",
                "collection_name": collection_name,
                "index_size": vector_store.index.ntotal if vector_store else 0,
                "index_path": index_path,
            }

        # Get MongoDB connection
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        collection = db[collection_name]

        # Check if collection exists and has data
        total_docs = collection.count_documents({})
        if total_docs == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{collection_name}' not found or empty. Please import data first."
            )

        # Get embeddings model
        embeddings = get_embeddings_model()

        if max_documents:
            total_docs = min(total_docs, max_documents)

        logger.info(f"Processing {total_docs} documents from {collection_name} for vector indexing...")

        documents = []
        processed_count = 0

        # Process documents in batches
        cursor = (
            collection.find({}).limit(max_documents)
            if max_documents
            else collection.find({})
        )

        batch = []
        for food_item in cursor:
            try:
                # Create text representation
                text_content = create_food_text_representation(food_item)

                # Create document
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "fdc_id": food_item.get("fdcId"),
                        "description": food_item.get("description", ""),
                        "brand_owner": food_item.get("brandOwner", ""),
                        "brand_name": food_item.get("brandName", ""),
                        "food_category": food_item.get("foodCategory", ""),
                        "gtin_upc": food_item.get("gtinUpc", ""),
                        "serving_size": food_item.get("servingSize", 0),
                        # Add key nutrition info to metadata for filtering
                        "calories_per_100g": food_item.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("energy_kcal", 0),
                        "protein_per_100g": food_item.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("protein_g", 0),
                        "fat_per_100g": food_item.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("total_fat_g", 0),
                        "carbs_per_100g": food_item.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("carbs_g", 0),
                        "primary_macro": food_item.get("nutrition_enhanced", {})
                        .get("macro_breakdown", {})
                        .get("primary_macro_category", "unknown"),
                        "is_high_protein": food_item.get("nutrition_enhanced", {})
                        .get("macro_breakdown", {})
                        .get("is_high_protein", False),
                        "nutrition_density_score": food_item.get(
                            "nutrition_enhanced", {}
                        ).get("nutrition_density_score", 0),
                    },
                )

                batch.append(doc)
                processed_count += 1

                # Process batch when it reaches batch_size
                if len(batch) >= batch_size:
                    if not documents:  # First batch - create vector store
                        logger.info(
                            f"Creating initial FAISS index with {len(batch)} documents..."
                        )
                        vector_store = FAISS.from_documents(batch, embeddings)
                    else:  # Subsequent batches - add to existing vector store
                        logger.info(
                            f"Adding {len(batch)} documents to existing index..."
                        )
                        additional_store = FAISS.from_documents(batch, embeddings)
                        vector_store.merge_from(additional_store)

                    batch = []
                    logger.info(
                        f"Processed {processed_count}/{total_docs} documents..."
                    )

            except Exception as e:
                logger.warning(
                    f"Error processing document {food_item.get('fdcId', 'unknown')}: {str(e)}"
                )
                continue

        # Process remaining documents
        if batch:
            if 'vector_store' not in locals():
                logger.info(
                    f"Creating FAISS index with remaining {len(batch)} documents..."
                )
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                logger.info(f"Adding final {len(batch)} documents to index...")
                additional_store = FAISS.from_documents(batch, embeddings)
                vector_store.merge_from(additional_store)

        # Save the vector store
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)

        # Update global variables
        if collection_name == "branded_foods":
            global vector_store_full
            vector_store_full = vector_store
        else:
            global vector_store_sample
            vector_store_sample = vector_store

        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Vector index creation completed for {collection_name}!")

        return {
            "status": "success",
            "message": f"FAISS vector index created successfully for {collection_name}",
            "collection_name": collection_name,
            "total_documents_processed": processed_count,
            "index_size": vector_store.index.ntotal,
            "processing_time_seconds": round(processing_time, 2),
            "index_path": index_path,
            "batch_size_used": batch_size,
        }

    except Exception as e:
        logger.error(f"Error creating vector index for {collection_name}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create vector index: {str(e)}"
        )


@nutrition_search.get("/vector_index_status/")
async def get_vector_index_status():
    """Get status of both vector indices"""
    try:
        status = {
            "full_database": {
                "index_path": "./nutrition_faiss_index_full",
                "exists": os.path.exists("./nutrition_faiss_index_full"),
                "collection_name": "branded_foods"
            },
            "sample_database": {
                "index_path": "./nutrition_faiss_index_sample",
                "exists": os.path.exists("./nutrition_faiss_index_sample"),
                "collection_name": "branded_foods_sample"
            },
            "legacy_index": {
                "index_path": "./nutrition_faiss_index",
                "exists": os.path.exists("./nutrition_faiss_index"),
                "note": "Legacy index - used as fallback for sample database"
            }
        }
        
        # Get index sizes if they exist
        for db_type in ["full_database", "sample_database"]:
            if status[db_type]["exists"]:
                try:
                    embeddings = get_embeddings_model()
                    vector_store = FAISS.load_local(
                        status[db_type]["index_path"], 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    status[db_type]["index_size"] = vector_store.index.ntotal
                    status[db_type]["embedding_dimension"] = vector_store.index.d
                except Exception as e:
                    status[db_type]["error"] = str(e)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting vector index status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get index status: {str(e)}"
        )


