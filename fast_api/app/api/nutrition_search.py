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


# Global variables for vector store and embeddings
vector_store = None
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
    """Semantic search using FAISS vector store
    
    Note: Vector search currently only supports the collection that was indexed.
    The use_full_database parameter is passed but may not affect results if
    the vector index was built on a different collection.
    """
    global vector_store

    start_time = datetime.now()

    try:
        # Load vector store if not already loaded
        if vector_store is None:
            vector_store_path = "./nutrition_faiss_index"
            if not os.path.exists(vector_store_path):
                raise HTTPException(
                    status_code=404,
                    detail="Vector index not found. Please create it first using /create_vector_index/",
                )

            embeddings = get_embeddings_model()
            vector_store = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
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


