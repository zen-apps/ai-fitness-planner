# ai-fitness-planner

download usda json branded foods data 
@workout.get("/load_usda_data/")

load data into mongo
@workout.post("/import_usda_data/")

search by product 
@workout.get("/search_nutrition/")

returns doc count and fields 
@workout.get("/check_usda_data/")
{
  "status": "data_exists",
  "collection": "branded_foods",
  "document_count": 452998,
  "sample_fields": [
    "_id",
    "foodClass",
    "description",
    "foodNutrients",
    "foodAttributes",
    "modifiedDate",
    "availableDate",
    "marketCountry",
    "brandOwner",
    "gtinUpc",
    "dataSource",
    "ingredients",
    "servingSize",
    "servingSizeUnit",
    "householdServingFullText",
    "labelNutrients",
    "tradeChannels",
    "microbes",
    "brandedFoodCategory",
    "dataType",
    "fdcId",
    "publicationDate",
    "foodUpdateLog"
  ],
  "message": "Found 452,998 branded food documents"
}