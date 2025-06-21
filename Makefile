VERSION := 0.0.1
BE_IMAGE_NAME := ai-fitness-planner-backend
DOCKER_COMPOSE_FILE  := docker-compose.yml 
FE_IMAGE_NAME := ai-fitness-planner-frontend

# Main commands
up: 
	docker-compose -f $(DOCKER_COMPOSE_FILE)  up -d --build

logs:
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f --tail 15 

logs-be:
	docker logs $(BE_IMAGE_NAME) -f --tail 150 

logs-fe:
	docker logs $(FE_IMAGE_NAME) -f --tail 150

# Tiered nutrition database setup
setup-demo: up
	@echo "🚀 Setting up DEMO mode - Quick start with sample data (2 minutes)"
	@echo "Perfect for: Blog demos, LangGraph testing, immediate results"
	docker-compose exec fast_api python /app/../scripts/setup_database.py --mode=demo
	@echo "✅ Demo setup complete! Your AI Fitness Planner is ready for testing."

setup-full: up  
	@echo "🔥 Setting up FULL mode - Complete USDA dataset (15 minutes)"
	@echo "Perfect for: Production use, complete nutrition database, blog showcase"
	docker-compose exec fast_api python /app/../scripts/setup_database.py --mode=full
	@echo "✅ Full setup complete! Production-ready with 300K+ foods."


# Development helpers  
clean-db:
	@echo "🧹 Cleaning nutrition database..."
	docker-compose exec mongodb_ai_fitness_planner mongosh --eval "use usda_nutrition; db.branded_foods.drop();"
	@echo "✅ Database cleaned. Run make setup-demo or make setup-full to repopulate."

# Help
help:
	@echo "AI Fitness Planner - Available Commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup-demo     - 2min setup with sample data (perfect for demos)"
	@echo "  make setup-full     - 15min setup with full USDA dataset (production)"
	@echo ""
	@echo "📊 Database:"
	@echo "  make db-stats       - Show database statistics"  
	@echo "  make test-search    - Test nutrition search functionality"
	@echo "  make clean-db       - Clean nutrition database"
	@echo ""
	@echo "🛠️ Development:"
	@echo "  make up            - Start all services"
	@echo "  make logs          - Show all service logs"
	@echo "  make logs-be       - Show backend logs"
	@echo "  make logs-fe       - Show frontend logs"
	@echo ""
	@echo "💡 For LangChain blog features:"
	@echo "  1. make setup-demo  (instant gratification)"
	@echo "  2. make setup-full  (complete showcase)"

.PHONY: up logs logs-be logs-fe setup-demo setup-full db-stats test-search clean-db help

