VERSION := 0.0.1
BE_CONTAINER_NAME := fast_api_ai_fitness_planner
DOCKER_COMPOSE_FILE  := docker-compose.yml 
FE_CONTAINER_NAME := streamlit_app_ai_fitness_planner

up: 
	docker-compose -f $(DOCKER_COMPOSE_FILE)  up -d --build

logs:
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f --tail 15 

logs-be:
	docker logs $(BE_CONTAINER_NAME) -f --tail 150 

logs-fe:
	docker logs $(FE_CONTAINER_NAME) -f --tail 150

setup-demo: up
	@echo "🚀 Setting up DEMO mode - Quick start with sample data (2 minutes)"
	@echo "Perfect for: Blog demos, LangGraph testing, immediate results"
	@sleep 10  # Give services time to start
	docker-compose exec fast_api_ai_fitness_planner python /app/scripts/setup_database.py --mode=demo
	@echo "✅ Demo setup complete! Your AI Fitness Planner is ready for testing."

clean-db:
	@echo "🧹 Cleaning nutrition database..."
	docker-compose exec mongodb_ai_fitness_planner mongosh --eval "use usda_nutrition; db.branded_foods.drop();"
	@echo "✅ Database cleaned. Run make setup-demo to repopulate."

help:
	@echo "AI Fitness Planner - Available Commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup-demo     - 2min setup with sample data (perfect for demos)"
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

.PHONY: up logs logs-be logs-fe setup-demo db-stats test-search clean-db help

# Cleanup commands
clean:
	@echo "🧼 Cleaning up Docker images and containers..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down --volumes --remove-orphans
	@echo "✅ All containers and volumes removed."
