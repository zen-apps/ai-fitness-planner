VERSION := 0.0.1
BE_IMAGE_NAME := ai-fitness-planner-backend
DOCKER_COMPOSE_FILE  := docker-compose-dev.yml 
FE_IMAGE_NAME := ai-fitness-planner-frontend

up: 
	docker-compose -f $(DOCKER_COMPOSE_FILE)  up -d --build

logs:
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f --tail 15 

logs-be:
	docker logs $(BE_IMAGE_NAME) -f --tail 150 

logs-fe:
	docker logs $(FE_IMAGE_NAME) -f --tail 150

