.PHONY: build-production
build-production: ## Build the production docker image.
	docker compose -f docker/docker-compose.yml build

.PHONY: build-production-slim
build-production-slim: ## Build the production docker image.
	docker compose -f docker/kwaainet_slim/docker-compose.yml build
