.PHONY: build-production
build-production: ## Build the production docker image.
	docker compose -f docker/docker-compose.yml build

.PHONY: build-production-api
build-production-api: ## Build the production docker image.
	docker compose -f docker/kwaainet_api/docker-compose.yml build

.PHONY: build-production-server
build-production-server: ## Build the production docker image.
	docker compose -f docker/kwaainet_server/docker-compose.yml build

.PHONY: build-production-bootstrap
build-production-bootstrap: ## Build the production docker image.
	docker compose -f docker/kwaainet_bootstrap_peers/docker-compose.yml build

.PHONY: build-production-health
build-production-health: ## Build the production docker image.
	docker compose -f docker/kwaainet_health/docker-compose.yml build