.PHONY: build-production-api
build-production-api: ## Build the production docker image.
	docker compose -f docker/kwaainet_api/docker-compose.yml build

.PHONY: build-production-server
build-production-server: ## Build the production docker image.
	docker compose -f docker/kwaainet_node/docker-compose.yml build

.PHONY: build-production-bootstrap
build-production-bootstrap: ## Build the production docker image.
	docker compose -f docker/kwaainet_bootstrap/docker-compose.yml build

.PHONY: build-production-server-intel
build-production-server-intel: ## Build the production docker image.
	docker compose -f docker/kwaainet_node_intel/docker-compose.yml build