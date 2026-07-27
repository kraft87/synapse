.PHONY: up logs test schema

## up: build and start the full stack (postgres, poller, mcp-server, dream)
up:
	docker compose up -d --build

## logs: follow logs for all services
logs:
	docker compose logs -f --tail=100

## test: run the test suite (needs SYNAPSE_TEST_URL pointing at a Postgres with the schema
##       applied — pytest.ini reads it from the repo-root .env via pytest-dotenv, so no
##       export is needed locally. Provision/refresh that DB with:
##       psql "$$ADMIN_DSN" -c 'CREATE DATABASE synapse_test' && scripts/apply_schema.sh "$$SYNAPSE_TEST_URL")
test:
	uv run pytest -x -q

## schema: apply all numbered migrations in schema/ to $$SYNAPSE_DB_URL
schema:
	scripts/apply_schema.sh "$$SYNAPSE_DB_URL"
