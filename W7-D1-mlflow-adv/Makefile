SHELL := /bin/bash

PY := python
ENV := . ./.venv/bin/activate

# Global defaults
VERSION ?= 2
STAGE   ?= Staging
DRY     ?= true
REASON  ?= manual rollback   # <-- no quotes here
IMG ?= w7-production-svc:latest

.PHONY: setup train_v1 train_v2 list gate promote rollback dry_rollback

setup:
	$(ENV) && pip install -r requirements.txt

train_v1:
	$(ENV) && $(PY) src/train.py --seed 1

train_v2:
	$(ENV) && $(PY) src/train.py --seed 2

list:
	$(ENV) && $(PY) src/list_registry.py

gate:
	$(ENV) && $(PY) src/compare_and_gate.py --candidate-version $(VERSION)

promote:
	$(ENV) && $(PY) src/promote.py --candidate-version $(VERSION) --to "$(STAGE)" --dry-run $(DRY)

rollback:
	@echo "Reason: $(REASON)"
	$(ENV) && $(PY) src/rollback.py --reason "$(REASON)" --dry-run false

dry_rollback:
	$(ENV) && $(PY) src/rollback.py --reason "dry-run check" --dry-run true

serve_staging:
	. .venv/bin/activate && MODEL_STAGE=Staging uvicorn src.serve_app:app --host 0.0.0.0 --port 8082

serve_production:
	. .venv/bin/activate && MODEL_STAGE=Production uvicorn src.serve_app:app --host 0.0.0.0 --port 8082

smoke:
	. .venv/bin/activate && python src/smoke_test.py --base-url http://54.147.138.39:8082 --requests 60 --concurrency 8 --p95-budget-ms 200

docker_build:
	docker build -t $(IMG) .

docker_run:
	docker run --rm -p 8083:8083 \
	  -e MODEL_STAGE=Production \
	  -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
	  $(IMG)

