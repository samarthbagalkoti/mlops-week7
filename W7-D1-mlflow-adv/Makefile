SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
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
	. .venv/bin/activate && MODEL_STAGE=Staging uvicorn src.serve_app:app --host 0.0.0.0 --port 8085

serve_production:
	. .venv/bin/activate && MODEL_STAGE=Production uvicorn src.serve_app:app --host 0.0.0.0 --port 8085

smoke:
	. .venv/bin/activate && python src/smoke_test.py --base-url http://54.147.138.39:8082 --requests 60 --concurrency 8 --p95-budget-ms 200

docker_build:
	docker build -t $(IMG) .

docker_run:
	docker run --rm -p 8083:8083 \
	  -e MODEL_STAGE=Production \
	  -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
	  $(IMG)

# ========= STAGING → SMOKE → GATE → PROD =========
promote_latest_safe:
	@. .venv/bin/activate; \
	VER=$$(python src/get_latest.py); \
	if [ -z "$$VER" ]; then echo "No versions found"; exit 1; fi; \
	echo "[INFO] Latest candidate = v$$VER"; \
	echo "[STEP] Promote v$$VER to Staging (dry-run=false)"; \
	python src/promote.py --candidate-version $$VER --to Staging --dry-run false --reason "auto stage for smoke" --promoted-by "ci"; \
	echo "[STEP] Start staging service (background)"; \
	(MODEL_STAGE=Staging uvicorn src.serve_app:app --host 0.0.0.0 --port 8080 >/tmp/staging.log 2>&1 &) ; \
	sleep 3; \
	echo "[STEP] Run smoke (p95<=200ms)"; \
	python src/smoke_test.py --base-url http://54.147.138.39:8082 --requests 60 --concurrency 8 --p95-budget-ms 200; \
	echo "[STEP] Gate against Production"; \
	python src/compare_and_gate.py --candidate-version $$VER; \
	echo "[STEP] Promote v$$VER to Production (dry-run=false)"; \
	python src/promote.py --candidate-version $$VER --to Production --dry-run false --reason "smoke+gate passed" --promoted-by "ci"; \
	echo "[DONE] Production updated."

# ========= ONE-CLICK ROLLBACK =========
rollback_preview:
	@. .venv/bin/activate; python src/rollback.py --reason "manual preview" --dry-run true

rollback_apply:
	@. .venv/bin/activate; python src/rollback.py --reason "manual rollback (W7:D4)" --dry-run false


# --- D5 governance & discoverability ---

SET ?=


tag_latest:
	. .venv/bin/activate; \
	VER=$$(python src/get_latest.py); \
	python src/tag_version.py --version $$VER --set "$(SET)"

#gov_check_staging:
#	. .venv/bin/activate && python src/check_required_tags.py --stage Staging

#gov_check_prod:
#	. .venv/bin/activate && python src/check_required_tags.py --stage Production

alias_sync:
	. .venv/bin/activate && python src/alias_sync.py

export_lineage:
	. .venv/bin/activate && python src/export_lineage.py

gov_check_staging:
	source .venv/bin/activate
	python - <<-'PY'
	import os
	from mlflow import MlflowClient
	name = os.environ.get("MODEL_NAME","w7d1_cancer_classifier")
	c = MlflowClient()
	staging = max(
	    (mv for mv in c.search_model_versions(f"name='{name}'") if mv.current_stage=="Staging"),
	    key=lambda mv:int(mv.version)
	)
	c.set_model_version_tag(
	    name=name, version=staging.version,
	    key="data.schema.features",
	    value=os.environ.get("DATA_FEATURES","30")
	)
	print(f"Tagged Staging v{staging.version} data.schema.features=",
	      os.environ.get("DATA_FEATURES","30"))
	PY


# --- D6 monitoring ---
serve_prod:
	. .venv/bin/activate && MODEL_STAGE=Production uvicorn src.serve_app:app --host 0.0.0.0 --port 8086

monitor_up:
	cd monitoring && docker compose up -d

monitor_down:
	cd monitoring && docker compose down

smoke_prod:
	. .venv/bin/activate && python src/smoke_test.py --base-url http://54.147.138.39:8085 --requests 120 --concurrency 12 --p95-budget-ms 200

build_reference:
	. .venv/bin/activate && python src/build_reference.py

drift_check:
	. .venv/bin/activate && python src/drift_check.py

# Optional: send Slack webhook on drift (set SLACK_WEBHOOK_URL env)
drift_alert:
	@. .venv/bin/activate; \
	python src/drift_check.py || ( \
	  if [ -n "$$SLACK_WEBHOOK_URL" ]; then \
	    echo "[ALERT] Sending Slack webhook..."; \
	    curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"W7 drift alert: see outputs/drift_report.json\"}" $$SLACK_WEBHOOK_URL >/dev/null || true; \
	  fi; \
	  exit 3 \
	)

