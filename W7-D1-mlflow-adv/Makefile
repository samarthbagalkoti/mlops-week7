SHELL := /bin/bash

PY := python
ENV := . ./.venv/bin/activate

# Global defaults
VERSION ?= 2
STAGE   ?= Staging
DRY     ?= true
REASON  ?= manual rollback   # <-- no quotes here

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

