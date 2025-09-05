# Exec Summary — W7: Registry, Gates, Staging Smoke, Rollback, Monitoring

**Problem**: Ship models safely without regressions or performance surprises; recover fast if issues occur.

**Architecture**:
- MLflow Registry (stages + aliases) with promotion gates
- FastAPI inference (loads `models:/name/@staging` / `@production`)
- Staging smoke test (p95 latency + health) as deploy gate
- Governance tags + CI check (owner, use.case, git.sha, data.version, risk.tier, PII)
- Monitoring: Prometheus + Grafana; alerts on p95, error rate
- One-click rollback with audit

**Key Choices**:
- **Gate** on AUC vs current Prod; no regressions allowed by policy
- **p95 latency** budget 200ms; smoke blocks promotion if exceeded
- **Aliases** decouple deploy routing from lifecycle stages
- **Audit**: JSONL log + artifacts/DB attached in CI runs

**Costs/Risks**:
- Extra CI minutes for training/smoke (kept tiny with sklearn demo)
- Latency/metric thresholds need tuning per service

**Next Steps**:
- Canary (10%) + auto-rollback on SLO breach
- Multi-env Terraform + IRSA for production-grade infra
- Expand drift checks (Evidently, feature store stats)

Also

**Demo script** 

Use this exactly while screen-recording (OBS/Teams/Zoom):

**Context (15s)**
“This is a safe MLOps release flow: we use MLflow Registry with stages/aliases, Staging smoke tests on p95 latency, and metric gates. Rollback is one click.”

**Registry & versions (30s)**

Show MLflow UI → Models → w7d1_cancer_classifier, versions v1/v2 (Prod/Staging).

Briefly point out tags (owner, git.sha, risk.tier) and aliases.

**Staging smoke (40s)**

Terminal: make deploy → say “This stages latest, runs smoke (p95≤200ms), runs the metric gate, then promotes to Prod.”

Show green output.

**Monitoring (30s)**

make monitor_up (already running) → Grafana board “W7 Inference Overview”.

Run make smoke_prod → watch p95 and error rate move.

**Governance & Lineage (25s)**

Show governance.tags.yaml and make report artifacts: lineage.mmd, metrics_snapshot.json, audit.jsonl.

**Rollback (20s)**

make dry_rollback → preview; then make rollback → confirm audit entry.

“MTTR is seconds with auditability.”

**Close (10s)**

“This delivers JD bullets: gated deployments, staging validation, governance, monitoring, and fast rollback.” 
