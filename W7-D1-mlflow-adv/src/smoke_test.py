import argparse, asyncio, time, statistics, json, os
import httpx
import numpy as np

DEFAULT_PAYLOAD = {
  # Provide 2 sample rows (length will be checked against model.n_features_in_)
  "rows": [
    [0.1]*30,
    [0.9]*30
  ]
}

async def one_call(client, url, payload):
  t0 = time.perf_counter()
  r = await client.post(url, json=payload)
  dt = (time.perf_counter() - t0) * 1000.0
  return r.status_code, dt, r.text

async def run(base_url, requests, concurrency, p95_budget_ms):
  # 1) Health
  async with httpx.AsyncClient(timeout=10.0) as c:
    h = await c.get(f"{base_url}/healthz")
    if h.status_code != 200:
      print(f"[SMOKE] /healthz failed: {h.status_code} {h.text}")
      return 1
    print(f"[SMOKE] /healthz OK: {h.text[:120]}...")

  # 2) Predict load
  latencies = []
  ok = 0
  payload = DEFAULT_PAYLOAD

  async def worker(sem, tasks_done):
    nonlocal ok
    async with httpx.AsyncClient(timeout=10.0) as c:
      while True:
        async with sem:
          if tasks_done["n"] >= requests:
            return
          tasks_done["n"] += 1
        code, dt, body = await one_call(c, f"{base_url}/predict", payload)
        latencies.append(dt)
        if code == 200:
          ok += 1

  sem = asyncio.Semaphore(concurrency)
  tasks_done = {"n": 0}
  workers = [asyncio.create_task(worker(sem, tasks_done)) for _ in range(concurrency)]
  await asyncio.gather(*workers)

  if ok == 0:
    print("[SMOKE] All predict calls failed")
    return 1

  p95 = float(np.percentile(latencies, 95))
  res = {
    "requests": requests,
    "ok": ok,
    "p95_ms": round(p95, 2),
    "budget_ms": p95_budget_ms
  }
  print("[SMOKE] Result:", json.dumps(res))
  return 0 if p95 <= p95_budget_ms else 2

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", default="http://127.0.0.1:8080")
  ap.add_argument("--requests", type=int, default=50)
  ap.add_argument("--concurrency", type=int, default=5)
  ap.add_argument("--p95-budget-ms", type=float, default=200.0)
  args = ap.parse_args()
  raise SystemExit(asyncio.run(run(args.base_url, args.requests, args.concurrency, args.p95_budget_ms)))

