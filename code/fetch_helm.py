"""Download HELM MMLU v1.3.0 per-instance predictions (all models, all subjects)."""
import json, subprocess, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE = "https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output"
OUT = Path("data/raw/helm")

def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            dest.write_bytes(data)
            return
        except Exception as e:  # retry transient failures
            err = e
    print("FAIL", url, err, file=sys.stderr)

runs = json.loads(urllib.request.urlopen(f"{BASE}/releases/v1.3.0/runs_to_run_suites.json").read())
(OUT / "runs_to_run_suites.json").write_text(json.dumps(runs))
jobs, seen_subject = [], set()
for run, suite in runs.items():
    if "method=multiple_choice_joint" not in run:
        continue
    subject = run.split("subject=")[1].split(",")[0]
    model = run.split("model=")[1].split(",")[0]
    jobs.append((f"{BASE}/runs/{suite}/{run}/display_predictions.json", OUT / "pred" / model / f"{subject}.json"))
    jobs.append((f"{BASE}/runs/{suite}/{run}/instances.json", OUT / "inst" / model / f"{subject}.json"))
print(len(jobs), "files")
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda j: get(*j), jobs))
print("done")
