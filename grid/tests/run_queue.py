"""Run the recorded jobs in id order like Slurm would (afterok dependencies); the analysis job is only listed."""
import json, os, subprocess, sys
q = sys.argv[1]
done = {}
for f in sorted(os.listdir(q)):
    if not f.endswith(".json"):
        continue
    j = json.load(open(f"{q}/{f}"))
    dep = j["opts"].get("dependency", "")
    deps = [int(x) for x in dep.split(":")[1:]] if dep else []
    if any(done.get(d) != 0 for d in deps):
        print(f"job {j['id']} {j['opts'].get('job-name')}: cancelled (dependency {dep} not satisfied)")
        done[j["id"]] = "cancelled"; continue
    if j["opts"].get("job-name") == "c2s-analyze":
        print(f"job {j['id']} c2s-analyze: would start now (dependency {dep}); not run in the mock")
        done[j["id"]] = "skipped"; continue
    env = dict(j["env"], SLURM_JOB_ID=str(j["id"]), SLURM_CPUS_PER_TASK="8")
    with open(f"{q}/{j['id']}.out", "w") as log:
        rc = subprocess.call(["bash", j["script"]], env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"job {j['id']} {j['opts'].get('job-name')} time={j['opts'].get('time')}: exit {rc}")
    done[j["id"]] = rc
json.dump(done, open(f"{q}/states.json", "w"))
