---
name: robotwin-benchmark
description: Run ROBOTWIN evaluation with the installed benchmark and policy Nodes.
---

# ROBOTWIN evaluation

The g05_external profile attaches to an already running, user-configured XPolicyLab G05 websocket server. It owns only the client, never starts or stops that external server. G05 model setup is a prerequisite; an unavailable server must fail startup.

1. Start the selected installed profile and require runtime status `ready`.
2. Query `robotwin.policy.describe`; require `ready=true` and the expected action dimension (14).
3. Query `robotwin.benchmark.describe`. Use its actual suite, task table and limits. Do not infer task IDs from names or invent dataset coverage.
4. Start `robotwin.benchmark.run` as a Session. Default smoke is task_ids=[0], init_state_ids=[0], num_runs=1, seed=0. Use the selected profile's suite. Ask for the desired evaluation scope when not provided.
5. Poll Session status, consume terminal events, then retrieve its result. Report completed_episodes, total_episodes, success_rate, failure_summary and result_file. Check the benchmark-owned result JSON for protocol-specific metrics and seed records.
6. To stop a batch, stop the Session and wait for its terminal result; the current episode finishes first. Do not kill an external model service.
7. A second batch may run after the first reaches a terminal state. Stop the Skill when finished; check the Gateway and owned Node processes are gone.

Treat missing weights, assets, runtime errors and partial/stopped batches as such. Unit tests or a stub policy do not establish model accuracy. A debug dataset/subset score must never be described as a full official benchmark score. For source-vs-official acceptance, align checkpoint hashes, preprocessing, tasks, episode seeds, reset and termination semantics, and preserve raw traces/results.
