> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Modal H200 development lab

Status: CLI 1.6.3 installed in `.venv-modal`; runner syntax and dry-run passed.
October30 update: CLI authenticated to akshaysanthosh; $0 spend cap verified in dashboard.
H200 experiments completed; read RESULTS.md and results/modal for evidence.
Reverify live billing before future execution.

User authorized Modal free credits only, with $0 spending limit. Before EACH execution:
1. Inspect the correct Modal workspace's Usage & Billing dashboard.
2. Set and visibly verify net **Spend limit $0**. A $30 usage budget alone is NOT equivalent.
3. Verify remaining applicable compute credits (at least $3 buffer for this bounded run).
4. Authenticate CLI to that same workspace, using a short-lived token; user completes browser authorization.
5. Record workspace, verified_at (UTC ISO8601), spend_limit_usd: 0, and
   remaining_compute_credits_usd in results/modal/billing-verified.json only from observed evidence.
6. Confirm CLI workspace matches that record before launching. Recheck usage after each run.

Commands from problem root:
```
.venv-modal/bin/modal token new --profile okc-free --expires-in 1d
.venv-modal/bin/python modal_lab/run_candidate.py candidates/lt/lt_original_h0.py
.venv-modal/bin/python modal_lab/run_candidate.py candidates/lt/lt_original_h0.py --execute
```

One ephemeral app, one H200, 4 CPU cores, 16GiB host memory, 900s function timeout,
zero retries, 2s scaledown. CPU image builds can also consume credits. No persistent volume,
web endpoint, cron, paid subscription, or official challenge credentials are uploaded.
Only reference/checks (development harness), champion, selected candidate and evaluator are uploaded.
The billing evidence file is a local launch interlock, not a substitute for Modal's server-side spend limit.

The evaluator compares candidate vs champion with fresh generated weights and inputs,
public correctness cases plus native-path cases, fresh-output lifetime, graph replay correctness,
and paired CUDA-event timing with a 128MiB L2 flush outside each measured replay.
No fixed graph overhead is subtracted. These are development estimates, NOT official scores;
confirm improvements using official free checks before a scored submission. The reference is
only imported by the development evaluator, never by the self-contained submitted candidate.

GPU linking, execution, graph capture and performance are still unverified. On first run,
inspect build errors and actual native dispatch before optimizing. Keep the 1.227411869x champion intact.
