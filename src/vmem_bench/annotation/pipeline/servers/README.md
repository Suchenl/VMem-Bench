# Annotation console (frontend + backend)

> **Who needs this:** only maintainers who want to *re-build* the gold annotations
> at scale. To simply **use** VMem-Bench (run/score a method), you do **not** need
> this console — download the published gold from Hugging Face and run the scoring
> CLIs. See the repo `README.md` and `docs/benchmark/running_eval.md`.

This subsystem was built against an internal GPU cluster reachable over SSH and an
internal HTTPS reverse proxy that enforces SSO. All internal names have been
replaced with neutral placeholders. **Adapt them to your own infrastructure** (see
[Adapt to your own cluster / SSO](#adapt-to-your-own-cluster--sso) below).

## Why the UI sometimes "won't open"

Three failure modes look similar in the browser:

1. **Process died** — an agent/`pkill` stopped `:8890`/`:7864`, or a crash left a
   stale pidfile. After a remote instance restart/snapshot, run
   `ensure_console.sh` again (or keep `--watch`).
2. **SSO session expired** — if you front the console with an SSO reverse proxy,
   the supported entry is your HTTPS host, which terminates TLS and proxies to
   instance `:8890`. When the SSO session expires, `/api/*` returns `302` → a login
   HTML page. Refresh the page to re-login; this is not a backend crash.
3. **Slow catalog under load** — `/api/samples` can take 15–25s on networked
   storage and could trip gateway timeouts while jobs run. The backend caches the
   catalog for a few seconds to avoid this.

## One command (preferred)

```bash
cd VMem-Bench
bash src/vmem_bench/annotation/pipeline/servers/ensure_console.sh
```

- Brings frontend/backend up if unhealthy
- Starts a **watchdog** by default (re-checks ~20s, restarts on repeated failures)
- Prints local + LAN URLs

Other modes:

```bash
bash .../ensure_console.sh --status   # health only
bash .../ensure_console.sh --once     # repair once, no watch
bash .../ensure_console.sh --stop     # stop watch + FE + BE
bash .../start_all.sh                 # start once (no watch)
```

## Access

| URL | When to use |
|-----|-------------|
| `https://<your-sso-host>/` | **Primary** entry if you deploy behind an SSO reverse proxy → instance `:8890` |
| `http://127.0.0.1:8890` | Local health/debug on the development machine itself |
| `http://<instance-ip>:8890` | Optional LAN debug (subject to your network policy) |

The frontend binds to development-machine port **8890**. Backend traffic stays
loopback-only on `7864`; the frontend proxies browser `/api/*` to it. SSH port
forwarding is not part of this path and is not required.

## Fleet naming and status

The console reads `MEMSTRATA_NODES_TSV` when available. Every endpoint is
displayed as:

```text
idle|busy|starting|break|broke · cluster/node/rank · role/model · start time
```

`break` is an explicit operator pause and is excluded from new dispatches
without killing the process. `broke` means an endpoint exited, failed health
checks, or stopped heartbeating; it is already excluded from dispatch.

`ensure_console.sh` also starts `fleet_health_monitor`. Every online VLM service
gets one continuously overwritten liveness log at:

```text
runtime/services/vlm_fleet/health/<cluster>/node<N>/rank<R>.log
```

The monitor removes that log when the service is no longer online. Manual
multimodal task probes are stored separately as `rank<R>.task.log`, so a
service-liveness refresh never overwrites task evidence.

Logs: `VMem-Bench/data/_services/annotation_console/logs/`

## VLM `--allowed-local-media-path`

vLLM accepts **one directory**. Do **not** pass comma lists like `/data,/tmp`
(treated as a single nonexistent path → every `file://` media review returns
HTTP 400). Launchers default to `/data` (a shared storage root for clip caches);
point this at your own media root.

## Adapt to your own cluster / SSO

The remote-dispatch layer expects your GPU nodes to be reachable over passwordless
SSH and listed in a `nodes.tsv`. Everything is neutral and overridable:

| Placeholder | What to change it to | Where |
|-------------|----------------------|-------|
| `execution_target: "remote"` | keep as-is; `"local"` runs on the dev box | frontend job form / API body |
| `REMOTE_CLUSTER_PREFIX` (default `gpu-`) | your cluster-label prefix, e.g. `dgx-`, `hpc-` | env var read by `backend/remote_dispatch.py` |
| cluster labels `gpu-a800`, `gpu-h800` | your own node/cluster labels in `nodes.tsv` | `nodes.tsv`, `--cluster` flags |
| `MEMSTRATA_NODES_TSV` / `TGPU_NODES_FILE` | path to your `nodes.tsv` | env var |
| `MEMSTRATA_REMOTE_*` (penalties/timeouts) | tune to your cluster if needed | env vars, `backend/remote_dispatch.py` |
| SSO reverse proxy / `https://<your-sso-host>/` | your own gateway host, or drop the proxy and use `127.0.0.1:8890` directly | deployment / reverse proxy config |

`nodes.tsv` is tab-separated with columns
`node<TAB>cluster<TAB>role<TAB>host<TAB>ip<TAB>tag`. Rows whose `cluster` starts
with `REMOTE_CLUSTER_PREFIX` are treated as SSH-reachable remote GPU nodes; all
others are ignored by the dispatcher. If you have no SSH GPU fleet, just use
`execution_target: "local"` and ignore this file entirely.
