# Annotation console (frontend + backend)

## Why the UI sometimes “won't open”

Three failure modes look similar in the browser:

1. **Process died** — an agent/`pkill` stopped `:8890`/`:7864`, or a crash left a stale pidfile. After a KML instance restart/snapshot, run `ensure_console.sh` again (or keep `--watch`).
2. **KML AccessProxy SSO expired** — the supported entry is the platform HTTPS host
   (`https://…example.com/`), which terminates TLS and proxies to
   instance `:8890`. When `accessproxy_session` expires, `/api/*` returns `302` →
   `example.com` (HTML). Refresh the page to re-login; this is not a backend crash.
3. **Slow catalog under load** — `/api/samples` used to take 15–25s on KFS and could trip
   gateway timeouts while jobs run. The backend now caches the catalog for a few seconds.

## One command (preferred)

```bash
cd benchmarks/MemStrata
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
| `https://…example.com/` | **Primary** user entry (KML AccessProxy → instance `:8890`) |
| `http://127.0.0.1:8890` | Local health/debug on the development machine itself |
| `http://<instance-ip>:8890` | Optional LAN debug; browser must bypass corporate squid |

The frontend binds to development-machine port **8890**. Backend traffic stays
loopback-only on `7864`; the frontend proxies browser `/api/*` to it. Cursor/SSH
port forwarding is not part of this path and is not required.

## Fleet naming and status

The console reads `MEMSTRATA_NODES_TSV` when available (the standard
development checkout derives it from the sibling `ssh_tunnel/nodes.tsv`).
Every endpoint is displayed as:

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

Logs: `benchmarks/MemStrata/data/_services/annotation_console/logs/`

## VLM `--allowed-local-media-path`

vLLM accepts **one directory**. Do **not** pass comma lists like `/data,/tmp`
(treated as a single nonexistent path → every `file://` S3 review returns HTTP 400).
Launchers default to `/data` (shared KFS root for clip caches).
