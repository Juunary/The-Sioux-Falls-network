# Sioux Falls Network Simulator & DRT Reinforcement Learning Testbed

**Branch:** model-update
**Last updated:** 2026-03-13

---

## 1. Project Overview

This repository is a multi-environment reinforcement learning testbed built around two transportation problems that share a common PPO training infrastructure.

| Domain | Environment | Network | Action Space | Algorithm |
|--------|-------------|---------|--------------|-----------|
| **Sioux Falls Bus Routing** | `SiouxFallsEnv` | 24 nodes, 76 directed edges | `Discrete(24)` node selection | MaskablePPO |
| **Dynamic Ride-Sharing (DRT)** | `DRTEnv` | 24-node OD matrix (1-based) | `Discrete(9)` request/reject | MaskablePPO |

Both environments expose the Gymnasium interface (`reset`, `step`, `action_masks`) and are trained through the same FastAPI + SB3 pipeline. Experiment results from both domains are stored in a shared SQLite database and browsable in the same UI, with domain isolation enforced at the schema and API layers.

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│  Browser (React + Vite + TypeScript)                    │
│  Network2D / Network3D  ·  Controls  ·  InfoPanel       │
│  ExperimentsPage  ·  ComparisonPage  ·  TrainingPage    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼────────────────────────────────────┐
│  FastAPI  (backend/main.py, port 8000)                  │
│  /api/datasets   /api/experiments   /api/training       │
│  /api/models     /api/training/{id}/ws  (WS stream)     │
└──────┬─────────────────────────────┬────────────────────┘
       │ subprocess                  │
┌──────▼──────────┐        ┌─────────▼──────────────────┐
│ training_worker │        │  job_manager (asyncio)     │
│  ─ SF branch    │        │  file-watcher → WS push    │
│  ─ DRT branch   │        └────────────────────────────┘
└──────┬──────────┘
       │
┌──────▼──────────────────────────────────────────────────┐
│  SB3 MaskablePPO + SubprocVecEnv + VecNormalize         │
│  ┌─────────────────┐   ┌────────────────────────────┐  │
│  │ SiouxFallsEnv   │   │ DRTEnv                     │  │
│  │ (semi-MDP, 96d) │   │ (semi-MDP, 157d)           │  │
│  └─────────────────┘   └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Semi-MDP Formulation (shared pattern)

Both environments use the same event-driven semi-MDP design:

```
one env.step(action) =
    1. apply action for the current pending agent
    2. advance simulation time until the next idle agent (or episode end)
    3. accumulate all event rewards during time advance
    4. return (obs, accumulated_reward, done, truncated, info)
```

PPO's GAE handles temporal credit assignment; no replay buffer or delayed-reward mechanism is needed.

---

## 3. Environments

### 3.1 Sioux Falls Bus Routing (`SiouxFallsEnv`)

- **Network:** 24 nodes (0-indexed), 76 directed edges, locked at topology v1
- **Observation:** 96-dimensional flat `Box[0, 1]`
- **Action:** `Discrete(24)` — select next node
- **Tick order:** `processPassengers()` → `stepBus()` → `replaySpawn()`
- **SIM\_DT:** 0.016 s (must match TypeScript frontend)
- **Charging stations:** nodes 2, 4, 7, 10, 14, 16, 19, 23
- **ONNX export:** supported via `backend/trainer/export.py`

#### Observation Space (96-dim)

| Dims | Feature |
|------|---------|
| 0 : 24 | Self node one-hot |
| 24 | Battery / 100 |
| 25 | Passengers / capacity |
| 26 : 30 | Status one-hot |
| 30 | Speed (normalized) |
| 31 : 55 | Onboard destination distribution / capacity |
| 55 : 79 | Waiting passengers per node / 5 (clipped) |
| 79 : 87 | Fleet at chargers / bus count |
| 87 : 95 | Charger distances / MAX\_DIST |
| 95 | Time normalized |

---

### 3.2 Dynamic Ride-Sharing (`DRTEnv`)

Ported from the KW\_DRT DDQN/TensorFlow prototype and converted to a Gymnasium-compatible PPO environment.

- **Network:** 24-node OD travel-time matrix (`od_matrix.csv`), 1-based node IDs
- **Vehicles:** `MAX_NUM_VEHICLES = 2`, capacity `VEH_CAPACITY = 5`
- **Request slots:** `MAX_NUM_REQUEST = 8` concurrent active slots
- **Episode time:** `MAX_EPISODE_TIME = 200` ticks
- **Observation:** 157-dimensional flat `Box[0, 1]`
- **Action:** `Discrete(9)` — 8 request slots + REJECT

#### Observation Space (157-dim)

| Dims | Feature |
|------|---------|
| 0 : 24 | Self current node one-hot (1-based → index `node_id − 1`) |
| 24 : 48 | Fleet node distribution (other vehicles, normalized) |
| 48 | Self capacity ratio (`remaining / VEH_CAPACITY`) |
| 49 : 52 | Self status one-hot (IDLE / PICKUP / DROPOFF) |
| 52 | Time normalized (`curr_time / MAX_EPISODE_TIME`) |
| 53 : 157 | 8 request slots × 13 dims each (see below) |

**Per-request slot (13 dims, base = 53 + i × 13):**

| Offset | Feature |
|--------|---------|
| 0 : 3 | Status one-hot (PENDING / ACCEPTED / PICKEDUP; all-zero if padding) |
| 3 | `is_valid` (1 if real request, 0 if padding) |
| 4 | `from_node / NUM_NODES` |
| 5 | `to_node / NUM_NODES` |
| 6 | `waiting_time / MAX_WAIT_TIME` |
| 7 | `arrival_due_left / (max_duration + ARRIVAL_TOL)` |
| 8 | `need_dropoff_by_me` (1 if this vehicle picked up this request) |
| 9 | `dur_to_target / max_duration` |
| 10 | `travel_time / max_duration` |
| 11 | `num_passengers / VEH_CAPACITY` |
| 12 | Reserved (0) |

#### Action Index Mapping

| Action | Meaning | Mask condition |
|--------|---------|----------------|
| 0 – 7 | Request slot i → PICKUP or DROPOFF | PENDING + spare capacity, or PICKEDUP by this vehicle |
| **8** | **REJECT** | Always `True` |

#### Reward Signal

| Event | Reward |
|-------|--------|
| PICKUP complete | `+0.5 × (1 − wait_time / MAX_WAIT_TIME)` |
| DROPOFF complete | `+1.0 × (1 − in_vehicle_time / MAX_INVEHICLE_TIME)` |
| CANCEL (wait exceeded) | `−1.0` |
| REJECT | `−0.1` |
| Time penalty | `−0.005 × elapsed_ticks` |

#### Design Decisions

- **Overflow requests:** when all 8 slots are occupied, newly arriving requests remain in `future_request_list`. Overflow requests remain in `future_request_list` until a slot opens; their original `request_time` is preserved, so promotion delay still affects eventual waiting/cancellation behavior. Requests are promoted in `request_time` order as slots open.
- **Immediate completion:** same-node PICKUP/DROPOFF is resolved in the same tick; vehicle state is explicitly reset to IDLE to prevent state corruption on the next tick.
- **`VehicleStatus.REJECT`:** preserved for semantic parity with the KW\_DRT original. Long-term simplification candidate (set IDLE immediately, remove enum value).

---

## 4. Training Pipeline

### 4.1 Starting a Training Job

#### Sioux Falls

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "sf",
    "split": "train",
    "total_timesteps": 200000,
    "n_envs": 4
  }'
```

#### DRT

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "drt",
    "drt_requests_path": "KW_DRT/data/requests_8.csv",
    "drt_vehicle_positions_path": "KW_DRT/data/vehicle_positions.csv",
    "drt_od_matrix_path": "KW_DRT/data/od_matrix.csv",
    "total_timesteps": 200000,
    "n_envs": 2
  }'
```

### 4.2 Training Config Reference

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `env_type` | `"sf"` \| `"drt"` | `"sf"` | Selects environment branch |
| `split` | string | `"train"` | SF only |
| `drt_requests_path` | string | — | DRT only; must be under `KW_DRT/data/` |
| `drt_vehicle_positions_path` | string | — | DRT only |
| `drt_od_matrix_path` | string | — | DRT only |
| `total_timesteps` | int | 100 000 | Max 5 000 000 |
| `n_envs` | int | 2 | Parallel SubprocVecEnv workers |
| `learning_rate` | float | 3e-4 | |
| `gamma` | float | 0.95 (SF) / 0.99 (DRT) | |
| `n_steps` | int | 2048 | Rollout steps per env |
| `batch_size` | int | 256 | |
| `ent_coef` | float | 0.01 | |
| `clip_range` | float | 0.2 | |
| `checkpoint_freq` | int | 50 000 | Steps between checkpoints |
| `seed` | int | 0 | |

### 4.3 Output Locations

| Path | Contents |
|------|----------|
| `data/scenarios/` | Generated SF scenario JSON files (train / val / test splits) |
| `data/experiments.db` | SQLite — SF and DRT experiment runs and per-episode metrics |
| `data/training/{job_id}/` | PPO checkpoints, `metrics.jsonl`, `vecnormalize.pkl`, `ppo_final.zip` |
| `data/models/` | Exported ONNX policy files (SF only) |
| `data/metrics/` | Batch evaluation JSONL (SF baseline evaluations) |
| `data/csv/` | Per-episode SF bus/request trace CSVs |
| `data/drt_csv/{exp_id}/{ep_id}/` | Per-episode DRT request/vehicle trace CSVs |

### 4.4 WebSocket Metrics Stream

Connect to `ws://localhost:8000/api/training/{job_id}/ws` to receive real-time JSONL records as the subprocess writes them. A `{"type": "ping"}` keepalive is sent every 30 seconds.

---

## 5. Baseline Evaluation

### 5.1 Sioux Falls — `demand_aware_greedy_v1`

```bash
curl -X POST http://localhost:8000/api/experiments/evaluate \
  -H "Content-Type: application/json" \
  -d '{"policy": "demand_aware_greedy_v1", "split": "test"}'
```

Runs in a background thread. Results stored in `data/experiments.db` under `episode_metrics`.

**SF Metrics:**

| Metric | Unit | Description |
|--------|------|-------------|
| `episode_reward` | — | Cumulative reward |
| `service_rate` | 0–1 | Fraction of spawned passengers served |
| `unserved_rate` | 0–1 | Fraction who gave up (patience expired) |
| `avg_wait_time_sec` | s | Mean boarding wait time (served only) |
| `avg_in_vehicle_sec` | s | Mean ride duration (boarding → alighting) |
| `avg_detour_px` | px | Mean excess distance vs. Dijkstra shortest path |
| `total_distance_px` | px | Total fleet distance (all buses combined) |
| `charge_events` | count | Battery-depletion events (battery hits 0; not charging-station visits) |

### 5.2 DRT — `drt_greedy_v1`

```bash
curl -X POST http://localhost:8000/api/experiments/evaluate/drt \
  -H "Content-Type: application/json" \
  -d '{
    "requests_path": "KW_DRT/data/requests_8.csv",
    "vehicle_positions_path": "KW_DRT/data/vehicle_positions.csv",
    "od_matrix_path": "KW_DRT/data/od_matrix.csv",
    "episode_id": "greedy_ep0",
    "output_csv": true
  }'
```

Paths must be under `KW_DRT/data/`; requests outside this prefix are rejected with HTTP 400.
Results stored in `data/experiments.db` under `drt_episode_metrics` (separate table — units are ticks, not seconds/pixels).

**DRT Metrics:**

| Metric | Unit | Description |
|--------|------|-------------|
| `episode_reward` | — | Cumulative reward |
| `serve_rate` | 0–1 | Fraction of requests successfully dropped off |
| `cancel_rate` | 0–1 | Fraction cancelled due to wait timeout |
| `mean_wait_ticks` | ticks | Mean time from request arrival to pickup (served only) |
| `mean_ivt_ticks` | ticks | Mean in-vehicle time from pickup to dropoff (served only) |
| `mean_detour_ticks` | ticks | Mean excess travel: `in_vehicle_time − travel_time` (served only, travel_time > 0) |

**Greedy policy rules (obs-only, `drt_greedy_v1`):**

1. Dropoff priority — nearest dropoff slot (`need_dropoff_by_me == 1`, smallest `dur_to_target`)
2. Nearest pickup — smallest `dur_to_target` among PENDING/ACCEPTED slots
3. REJECT (action 8) — fallback when no valid non-reject slot exists

**CSV output** (when `output_csv: true`):

| File | Contents |
|------|----------|
| `data/drt_csv/{exp_id}/{ep_id}/requests.csv` | Per-request trace: status, waiting_time, in_vehicle_time, travel_time, detour_ticks, pickup_at, dropoff_at |
| `data/drt_csv/{exp_id}/{ep_id}/vehicles.csv` | Per-vehicle final state: final_node, num_accept, num_serve, idle_time |
| `data/drt_csv/{exp_id}/episodes.csv` | Cumulative summary — one row appended per evaluation |

### 5.3 Cross-Domain Comparison

The Experiments page lists all runs with a **Domain** column (`sf` / `drt`). The Comparison page allows selecting any two experiments side by side; a warning banner is shown if the two experiments have different domains, since their metrics are not directly comparable.

PPO-trained DRT models can be evaluated using the same evaluator interface:

```python
from backend.datasets.drt_evaluator import run_drt_episode, make_ppo_policy_fn
from backend.env.drt_env import DRTEnv

env = DRTEnv(requests_path, vehicle_positions_path, od_matrix_path)
policy = make_ppo_policy_fn(model)   # model = loaded SB3 MaskablePPO
metrics = run_drt_episode(env, policy, "maskable_ppo", episode_id="ppo_ep0")
```

---

## 6. ONNX Export and Browser Inference (Sioux Falls)

```bash
curl -X POST http://localhost:8000/api/models/{job_id}/export_onnx
```

The exported file is saved to `data/models/{job_id}.onnx`. In the browser, select the model and click **Load Model** — inference runs client-side via `onnxruntime-web` (WASM, `numThreads=1`). The model is cached in memory; reload the page to clear it.

DRT ONNX export works via the same `export.py` pipeline (SB3 MlpPolicy) but browser-side DRT inference is not yet wired to the UI.

---

## 7. Quick Start

### Frontend

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

### Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Tests

```bash
pytest                     # all backend tests
npx tsc --noEmit           # TypeScript type check
```

---

## 8. Project Structure

```
src/
  types/network.ts          Core domain types
  data/network.ts           Graph topology — 24 nodes, 76 directed edges (v1 locked)
  utils/
    graph.ts                Adjacency helpers, edge interpolation, lane offsets
    pathfinding.ts          Dijkstra + path reconstruction + nearest-CS finder
    rng.ts                  mulberry32 PRNG, Fisher-Yates sample, simRng singleton
    onnxInference.ts        ONNX session management, 96-dim obs builder, inference cache
  sim/
    store.ts                Zustand store — shared state for 2D and 3D views
    engine.ts               simulateTick(), runEpisode(), useSimulationEngine() RAF hook
    routing.ts              decideNextRoute() with route invariant; ppoDecision()
  components/
    Network2D.tsx           D3-powered SVG canvas (pan/zoom, interactions)
    Network3D.tsx           @react-three/fiber scene (OrbitControls)
    Controls.tsx            Left sidebar — playback, settings, PPO model management
    InfoPanel.tsx           Right sidebar — bus/passenger/node selection details
  pages/
    ExperimentsPage.tsx     Run SF and DRT evaluations; Domain column in experiment list
    ComparisonPage.tsx      Side-by-side aggregate comparison; domain mismatch warning
    TrainingPage.tsx        Configure, start, and monitor PPO training jobs
  api/
    experiments.ts          SF + DRT evaluation API calls; AggregateStats (SF+DRT fields)
  App.tsx                   Router root (NavBar + page routing)
  pages/SimulatorPage.tsx   Mounts simulation engine; handles 2D/3D view toggle

backend/
  main.py                   FastAPI app entry point
  api/
    schemas.py              Pydantic types mirroring TS types
    routes/
      training.py           POST /start (env_type: "sf"|"drt"), stop, status, WS stream
      experiments.py        SF + DRT evaluation; POST /evaluate/drt; domain-aware aggregate
      datasets.py           SF scenario generation
      models.py             ONNX export and model listing
  env/
    sioux_falls_env.py      SF Gymnasium environment (semi-MDP, MaskablePPO)
    bus.py                  BusState + stepBus + decideNextRoute
    passenger.py            Passenger + processPassengers + replaySpawn
    pathfinding.py          Dijkstra with tie-break; REACHABILITY matrix
    observation.py          96-dim SF obs builder (Box[0,1])
    network.py              Python mirror of SF graph topology
    drt_env.py              DRT Gymnasium environment (semi-MDP, MaskablePPO)
    drt_config.py           DRT constants (NUM_NODES=24, OBS_DIM_DRT=157, POSSIBLE_ACTION=9, …)
    drt_network.py          DRTNetwork — OD matrix loader and duration lookup
    drt_vehicle.py          Vehicle + VehicleStatus domain objects
    drt_request.py          Request + RequestStatus domain objects
    drt_observation.py      157-dim DRT obs builder (Box[0,1])
  baseline/
    demand_aware.py         demand_aware_greedy_v1 (SF deterministic baseline)
    drt_greedy.py           drt_greedy_v1 — obs-only greedy (dropoff > nearest pickup > reject)
  datasets/
    generator.py            Deterministic SF scenario generator (numpy seed)
    evaluator.py            SF episode runner + batch evaluation
    csv_logger.py           Per-episode SF bus/request trace CSV logging
    drt_loader.py           DRT CSV loader (requests, vehicle positions, OD matrix)
    drt_evaluator.py        DRT episode runner; run_drt_episode(); make_ppo_policy_fn()
                            writes requests.csv / vehicles.csv / episodes.csv per episode
  experiments/
    tracker.py              SQLite-backed experiment tracker
                            experiments(…, domain) + episode_metrics + drt_episode_metrics
  trainer/
    ppo_trainer.py          build_model() (SF) + build_drt_model() (DRT)
                            RotatingScenarioEnv + RotatingDRTEpisodeEnv wrappers
    training_worker.py      Subprocess entry point — env_type branch (sf / drt)
    callbacks.py            CheckpointCallback + MetricsCallback (JSONL)
    export.py               MaskablePPO → ONNX export
  worker/
    job_manager.py          Subprocess spawn + asyncio file watcher + WS broadcast
  tests/
    test_parity.py          Generator determinism + spawn boundary + episode determinism
    test_pathfinding.py     Dijkstra + route invariant tests

KW_DRT/                     Original KW_DRT prototype (reference only; not imported by backend)
  data/
    od_matrix.csv           24-node OD travel-time matrix (source for DRTNetwork)
    requests_8.csv          8-request test scenario
    requests_80.csv         80-request full scenario
    vehicle_positions.csv   Initial vehicle positions (9 rows; loader truncates to MAX_NUM_VEHICLES=2)
  app/                      Original DDQN/TensorFlow implementation (reference only)
```

---

## 9. Database Schema

```sql
-- Shared experiment registry
experiments(
  experiment_id, policy, split, created_at, git_commit,
  graph_version, scenario_count, config_json,
  domain TEXT DEFAULT 'sf'   -- 'sf' | 'drt'
)

-- Sioux Falls episode metrics
episode_metrics(
  experiment_id, scenario_id, policy, episode_reward,
  passengers_spawned, passengers_served, passengers_gone,
  service_rate, unserved_rate, avg_wait_time_sec,
  charge_events, total_distance_px, avg_in_vehicle_sec,
  avg_detour_px, wall_time_sec
)

-- DRT episode metrics (separate table; ticks, not seconds/pixels)
drt_episode_metrics(
  experiment_id, episode_id, policy, episode_reward,
  total_requests, served_count, cancelled_count,
  serve_rate, cancel_rate,
  mean_wait_ticks, mean_ivt_ticks, mean_detour_ticks,
  total_steps, total_ticks, wall_time_sec
)
```

`GET /api/experiments/{id}/aggregate` dispatches to the appropriate aggregator based on `domain`.
SF and DRT experiments are stored in separate tables to prevent unit confusion (ticks vs. seconds/pixels).

---

## 10. Key Implementation Constraints

### 10.1 Sioux Falls

- **Topology lock:** `src/data/network.ts` is the canonical v1 graph. Do not modify node/edge definitions.
- **Tick order:** `processPassengers()` → `stepBus()` → `replaySpawn()` (must match Python env exactly).
- **Route state invariant:** if `currentEdge != null`, route does not contain the edge's target node; if `currentEdge == null`, `route[0]` is the next hop.
- **SIM\_DT = 0.016 s** — fixed timestep; both frontend and backend must agree.

### 10.2 DRT

- **Node index convention:** all node IDs are 1-based in CSV and domain objects; `drt_observation.py` converts to 0-based indices via `node_id − 1`.
- **Vehicle count:** `drt_loader.py` truncates `vehicle_positions.csv` to `MAX_NUM_VEHICLES = 2`, mirroring KW\_DRT's `range(cfg.MAX_NUM_VEHICLES)`.
- **Slot cap:** `_load_arriving_requests()` caps `active_request_list` at `MAX_NUM_REQUEST = 8`; overflow requests stay in `future_request_list` and are promoted in arrival-time order.
- **Path validation:** `/api/experiments/evaluate/drt` rejects any file path not starting with `KW_DRT/data/` (HTTP 400).
- **`KW_DRT/app/`** files are retained as reference only. They are not imported by the backend.

---

## 11. Known Limitations

- **`avg_detour_px`** (SF): can be slightly negative due to odometer sampling before `stepBus()` runs. Small negatives are expected.
- **`charge_events`** (SF): counts battery-depletion events (battery hits 0), not charging station visits.
- **PPO browser inference** (SF): model cached after first load; reload to clear. WASM served via CDN (`onnxruntime-web 1.19.0`).
- **DRT browser inference:** ONNX export works via `export.py` but browser-side inference is not yet wired to the UI (Phase 5).
- **DRT batch evaluation:** `evaluate_drt_baseline()` runs a single episode. Multi-episode batch evaluation over `requests_80.csv` is not yet implemented.
- **Observation space** (SF): a planned 96→128 dim expansion (urgency/slack features) is deferred pending PPO retraining.
