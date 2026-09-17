# Chaos Monkey Resilience Testing Framework

The **Chaos Monkey** framework (`chaos/`) is an automated resilience testing component designed to validate the fault tolerance and self-healing mechanisms of the **RAG Builder Scheduler Server** and its managed worker container pool.

---

## 📌 Architecture & Design Philosophy

```text
             CHAOS MONKEY
                  │
        Intentionally injects controlled failures
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
 Worker Stop   RabbitMQ    Scheduler
  & Crash       Outage      Restart
     │            │            │
     └────────────┬────────────┘
                  ▼
          Scheduler Server
     (Detects, Heals & Reconciles)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Requeue / Retry      Prevent Duplicates
  (max_retries=3)       (Idempotency)
        │                   │
        └─────────┬─────────┘
                  ▼
      Healthy Alternative Worker
                  │
                  ▼
              Completed
```

### Key Principle
> **The Scheduler performs business logic. The Chaos component only creates controlled test failures.**

Failure logic is never mixed directly into scheduler production code. Chaos Monkey acts as an external test harness operating against Docker, RabbitMQ, PostgreSQL, and the Scheduler TCP control port (`3200`).

---

## 🧪 Chaos Monkey Test Matrix

| Experiment | Failure Injected | Expected Resilience | Verification Flow |
| :--- | :--- | :--- | :--- |
| **CH-01** | Worker Stopped | Job recovered/reassigned; slot marked unhealthy | Worker container is stopped; scheduler detects offline slot within reconciliation loop and routes work elsewhere. |
| **CH-02** | Worker Crash In-Flight | Job requeued/retried (max 3 retries); alternate worker completes | Worker container killed mid-processing; scheduler increments `retry_count`, resets job to `QUEUED`, and dispatches to another healthy worker. |
| **CH-03** | Scheduler Restarted | Scheduler recovers state from PostgreSQL | Scheduler daemon restarted; automatically recovers interrupted `DISPATCHING` jobs, reconnects to RabbitMQ, and resumes scheduling without losing jobs. |
| **CH-04** | RabbitMQ Stopped | Scheduler consumer reconnects upon broker restoration | RabbitMQ container paused; scheduler logs disconnect, executes exponential backoff, and re-establishes channel cleanly when restored. |
| **CH-05** | PostgreSQL Stopped | DB connection recovers without scheduler panic | Database paused; scheduler catches connection exceptions gracefully and resumes tracking upon restoration. |
| **CH-06** | Worker Slot Exhaustion | Scheduler waits/dynamically scales up to `MAX_PARALLEL_JOBS` | When capacity is saturated, scheduler holds incoming messages safely in queue without dropping jobs. |
| **CH-07** | Slow Worker Timeout | Scheduler cancels job via `POST /stop` and marks `TIMED_OUT` | Active job exceeding `job_timeout_seconds` is stopped, slot is reset to idle, and job state marked `TIMED_OUT`. |
| **CH-08** | Duplicate Message | Idempotency guard prevents duplicate processing | Duplicate message for already `COMPLETED` job is acknowledged and skipped without re-running document pipeline. |

---

## 🚀 Running Chaos Monkey

### 1. Dry-Run / Simulation Mode
Run the entire suite safely in dry-run mode without stopping real production containers:
```bash
python -m chaos.chaos_controller --dry-run
```

### 2. Run a Specific Experiment
Target a single failure scenario against live containers:
```bash
python -m chaos.chaos_controller --test CH-01
python -m chaos.chaos_controller --test CH-02
python -m chaos.chaos_controller --test CH-08
```

### 3. Run the Entire Resilience Matrix
Execute all experiments against the test environment:
```bash
python -m chaos.chaos_controller --all
```

---

## 📊 Expected Output Report

```text
==========================================================================================
                     CHAOS MONKEY RESILIENCE VERIFICATION MATRIX
==========================================================================================
ID      | Scenario                         | Result   | Recovery   | Details
------------------------------------------------------------------------------------------
CH-01   | Worker Stopped                   | PASSED   | 12.4s      | Stopped doc_processor_1; scheduler marked...
CH-02   | Worker Crashes During Processing | PASSED   | 15.2s      | In-flight crash recovered; retry (1/3)...
CH-03   | Scheduler Restarted              | PASSED   | 18.3s      | Reconnected to RabbitMQ; state recovered...
CH-04   | RabbitMQ Stopped & Reconnected   | PASSED   | 14.1s      | Broker disconnect handled; reconnected...
CH-05   | PostgreSQL Stopped & Reconnected | PASSED   | 11.5s      | Database reconnection verified...
CH-06   | Worker Slot Exhaustion           | PASSED   | 4.2s       | Saturation buffered cleanly in queue...
CH-07   | Slow Worker Timeout              | PASSED   | 3.8s       | Timeout reaper active; POST /stop sent...
CH-08   | Duplicate Message (Idempotency)  | PASSED   | 0.4s       | Duplicate message discarded; 0 dups...
==========================================================================================
RESILIENCE METRICS SUMMARY:
  Total Experiments Run : 8
  Passed                : 8
  Failed                : 0
  Jobs Submitted        : 10
  Jobs Completed        : 10
  Jobs Lost             : 0
  Unexpected Duplicates : 0
  Jobs Stuck            : 0
==========================================================================================
```
