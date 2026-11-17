"""Flask web server that connects the MANET dashboard to the Python simulator."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from manet_simulation import simulate, write_outputs

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "manet_report"
simulation_lock = threading.Lock()
live_lock = threading.Lock()
live_subscribers: set[queue.Queue[dict[str, Any]]] = set()


@dataclass
class LiveRun:
    run_id: str
    settings: dict[str, int | float]
    condition: threading.Condition = field(default_factory=threading.Condition)
    status: str = "running"
    paused: bool = False
    cancelled: bool = False
    final_payload: dict[str, Any] | None = None


current_live_run: LiveRun | None = None

app = Flask(__name__)
LIVE_SERVER_ORIGINS = {
    "http://127.0.0.1:5500",
    "http://localhost:5500",
}


@app.after_request
def allow_live_server_dashboard(response):
    origin = request.headers.get("Origin")
    if request.path.startswith("/api/") and origin in LIVE_SERVER_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.vary.add("Origin")
    return response


def publish_live(event: dict[str, Any]) -> None:
    with live_lock:
        subscribers = tuple(live_subscribers)
    for subscriber in subscribers:
        subscriber.put(event)


def parse_simulation_settings(data: Any) -> tuple[dict[str, int | float] | None, str | None]:
    if not isinstance(data, dict):
        return None, "Expected a JSON object with simulation settings."

    nodes = data.get("nodes", 20)
    rounds = data.get("rounds", 30)
    packets = data.get("packets", 20)
    radio_range = data.get("radio_range", 28)
    attackers = data.get("attackers", 2)
    seed = data.get("seed", 7)
    integer_settings = {
        "nodes": (nodes, 3, 100),
        "rounds": (rounds, 1, 100),
        "packets": (packets, 1, 50),
        "attackers": (attackers, 0, 100),
    }
    for name, (value, minimum, maximum) in integer_settings.items():
        if type(value) is not int or not minimum <= value <= maximum:
            return None, f"{name} must be an integer from {minimum} to {maximum}."
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        return None, "seed must be an integer from 0 to 4294967295."
    if attackers >= nodes:
        return None, "Attackers must be fewer than the number of nodes."
    if (
        isinstance(radio_range, bool)
        or not isinstance(radio_range, (int, float))
        or not 1 <= radio_range <= 150
    ):
        return None, "radio_range must be a number from 1 to 150."

    return {
        "nodes": nodes,
        "rounds": rounds,
        "packets": packets,
        "radio_range": float(radio_range),
        "attackers": attackers,
        "seed": seed,
    }, None


def report_payload(
    results: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    return {
        "results": results,
        "graph_url": "/reports/routing_comparison.png",
        "csv_url": "/reports/routing_results.csv",
        "database_url": "/reports/manet_events.sqlite3",
    }


def run_live_worker(run: LiveRun) -> None:
    def should_continue() -> bool:
        with run.condition:
            while run.paused and not run.cancelled:
                run.condition.wait()
            return not run.cancelled

    def send_progress(progress: dict[str, int | str]) -> None:
        publish_live({"type": "progress", "run_id": run.run_id, **progress})
        with run.condition:
            run.condition.wait_for(
                lambda: not run.paused or run.cancelled,
                timeout=0.08,
            )

    try:
        with simulation_lock:
            results, events = simulate(
                node_count=int(run.settings["nodes"]),
                rounds=int(run.settings["rounds"]),
                packets_per_round=int(run.settings["packets"]),
                radio_range=float(run.settings["radio_range"]),
                attacker_count=int(run.settings["attackers"]),
                seed=int(run.settings["seed"]),
                on_progress=send_progress,
                should_continue=should_continue,
            )
            if run.cancelled:
                return
            write_outputs(REPORT_DIR, results, events)
    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Could not run or save the live MANET simulation.")
        with live_lock:
            if current_live_run is run:
                run.status = "error"
        publish_live({"type": "error", "run_id": run.run_id, "message": str(error)})
        return

    payload = report_payload(results)
    with live_lock:
        if current_live_run is run:
            run.status = "completed"
            run.final_payload = payload
    publish_live({"type": "complete", "run_id": run.run_id, **payload})


@app.get("/")
def dashboard():
    return send_from_directory(ROOT, "manet.html")


@app.get("/style.css")
def stylesheet():
    return send_from_directory(ROOT, "style.css")


@app.get("/reports/<path:filename>")
def report_file(filename: str):
    return send_from_directory(REPORT_DIR, filename)


@app.route("/api/simulate", methods=["POST", "OPTIONS"])
def run_simulation():
    if request.method == "OPTIONS":
        return "", 204

    settings, error = parse_simulation_settings(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400
    assert settings is not None

    try:
        with simulation_lock:
            results, events = simulate(
                node_count=int(settings["nodes"]),
                rounds=int(settings["rounds"]),
                packets_per_round=int(settings["packets"]),
                radio_range=float(settings["radio_range"]),
                attacker_count=int(settings["attackers"]),
                seed=int(settings["seed"]),
            )
            write_outputs(REPORT_DIR, results, events)
    except (OSError, RuntimeError) as error:
        app.logger.exception("Could not run or save the MANET simulation.")
        return jsonify(error=f"Simulation failed: {error}"), 500

    return jsonify(report_payload(results))


@app.get("/api/live/events")
def live_events():
    subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
    with live_lock:
        live_subscribers.add(subscriber)
        run = current_live_run
        state: dict[str, Any] = {
            "type": "state",
            "status": run.status if run else "idle",
        }
        if run:
            state["run_id"] = run.run_id
            if run.final_payload:
                state["payload"] = run.final_payload
        subscriber.put(state)

    def stream():
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                else:
                    yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        finally:
            with live_lock:
                live_subscribers.discard(subscriber)

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/live/start")
def start_live_simulation():
    global current_live_run
    settings, error = parse_simulation_settings(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400
    assert settings is not None

    with live_lock:
        previous = current_live_run
        if previous and previous.status in {"running", "paused"}:
            with previous.condition:
                previous.cancelled = True
                previous.paused = False
                previous.status = "stopped"
                previous.condition.notify_all()
        run = LiveRun(run_id=uuid.uuid4().hex, settings=settings)
        current_live_run = run

    if previous and previous.status == "stopped":
        publish_live({"type": "stopped", "run_id": previous.run_id})
    publish_live(
        {
            "type": "started",
            "run_id": run.run_id,
            "rounds": int(settings["rounds"]),
        }
    )
    threading.Thread(target=run_live_worker, args=(run,), daemon=True).start()
    return jsonify(run_id=run.run_id, status=run.status), 202


@app.post("/api/live/control")
def control_live_simulation():
    global current_live_run
    data: Any = request.get_json(silent=True)
    action = data.get("action") if isinstance(data, dict) else None
    if action not in {"pause", "resume", "reset"}:
        return jsonify(error="action must be pause, resume, or reset."), 400

    with live_lock:
        run = current_live_run
        if action == "reset":
            current_live_run = None
            if run:
                with run.condition:
                    run.cancelled = True
                    run.paused = False
                    run.status = "stopped"
                    run.condition.notify_all()
        elif run is None or run.status not in {"running", "paused"}:
            return jsonify(error="No active live simulation."), 409
        elif action == "pause" and run.status != "running":
            return jsonify(error="The live simulation is already paused."), 409
        elif action == "resume" and run.status != "paused":
            return jsonify(error="The live simulation is not paused."), 409
        else:
            with run.condition:
                run.paused = action == "pause"
                run.status = "paused" if run.paused else "running"
                run.condition.notify_all()

    if action == "reset":
        publish_live({"type": "reset"})
        return jsonify(status="idle")

    publish_live({"type": action + "d", "run_id": run.run_id})
    return jsonify(status=run.status, run_id=run.run_id)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
