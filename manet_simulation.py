"""Offline MANET routing simulation for college reports.

Install dependencies with ``python -m pip install -r requirements.txt``. Run
``python app.py`` for the dashboard integration, or
``python manet_simulation.py`` to run the simulator on its own. This simulation
does not scan or connect to real devices.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import statistics
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

Point = tuple[float, float]
RouteFn = Callable[[nx.Graph, int, int], list[int] | None]
ProgressFn = Callable[[dict[str, int | str]], None]
ContinueFn = Callable[[], bool]


@dataclass
class Node:
    x: float
    y: float
    destination: Point
    speed: float


def distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class RandomWaypoint:
    """Simple Random Waypoint mobility over a bounded 2D area."""

    def __init__(
        self, count: int, width: float, height: float, rng: random.Random
    ) -> None:
        self.width = width
        self.height = height
        self.rng = rng
        self.nodes = [
            Node(
                x=rng.uniform(0, width),
                y=rng.uniform(0, height),
                destination=(rng.uniform(0, width), rng.uniform(0, height)),
                speed=rng.uniform(1.0, 5.0),
            )
            for _ in range(count)
        ]

    def step(self) -> None:
        for node in self.nodes:
            current = (node.x, node.y)
            remaining = distance(current, node.destination)
            if remaining <= node.speed:
                node.x, node.y = node.destination
                node.destination = (
                    self.rng.uniform(0, self.width),
                    self.rng.uniform(0, self.height),
                )
                continue
            dx = (node.destination[0] - node.x) / remaining * node.speed
            dy = (node.destination[1] - node.y) / remaining * node.speed
            node.x += dx
            node.y += dy

    def graph(self, radio_range: float) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.nodes)))
        for left in range(len(self.nodes)):
            for right in range(left + 1, len(self.nodes)):
                a = (self.nodes[left].x, self.nodes[left].y)
                b = (self.nodes[right].x, self.nodes[right].y)
                if distance(a, b) <= radio_range:
                    graph.add_edge(left, right)
        return graph


def shortest_route(graph: nx.Graph, source: int, target: int) -> list[int] | None:
    try:
        return nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return None


class DSRRouter:
    """DSR-style route cache with route rediscovery after a broken path."""

    def __init__(self) -> None:
        self.cache: dict[tuple[int, int], list[int]] = {}

    def route(self, graph: nx.Graph, source: int, target: int) -> list[int] | None:
        key = (source, target)
        cached = self.cache.get(key)
        if cached and all(graph.has_edge(a, b) for a, b in zip(cached, cached[1:])):
            return cached
        path = shortest_route(graph, source, target)
        if path:
            self.cache[key] = path
        else:
            self.cache.pop(key, None)
        return path


class OLSRRouter:
    """OLSR-style proactive shortest-path table, refreshed each mobility step."""

    def __init__(self) -> None:
        self.routes: dict[tuple[int, int], list[int]] = {}

    def refresh(self, graph: nx.Graph) -> None:
        self.routes.clear()
        for source, routes in nx.all_pairs_shortest_path(graph):
            for target, path in routes.items():
                self.routes[(source, target)] = path

    def route(self, graph: nx.Graph, source: int, target: int) -> list[int] | None:
        del graph
        return self.routes.get((source, target))


def aodv_route(graph: nx.Graph, source: int, target: int) -> list[int] | None:
    return shortest_route(graph, source, target)


ROUTING_ALGORITHMS = ("AODV", "DSR", "OLSR")


def validate_simulation_settings(
    *,
    node_count: int,
    rounds: int,
    packets_per_round: int,
    radio_range: float,
    attacker_count: int,
) -> None:
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 3:
        raise ValueError("node_count must be an integer of at least 3.")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer.")
    if isinstance(packets_per_round, bool) or not isinstance(packets_per_round, int) or packets_per_round < 1:
        raise ValueError("packets_per_round must be a positive integer.")
    if isinstance(radio_range, bool) or not isinstance(radio_range, (int, float)) or radio_range <= 0:
        raise ValueError("radio_range must be a positive number.")
    if isinstance(attacker_count, bool) or not isinstance(attacker_count, int) or attacker_count < 0:
        raise ValueError("attacker_count must be a non-negative integer.")
    if attacker_count >= node_count:
        raise ValueError("attacker_count must be fewer than node_count.")


def simulate(
    *,
    node_count: int,
    rounds: int,
    packets_per_round: int,
    radio_range: float,
    attacker_count: int,
    seed: int,
    on_progress: ProgressFn | None = None,
    should_continue: ContinueFn | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, object]]]:
    validate_simulation_settings(
        node_count=node_count,
        rounds=rounds,
        packets_per_round=packets_per_round,
        radio_range=radio_range,
        attacker_count=attacker_count,
    )
    rng = random.Random(seed)
    mobility = RandomWaypoint(node_count, 100.0, 100.0, rng)
    candidates = list(range(1, node_count))
    attackers = set(rng.sample(candidates, min(attacker_count, len(candidates))))
    dsr = DSRRouter()
    olsr = OLSRRouter()
    routers: dict[str, RouteFn] = {
        "AODV": aodv_route,
        "DSR": dsr.route,
        "OLSR": olsr.route,
    }
    stats = {
        name: {"sent": 0, "delivered": 0, "hops": [], "delay": []}
        for name in ROUTING_ALGORITHMS
    }
    node_stats = {
        name: {node_id: {"forwarded": 0, "dropped": 0} for node_id in range(node_count)}
        for name in ROUTING_ALGORITHMS
    }
    events: list[dict[str, object]] = []
    per_hop_delay_ms = 8.0
    drop_probability = 0.75
    quarantine_threshold = 2
    quarantined: dict[str, set[int]] = {name: set() for name in ROUTING_ALGORITHMS}

    cancelled = False
    for round_number in range(rounds):
        mobility.step()
        graph = mobility.graph(radio_range)
        for algorithm, route_fn in routers.items():
            active_graph = graph.copy()
            active_graph.remove_nodes_from(quarantined[algorithm])
            if algorithm == "OLSR":
                olsr.refresh(active_graph)
            active_nodes = list(active_graph.nodes)
            round_sent = round_delivered = round_dropped = round_no_route = 0
            for _ in range(packets_per_round):
                if should_continue is not None and not should_continue():
                    cancelled = True
                    break
                stats[algorithm]["sent"] += 1
                round_sent += 1
                if len(active_nodes) < 2:
                    round_no_route += 1
                    events.append(
                        {
                            "round": round_number,
                            "algorithm": algorithm,
                            "source": -1,
                            "target": -1,
                            "status": "no_route",
                            "hops": 0,
                        }
                    )
                    continue
                source, target = rng.sample(active_nodes, 2)
                path = route_fn(active_graph, source, target)
                if not path:
                    round_no_route += 1
                    events.append(
                        {
                            "round": round_number,
                            "algorithm": algorithm,
                            "source": source,
                            "target": target,
                            "status": "no_route",
                            "hops": 0,
                        }
                    )
                    continue

                dropped_by: int | None = None
                for node_id in path[1:-1]:
                    observed = node_stats[algorithm][node_id]
                    observed["forwarded"] += 1
                    if node_id in attackers and rng.random() < drop_probability:
                        observed["dropped"] += 1
                        dropped_by = node_id
                        break

                if dropped_by is not None:
                    round_dropped += 1
                    if node_stats[algorithm][dropped_by]["dropped"] >= quarantine_threshold:
                        quarantined[algorithm].add(dropped_by)
                    status = "simulated_drop"
                else:
                    round_delivered += 1
                    stats[algorithm]["delivered"] += 1
                    stats[algorithm]["hops"].append(len(path) - 1)
                    stats[algorithm]["delay"].append((len(path) - 1) * per_hop_delay_ms)
                    status = "delivered"

                events.append(
                    {
                        "round": round_number,
                        "algorithm": algorithm,
                        "source": source,
                        "target": target,
                        "status": status,
                        "hops": len(path) - 1,
                    }
                )

            if on_progress is not None and round_sent:
                on_progress(
                    {
                        "round": round_number + 1,
                        "rounds": rounds,
                        "algorithm": algorithm,
                        "sent": round_sent,
                        "delivered": round_delivered,
                        "dropped": round_dropped,
                        "no_route": round_no_route,
                    }
                )
            if cancelled:
                break
        if cancelled:
            break

    results: list[dict[str, float | int | str]] = []
    for algorithm, values in stats.items():
        sent = int(values["sent"])
        delivered = int(values["delivered"])
        hops = values["hops"]
        delays = values["delay"]
        results.append(
            {
                "algorithm": algorithm,
                "sent": sent,
                "delivered": delivered,
                "pdr_percent": delivered * 100.0 / sent if sent else 0.0,
                "average_delay_ms": statistics.fmean(delays) if delays else 0.0,
                "average_hops": statistics.fmean(hops) if hops else 0.0,
                "detected_nodes": len(quarantined[algorithm]),
            }
        )
    return results, events


def write_outputs(
    output_dir: Path,
    results: list[dict[str, float | int | str]],
    events: list[dict[str, object]],
) -> None:
    if not results:
        raise ValueError("results must contain at least one algorithm row.")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "routing_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    database_path = output_dir / "manet_events.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    round INTEGER,
                    algorithm TEXT,
                    source INTEGER,
                    target INTEGER,
                    status TEXT,
                    hops INTEGER
                )
                """
            )
            connection.execute("DELETE FROM events")
            connection.executemany(
                "INSERT INTO events VALUES (:round, :algorithm, :source, :target, :status, :hops)",
                events,
            )

    labels = [str(row["algorithm"]) for row in results]
    metrics = (
        ("pdr_percent", "Packet delivery ratio (%)"),
        ("average_delay_ms", "Average delay (ms)"),
        ("average_hops", "Average hop count"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (key, title) in zip(axes, metrics):
        axis.bar(labels, [float(row[key]) for row in results], color="#2563eb")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "routing_comparison.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MANET routing algorithms.")
    parser.add_argument("--nodes", type=int, default=20, help="Number of mobile nodes.")
    parser.add_argument("--rounds", type=int, default=30, help="Mobility update rounds.")
    parser.add_argument("--packets", type=int, default=20, help="Packets per algorithm per round.")
    parser.add_argument("--range", dest="radio_range", type=float, default=28.0, help="Radio range.")
    parser.add_argument("--attackers", type=int, default=2, help="Simulated packet-dropping nodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable results.")
    parser.add_argument("--output", type=Path, default=Path("manet_report"), help="Output directory.")
    args = parser.parse_args()
    if args.nodes < 3:
        parser.error("--nodes must be at least 3.")
    if args.rounds < 1 or args.packets < 1 or args.radio_range <= 0:
        parser.error("--rounds, --packets, and --range must be positive.")
    if args.attackers < 0:
        parser.error("--attackers cannot be negative.")
    if args.attackers >= args.nodes:
        parser.error("--attackers must be fewer than --nodes.")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    results, events = simulate(
        node_count=args.nodes,
        rounds=args.rounds,
        packets_per_round=args.packets,
        radio_range=args.radio_range,
        attacker_count=args.attackers,
        seed=args.seed,
    )
    write_outputs(args.output, results, events)
    print(f"{'Algorithm':<12}{'PDR (%)':>12}{'Delay (ms)':>14}{'Hops':>10}{'Detected':>12}")
    for row in results:
        print(
            f"{row['algorithm']:<12}{row['pdr_percent']:>12.2f}"
            f"{row['average_delay_ms']:>14.2f}{row['average_hops']:>10.2f}"
            f"{row['detected_nodes']:>12}"
        )
    print(f"\nReport files saved in: {args.output.resolve()}")
    print(f"Simulation completed in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
