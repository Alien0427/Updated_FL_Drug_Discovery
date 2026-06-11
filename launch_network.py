"""Network Orchestrator & Live CLI Dashboard for the P2P Drug Discovery Swarm.

Usage:
    python launch_network.py --nodes 4 --base-port 8001

Spawns N peer nodes as background processes, bootstraps them into a unified
swarm, then continuously polls their REST APIs and renders a live ASCII table
showing each node's status, discovered peers, and CRDT ledger size.

Press Ctrl+C to gracefully terminate all child processes.
"""

import argparse
import os
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 2
VENV_PY = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
# Fall back to the current interpreter if the venv wrapper isn't found
if not os.path.exists(VENV_PY):
    VENV_PY = sys.executable

PEER_SCRIPT = "peer_node.py"
GRAPH_FILES = [
    "data/client_1_graph.graphml",
    "data/client_2_graph.graphml",
    "data/client_3_graph.graphml",
]


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def color(text: str, code: str) -> str:
    """Wrap text in an ANSI color code (no-op on Windows if colors unsupported)."""
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:  return color(t, "32")
def yellow(t: str) -> str: return color(t, "33")
def red(t: str) -> str:    return color(t, "31")
def bold(t: str) -> str:   return color(t, "1")
def cyan(t: str) -> str:   return color(t, "36")


# ---------------------------------------------------------------------------
# Node polling
# ---------------------------------------------------------------------------

def poll_node(port: int) -> dict:
    """Query a single peer's /ping, /peers, and /crdt_state endpoints.

    Returns a dict with keys:
        name, port, status, known_peers, ledger_size, node_id_prefix
    """
    base = f"http://localhost:{port}"
    result = {
        "name": f"Peer_{port}",   # placeholder until we get the real name
        "port": port,
        "status": "BOOTING",
        "known_peers": "-",
        "ledger_size": "-",
        "node_id_prefix": "-",
    }

    # --- /ping ---------------------------------------------------------------
    try:
        ping = requests.get(f"{base}/ping", timeout=REQUEST_TIMEOUT_SECONDS)
        if ping.status_code == 200:
            data = ping.json()
            result["status"] = "ALIVE"
            result["name"] = data.get("peer_id", result["name"])
            result["node_id_prefix"] = data.get("node_id_hex_prefix", "-")
        else:
            result["status"] = "ERROR"
            return result
    except requests.exceptions.ConnectionError:
        result["status"] = "BOOTING"
        return result
    except requests.exceptions.Timeout:
        result["status"] = "TIMEOUT"
        return result
    except Exception:
        result["status"] = "OFFLINE"
        return result

    # --- /peers --------------------------------------------------------------
    try:
        peers_resp = requests.get(f"{base}/peers", timeout=REQUEST_TIMEOUT_SECONDS)
        if peers_resp.status_code == 200:
            result["known_peers"] = peers_resp.json().get("total_known", 0)
    except Exception:
        result["known_peers"] = "?"

    # --- /crdt_state ---------------------------------------------------------
    try:
        crdt_resp = requests.get(f"{base}/crdt_state", timeout=REQUEST_TIMEOUT_SECONDS)
        if crdt_resp.status_code == 200:
            result["ledger_size"] = crdt_resp.json().get("ledger_size", 0)
    except Exception:
        result["ledger_size"] = "?"

    return result


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

COL_WIDTHS = {
    "name":          10,
    "port":           6,
    "status":        10,
    "known_peers":   13,
    "ledger_size":   18,
    "node_id":       18,
}

DIVIDER = (
    "+-" + "-" * COL_WIDTHS["name"] + "-+-"
    + "-" * COL_WIDTHS["port"] + "-+-"
    + "-" * COL_WIDTHS["status"] + "-+-"
    + "-" * COL_WIDTHS["known_peers"] + "-+-"
    + "-" * COL_WIDTHS["ledger_size"] + "-+-"
    + "-" * COL_WIDTHS["node_id"] + "-+"
)


def fmt(value, width: int, align: str = "<") -> str:
    """Format a value into a fixed-width column (strip ANSI for length calc)."""
    s = str(value)
    # Strip ANSI codes for width measurement
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", s)
    padding = max(0, width - len(plain))
    if align == ">":
        return " " * padding + s
    return s + " " * padding


def render_dashboard(nodes_info: list[dict], num_nodes: int, elapsed: float) -> None:
    """Clear the terminal and draw the live ASCII table."""
    clear_screen()

    # Header
    print(bold(cyan("+=================================================================+")))
    print(bold(cyan("|    P2P DRUG DISCOVERY  --  NETWORK ORCHESTRATOR DASHBOARD      |")))
    print(bold(cyan("+=================================================================+")))
    print(f"  Nodes: {bold(str(num_nodes))}   |   Uptime: {bold(f'{elapsed:.0f}s')}   |   Refresh: every {POLL_INTERVAL_SECONDS}s   |   Ctrl+C to shutdown\n")

    # Table header
    print(DIVIDER)
    header = (
        "| "
        + fmt(bold("Node"),         COL_WIDTHS["name"])        + " | "
        + fmt(bold("Port"),         COL_WIDTHS["port"],  ">")  + " | "
        + fmt(bold("Status"),       COL_WIDTHS["status"])       + " | "
        + fmt(bold("Known Peers"),  COL_WIDTHS["known_peers"])  + " | "
        + fmt(bold("CRDT Ledger"),  COL_WIDTHS["ledger_size"])  + " | "
        + fmt(bold("DHT ID Prefix"), COL_WIDTHS["node_id"])     + " |"
    )
    print(header)
    print(DIVIDER)

    # Table rows
    for info in nodes_info:
        status = info["status"]
        if status == "ALIVE":
            status_str = green(fmt("[ALIVE]", COL_WIDTHS["status"]))
        elif status == "BOOTING":
            status_str = yellow(fmt("[BOOT.]", COL_WIDTHS["status"]))
        elif status == "TIMEOUT":
            status_str = yellow(fmt("[TIME.]", COL_WIDTHS["status"]))
        else:
            status_str = red(fmt("[" + status[:6] + "]", COL_WIDTHS["status"]))

        # Color-code known_peers: green if fully meshed, yellow if partial
        kp = info["known_peers"]
        if isinstance(kp, int):
            if kp >= num_nodes - 1:
                kp_str = green(fmt(str(kp), COL_WIDTHS["known_peers"]))
            elif kp > 0:
                kp_str = yellow(fmt(str(kp), COL_WIDTHS["known_peers"]))
            else:
                kp_str = fmt(str(kp), COL_WIDTHS["known_peers"])
        else:
            kp_str = fmt(str(kp), COL_WIDTHS["known_peers"])

        # Color-code ledger: cyan if > 0
        ls = info["ledger_size"]
        if isinstance(ls, int) and ls > 0:
            ls_str = cyan(fmt(str(ls), COL_WIDTHS["ledger_size"]))
        else:
            ls_str = fmt(str(ls), COL_WIDTHS["ledger_size"])

        row = (
            "| "
            + fmt(info["name"],             COL_WIDTHS["name"])         + " | "
            + fmt(str(info["port"]),        COL_WIDTHS["port"],  ">")   + " | "
            + status_str                                                  + " | "
            + kp_str                                                      + " | "
            + ls_str                                                      + " | "
            + fmt(str(info["node_id_prefix"]), COL_WIDTHS["node_id"])    + " |"
        )
        print(row)

    print(DIVIDER)

    # Summary line
    alive = sum(1 for n in nodes_info if n["status"] == "ALIVE")
    fully_meshed = sum(
        1 for n in nodes_info
        if isinstance(n["known_peers"], int) and n["known_peers"] >= num_nodes - 1
    )
    total_ledger = sum(
        n["ledger_size"] for n in nodes_info if isinstance(n["ledger_size"], int)
    )
    print(
        f"\n  Alive: {green(str(alive))}/{num_nodes}  |  "
        f"Fully Meshed: {green(str(fully_meshed))}/{num_nodes}  |  "
        f"Total CRDT Events: {cyan(str(total_ledger))}"
    )


# ---------------------------------------------------------------------------
# Process spawning
# ---------------------------------------------------------------------------

def spawn_nodes(num_nodes: int, base_port: int) -> list[subprocess.Popen]:
    """Launch all N peer nodes as background subprocesses."""
    processes: list[subprocess.Popen] = []

    # --- Node 1: The seed (no bootstrap) ------------------------------------
    seed_port = base_port
    seed_cmd = [
        VENV_PY, PEER_SCRIPT,
        "--port", str(seed_port),
        "--file", GRAPH_FILES[0],
        "--name", "Peer_1",
    ]
    print(f"  [+] Launching Peer_1 on port {seed_port} (seed node)...")
    proc = subprocess.Popen(seed_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(proc)

    print("  Waiting 3s for seed node to initialize...")
    time.sleep(3)

    # --- Nodes 2..N: Bootstrap into the swarm --------------------------------
    bootstrap_url = f"http://localhost:{base_port}"
    for i in range(1, num_nodes):
        port = base_port + i
        name = f"Peer_{i + 1}"
        graph = GRAPH_FILES[i % len(GRAPH_FILES)]
        cmd = [
            VENV_PY, PEER_SCRIPT,
            "--port", str(port),
            "--file", graph,
            "--name", name,
            "--bootstrap", bootstrap_url,
        ]
        print(f"  [+] Launching {name} on port {port} (graph={graph}, bootstrap={bootstrap_url})...")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)
        time.sleep(0.5)   # Small stagger to avoid port conflicts during init

    print(f"\n  All {num_nodes} node(s) spawned. Starting dashboard in 2s...")
    time.sleep(2)
    return processes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a P2P drug discovery swarm and display a live dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nodes", type=int, default=3,
        help="Number of peer nodes to spawn.",
    )
    parser.add_argument(
        "--base-port", type=int, default=8001,
        help="Starting port number (nodes use base_port, base_port+1, ...).",
    )
    args = parser.parse_args()

    num_nodes = max(1, args.nodes)
    base_port = args.base_port
    ports = list(range(base_port, base_port + num_nodes))

    print(bold(cyan("\n  P2P Drug Discovery -- Network Orchestrator\n")))
    print(f"  Spawning {num_nodes} node(s) starting at port {base_port}...\n")

    processes: list[subprocess.Popen] = []
    start_time = time.time()

    try:
        processes = spawn_nodes(num_nodes, base_port)

        # ----------------------------------------------------------------
        # Live monitoring loop
        # ----------------------------------------------------------------
        while True:
            elapsed = time.time() - start_time
            nodes_info = [poll_node(port) for port in ports]
            render_dashboard(nodes_info, num_nodes, elapsed)
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\n  {yellow('Ctrl+C detected -- shutting down swarm...')}\n")

    finally:
        # Graceful shutdown -- terminate all child processes
        for i, proc in enumerate(processes):
            port = base_port + i
            name = f"Peer_{i + 1}"
            if proc.poll() is None:   # still running
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                    print(f"  [OK] {name} (port {port}) terminated.")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"  [!!] {name} (port {port}) force-killed.")
            else:
                print(f"  [--] {name} (port {port}) was already stopped.")

        print(f"\n  {green('All nodes stopped. Goodbye!')}\n")


if __name__ == "__main__":
    main()
