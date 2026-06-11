"""Hybrid P2P peer: local graph client and federated query initiator."""

import argparse
import asyncio
import hashlib
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import networkx as nx
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

G = None
PEER_NAME = "Unknown"
PEER_PORT = 8001
MODEL_VERSION = "v1.0.0"

NODE_TO_IDX: dict = {}
NUM_NODES = 0
EMBEDDING_DIM = 64
GLOBAL_EMBEDDING_VOCAB_SIZE = 100_000

POSITIVE_TRAIN_EDGES = 256
HOLDOUT_EDGES = 64
TRAINING_EPOCHS = 2
TOP_K = 50

REGRESSION_HIGH_AFFINITY = (5.0, 10.0)
REGRESSION_LOW_AFFINITY = (0.0, 3.0)
REGRESSION_SCORE_THRESHOLD = 5.0

# ---------------------------------------------------------------------------
# DHT Identity — generated at startup from PEER_NAME + PORT
# ---------------------------------------------------------------------------
NODE_ID: int = 0          # 256-bit integer Kademlia node ID
NODE_ID_HEX: str = ""     # First 16 hex chars for human-readable logs


def _generate_node_id(peer_name: str, port: int) -> tuple[int, str]:
    """Derive a stable 256-bit node ID from peer identity."""
    identity = f"{peer_name}:{port}"
    hex_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return int(hex_digest, 16), hex_digest[:16]


# ---------------------------------------------------------------------------
# Peer Membership State
# Key   : peer base URL  (e.g. "http://localhost:8002")
# Value : {
#     "last_seen" : ISO-8601 str | None,
#     "status"    : "active" | "offline",
#     "node_id"   : int | None   <- DHT identity of the remote peer
# }
# ---------------------------------------------------------------------------
KNOWN_PEERS: dict[str, dict] = {}

HEARTBEAT_INTERVAL_SECONDS = 10
HEARTBEAT_TIMEOUT_SECONDS = 3

# ---------------------------------------------------------------------------
# CRDT Ledger — LWW-Map (Last-Writer-Wins)
# Key   : update_id  (UUID string, globally unique per FL round event)
# Value : {
#     "status"    : str   (e.g. "update_committed"),
#     "timestamp" : float (Unix epoch — used for LWW conflict resolution),
#     "client_id" : str   (originating peer name)
# }
# ---------------------------------------------------------------------------
CRDT_LEDGER: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Bootstrap peer — URL of an existing peer to contact on startup (optional)
# ---------------------------------------------------------------------------
BOOTSTRAP_PEER: str | None = None


# ---------------------------------------------------------------------------
# Peer churn helper
# ---------------------------------------------------------------------------

def mark_peer_offline(peer_url: str) -> None:
    """Immediately mark a peer as offline after a network failure.

    Called both by the heartbeat loop on ping failure and by the DHT routing
    layer when a forwarded request fails mid-query.
    """
    info = KNOWN_PEERS.get(peer_url)
    if info and info.get("status") == "active":
        KNOWN_PEERS[peer_url]["status"] = "offline"
        print(f"[Fault Tolerance] Peer {peer_url} marked OFFLINE due to network error.")


# ---------------------------------------------------------------------------
# Background heartbeat
# ---------------------------------------------------------------------------

async def heartbeat_loop() -> None:
    """Continuously ping every known peer and discover new peers via gossip.

    On each cycle:
    - Pings every known peer to track online/offline status and capture node_id.
    - For each active peer, also fetches its /peers routing table and merges
      any newly discovered peers into local KNOWN_PEERS (gossip-based discovery).
    """
    print("[Heartbeat] Background task started.")
    my_url = f"http://localhost:{PEER_PORT}"
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if not KNOWN_PEERS:
            continue

        async with httpx.AsyncClient() as client:
            for url, peer_info in list(KNOWN_PEERS.items()):
                previous_status = peer_info.get("status", "offline")
                try:
                    resp = await client.get(
                        f"{url}/ping",
                        timeout=HEARTBEAT_TIMEOUT_SECONDS,
                    )
                    resp.raise_for_status()
                    now = datetime.now(timezone.utc).isoformat()
                    KNOWN_PEERS[url]["last_seen"] = now
                    KNOWN_PEERS[url]["status"] = "active"
                    # Capture node_id from ping response if not yet stored
                    ping_data = resp.json()
                    if KNOWN_PEERS[url].get("node_id") is None and "node_id" in ping_data:
                        KNOWN_PEERS[url]["node_id"] = ping_data["node_id"]
                    if previous_status != "active":
                        nid_short = str(KNOWN_PEERS[url].get("node_id", ""))[:8] or "unknown"
                        print(f"[Heartbeat] [ONLINE] Peer back ONLINE: {url} (node_id_prefix={nid_short})")

                    # Gossip-based peer discovery: pull routing table and merge new peers
                    try:
                        peers_resp = await client.get(
                            f"{url}/peers",
                            timeout=HEARTBEAT_TIMEOUT_SECONDS,
                        )
                        if peers_resp.status_code == 200:
                            remote_peers: dict = peers_resp.json().get("peers", {})
                            for remote_url, remote_info in remote_peers.items():
                                remote_url = remote_url.rstrip("/")
                                if remote_url == my_url:
                                    continue  # Never add self
                                if remote_url not in KNOWN_PEERS:
                                    KNOWN_PEERS[remote_url] = {
                                        "last_seen": remote_info.get("last_seen"),
                                        "status": remote_info.get("status", "active"),
                                        "node_id": remote_info.get("node_id"),
                                    }
                                    print(f"[Heartbeat] [DISCOVERY] New peer via gossip: {remote_url}")
                    except Exception:
                        pass  # Gossip failures are non-critical

                except (httpx.TimeoutException, httpx.ConnectError, Exception):
                    mark_peer_offline(url)
                    if previous_status != "offline":
                        print(f"[Heartbeat] [OFFLINE] Peer went OFFLINE: {url}")


# ---------------------------------------------------------------------------
# Background Anti-Entropy CRDT Gossip Loop
# ---------------------------------------------------------------------------

async def crdt_gossip_loop() -> None:
    """Anti-entropy background loop: periodically push-pull CRDT ledger with a random peer.

    Every 15 seconds this loop:
    - Selects one random ACTIVE peer from KNOWN_PEERS.
    - POSTs the local CRDT_LEDGER to that peer's /crdt_sync endpoint.
    - Receives the peer's own ledger in return and merges any new events
      (LWW strategy via merge_crdt_event).

    15s interval offsets from the 10s heartbeat to avoid network spikes.
    """
    GOSSIP_INTERVAL_SECONDS = 15
    print("[Gossip] Anti-entropy CRDT gossip task started.")
    while True:
        await asyncio.sleep(GOSSIP_INTERVAL_SECONDS)

        active_peers = [
            url for url, info in KNOWN_PEERS.items()
            if info.get("status") == "active"
        ]
        if not active_peers:
            continue

        peer_url = random.choice(active_peers)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{peer_url}/crdt_sync",
                    json=CRDT_LEDGER,
                    timeout=5.0,
                )
                resp.raise_for_status()
                remote_ledger: dict = resp.json().get("ledger", {})

                # Pull-merge: absorb any events the remote peer knew that we didn't
                pulled = 0
                for update_id, payload in remote_ledger.items():
                    if isinstance(payload, dict) and merge_crdt_event(update_id, payload):
                        pulled += 1

                if pulled > 0 or len(CRDT_LEDGER) > 0:
                    print(
                        f"[Gossip] Synced CRDT with {peer_url}. "
                        f"Pulled {pulled} new events. Local ledger size: {len(CRDT_LEDGER)}"
                    )
        except Exception as exc:
            # Gossip failures are non-critical — just log and continue
            print(f"[Gossip] WARNING: Sync with {peer_url} failed: {type(exc).__name__}")




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""

    # -----------------------------------------------------------------------
    # Bootstrap: join the network via one known peer (if --bootstrap given)
    # -----------------------------------------------------------------------
    if BOOTSTRAP_PEER:
        my_url = f"http://localhost:{PEER_PORT}"
        print(f"[Bootstrap] Contacting bootstrap peer {BOOTSTRAP_PEER} ...")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{BOOTSTRAP_PEER}/bootstrap",
                    params={"peer_url": my_url},
                    timeout=5.0,
                )
                resp.raise_for_status()
                routing_table: dict = resp.json().get("known_peers", {})
                merged = 0

                # Always register the bootstrap peer itself first
                # (a node never puts itself in its own KNOWN_PEERS)
                bootstrap_url = BOOTSTRAP_PEER.rstrip("/")
                if bootstrap_url != my_url and bootstrap_url not in KNOWN_PEERS:
                    KNOWN_PEERS[bootstrap_url] = {
                        "last_seen": datetime.now(timezone.utc).isoformat(),
                        "status": "active",
                        "node_id": None,
                    }
                    merged += 1
                    print(f"[Bootstrap] Registered bootstrap peer: {bootstrap_url}")

                # Merge the rest of the returned routing table
                for url, info in routing_table.items():
                    url = url.rstrip("/")
                    if url == my_url or url == bootstrap_url:
                        continue  # Skip self and already-added bootstrap peer
                    if url not in KNOWN_PEERS:
                        KNOWN_PEERS[url] = {
                            "last_seen": info.get("last_seen"),
                            "status": info.get("status", "active"),
                            "node_id": info.get("node_id"),
                        }
                        merged += 1
                print(
                    f"[Bootstrap] Merged {merged} peers from {BOOTSTRAP_PEER}. "
                    f"Total known: {len(KNOWN_PEERS)}"
                )
        except Exception as exc:
            print(f"[Bootstrap] WARNING: Could not contact {BOOTSTRAP_PEER}: {exc}")

    task_heartbeat = asyncio.create_task(heartbeat_loop())
    task_gossip = asyncio.create_task(crdt_gossip_loop())
    print("[Lifespan] Heartbeat task created.")
    print("[Lifespan] CRDT gossip task created.")
    try:
        yield
    finally:
        task_heartbeat.cancel()
        task_gossip.cancel()
        for t in (task_heartbeat, task_gossip):
            try:
                await t
            except asyncio.CancelledError:
                pass
        print("[Lifespan] Background tasks stopped.")


app = FastAPI(title="Hybrid P2P Peer Node", lifespan=lifespan)


def load_peer_graph(graph_path: str, peer_name: str | None = None) -> None:
    """Load the peer's private graph partition and build node index mappings."""
    global G, NODE_TO_IDX, NUM_NODES
    G = nx.read_graphml(graph_path)
    NODE_TO_IDX = {node: index for index, node in enumerate(G.nodes())}
    NUM_NODES = len(NODE_TO_IDX)
    label = peer_name or "peer"
    print(
        f"[Peer Node] {label} loaded {graph_path}: "
        f"{NUM_NODES} nodes, {G.number_of_edges()} edges mapped to indices"
    )


class LinkPredictor(nn.Module):
    """Embedding-based MLP for link classification or DeepDTA-style affinity regression."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        task_type: str = "classification",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.embeddings = nn.Embedding(
            num_embeddings=GLOBAL_EMBEDDING_VOCAB_SIZE,
            embedding_dim=embedding_dim,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, drug_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        drug_emb = self.embeddings(drug_idx.long())
        target_emb = self.embeddings(target_idx.long())
        combined = torch.cat([drug_emb, target_emb], dim=-1)
        return self.mlp(combined)

    def predict_score(self, drug_idx: int, target_idx: int) -> float:
        """Return a classification probability or raw affinity prediction."""
        self.eval()
        with torch.no_grad():
            drug_tensor = torch.tensor([drug_idx], dtype=torch.long)
            target_tensor = torch.tensor([target_idx], dtype=torch.long)
            logit = self.forward(drug_tensor, target_tensor)
            if self.task_type == "regression":
                return float(logit.item())
            return float(torch.sigmoid(logit).item())


def state_dict_to_lists(state_dict: dict) -> dict:
    return {key: tensor.detach().cpu().tolist() for key, tensor in state_dict.items()}


def lists_to_state_dict(lists_dict: dict) -> dict:
    """Convert a JSON list-based weight dict back into PyTorch tensors.

    This is the inverse of ``state_dict_to_lists`` and is used when a peer
    receives the global FedAvg model over HTTP so it can load it into a
    ``LinkPredictor`` via ``load_state_dict``.
    """
    return {key: torch.tensor(value) for key, value in lists_dict.items()}


def summarize_state_dict(state_dict: dict) -> dict:
    """Return layer shapes only — safe for browser/curl demos."""
    summary: dict = {}
    for layer_name, tensor in state_dict.items():
        shape = list(tensor.shape)
        summary[layer_name] = {"shape": shape, "num_params": int(tensor.numel())}
    return summary


def peer_checkpoint_path() -> str:
    """Return the canonical checkpoint archive path for this peer."""
    slug = PEER_NAME.strip().lower().replace(" ", "_")
    return f"checkpoints/{slug}.pt"


def compute_batch_hash(targets: list) -> str:
    """Build an SHA-256 digest over batched targets."""
    targets_string = ",".join(str(target) for target in targets)
    return hashlib.sha256(targets_string.encode("utf-8")).hexdigest()


def calculate_local_confidence(evidence_paths_count: int) -> float:
    """Estimate local retrieval confidence from the amount of graph evidence."""
    if evidence_paths_count == 0:
        return 0.0

    return round(min(0.99, 0.50 + (evidence_paths_count * 0.005)), 2)


def _sample_negative_edges(sample_count: int, rng: random.Random) -> list[tuple]:
    """Sample random node pairs that do not share an edge in the graph."""
    nodes = list(NODE_TO_IDX.keys())
    negative_edges: list[tuple] = []
    attempts = 0
    max_attempts = sample_count * 20

    while len(negative_edges) < sample_count and attempts < max_attempts:
        attempts += 1
        source = rng.choice(nodes)
        destination = rng.choice(nodes)
        if source == destination or G.has_edge(source, destination):
            continue
        negative_edges.append((source, destination))

    return negative_edges


def _empty_metrics(task_type: str) -> dict:
    """Return zeroed metrics for the active training task."""
    if task_type == "regression":
        return {"mse": 0.0, "r2": 0.0}
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "top_50_precision": 0.0,
    }


def _affinity_labels(edge_count: int, is_positive: bool, task_type: str, rng: random.Random) -> list[float]:
    """Build binary or mock DeepDTA affinity labels for an edge batch."""
    if task_type == "regression":
        low, high = REGRESSION_HIGH_AFFINITY if is_positive else REGRESSION_LOW_AFFINITY
        return [rng.uniform(low, high) for _ in range(edge_count)]
    label_value = 1.0 if is_positive else 0.0
    return [label_value] * edge_count


def _edges_to_index_tensors(edge_pairs: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert node-name edge pairs to drug/target index tensors for the model."""
    drug_indices = [NODE_TO_IDX[source] for source, _ in edge_pairs]
    target_indices = [NODE_TO_IDX[destination] for _, destination in edge_pairs]
    return (
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(target_indices, dtype=torch.long),
    )


def _evaluate_model(
    model: LinkPredictor,
    holdout_positive: list[tuple],
    holdout_negative: list[tuple],
    task_type: str,
    rng: random.Random,
) -> dict:
    """Compute classification or DeepDTA regression metrics on a holdout edge set."""
    evaluation_edges = holdout_positive + holdout_negative
    if not evaluation_edges:
        return _empty_metrics(task_type)

    y_true = _affinity_labels(len(holdout_positive), True, task_type, rng) + _affinity_labels(
        len(holdout_negative), False, task_type, rng
    )
    drug_tensor, target_tensor = _edges_to_index_tensors(evaluation_edges)

    model.eval()
    with torch.no_grad():
        logits = model(drug_tensor, target_tensor).squeeze(-1)
        if task_type == "regression":
            y_pred = logits.tolist()
        else:
            y_pred = torch.sigmoid(logits).tolist()

    if isinstance(y_pred, float):
        y_pred = [y_pred]

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"mse": round(float(mse), 3), "r2": round(float(r2), 3)}

    y_pred_binary = [1.0 if probability >= 0.5 else 0.0 for probability in y_pred]

    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    ranked = sorted(zip(y_pred, y_true), key=lambda pair: pair[0], reverse=True)
    top_k = ranked[: min(TOP_K, len(ranked))]
    if top_k:
        p_at_k = sum(label for _, label in top_k) / len(top_k)
    else:
        p_at_k = 0.0

    return {
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1_score": round(float(f1), 3),
        "top_50_precision": round(float(p_at_k), 3),
    }


def train_model(drug_id: str, task_type: str = "classification") -> tuple[LinkPredictor, dict]:
    """Train the link predictor on real graph edges and return it with metrics."""
    if task_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported task_type: {task_type}")

    seed = abs(hash(drug_id)) % (2**32)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = LinkPredictor(task_type=task_type)

    all_edges = list(G.edges())
    if not all_edges or NUM_NODES == 0:
        return model, _empty_metrics(task_type)

    rng.shuffle(all_edges)
    positive_train = all_edges[:POSITIVE_TRAIN_EDGES]
    holdout_positive = all_edges[POSITIVE_TRAIN_EDGES : POSITIVE_TRAIN_EDGES + HOLDOUT_EDGES]

    negative_pool = _sample_negative_edges(len(positive_train) + HOLDOUT_EDGES, rng)
    negative_train = negative_pool[: len(positive_train)]
    holdout_negative = negative_pool[len(positive_train) :]

    train_edges = positive_train + negative_train
    train_labels = _affinity_labels(len(positive_train), True, task_type, rng) + _affinity_labels(
        len(negative_train), False, task_type, rng
    )
    if not train_edges:
        return model, _empty_metrics(task_type)

    drug_tensor, target_tensor = _edges_to_index_tensors(train_edges)
    label_tensor = torch.tensor(train_labels, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if task_type == "regression":
        criterion = nn.MSELoss()
        loss_name = "MSE"
    else:
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCE"

    print(
        f"[ML Training] {PEER_NAME} | task={task_type} | drug={drug_id} | nodes={NUM_NODES} | "
        f"train_edges={len(train_edges)} (pos={len(positive_train)}, neg={len(negative_train)})"
    )

    model.train()
    for epoch in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        logits = model(drug_tensor, target_tensor).squeeze(-1)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()
        print(
            f"[ML Training] {PEER_NAME} epoch {epoch + 1}/{TRAINING_EPOCHS} "
            f"{loss_name} loss={loss.item():.4f}"
        )

    metrics = _evaluate_model(model, holdout_positive, holdout_negative, task_type, rng)
    print(f"[ML Training] {PEER_NAME} holdout metrics: {metrics}")
    return model, metrics


def score_candidate_paths(
    local_model: LinkPredictor,
    drug_id: str,
    candidate_targets: list,
) -> list[dict]:
    """Rank candidate graph edges by the local link-prediction model score."""
    scored_paths = []
    local_model.eval()

    drug_idx = NODE_TO_IDX.get(drug_id, 0)
    score_threshold = (
        REGRESSION_SCORE_THRESHOLD
        if local_model.task_type == "regression"
        else 0.5
    )

    with torch.no_grad():
        for target in candidate_targets:
            target_idx = NODE_TO_IDX.get(target, 0)
            score = local_model.predict_score(drug_idx, target_idx)
            if score > score_threshold:
                scored_paths.append(
                    {
                        "path": f"{drug_id} -> {target}",
                        "ml_score": round(score, 3),
                    }
                )

    scored_paths.sort(key=lambda item: item["ml_score"], reverse=True)
    return scored_paths[:50]


def _build_local_payload(
    *,
    drug_id: str,
    task_type: str,
    status: str,
    scored_paths: list,
    metrics: dict,
    state_dict: dict,
    include_weights: bool,
) -> dict:
    """Assemble the local peer retrieval response."""
    response_id = str(uuid.uuid4())
    payload = {
        "peer_id": PEER_NAME,
        "client_id": PEER_NAME,
        "drug_id": drug_id,
        "task_type": task_type,
        "targets": scored_paths,
        "status": status,
        "batch_hash": compute_batch_hash(scored_paths),
        "local_confidence": calculate_local_confidence(len(scored_paths)),
        "metrics": metrics,
        "response_id": response_id,
        "checkpoint_path": peer_checkpoint_path(),
        "model_version": MODEL_VERSION,
        "update_id": str(uuid.uuid4()),
    }
    if include_weights:
        payload["model_weights"] = state_dict_to_lists(state_dict)
    else:
        payload["model_weights_summary"] = summarize_state_dict(state_dict)
    return payload


def run_local_retrieve(
    drug_id: str,
    task_type: str = "classification",
    include_weights: bool = True,
) -> dict:
    """Execute local graph retrieval and model training for one drug query."""
    if task_type not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'",
        )
    if G is None:
        raise HTTPException(status_code=503, detail="Peer graph is not loaded")

    print(f"[Local Retrieve] {PEER_NAME} {task_type} training started for {drug_id}")
    local_model, metrics = train_model(drug_id, task_type=task_type)
    state_dict = local_model.state_dict()
    print(f"[Local Retrieve] {PEER_NAME} training complete for {drug_id}")

    if drug_id not in G:
        return _build_local_payload(
            drug_id=drug_id,
            task_type=task_type,
            status="not_found",
            scored_paths=[],
            metrics=metrics,
            state_dict=state_dict,
            include_weights=include_weights,
        )

    targets = list(G.neighbors(drug_id))
    scored_paths = score_candidate_paths(local_model, drug_id, targets)
    return _build_local_payload(
        drug_id=drug_id,
        task_type=task_type,
        status="success",
        scored_paths=scored_paths,
        metrics=metrics,
        state_dict=state_dict,
        include_weights=include_weights,
    )


def _targets_to_evidence_paths(local_response: dict) -> list[dict]:
    """Convert local scored targets into federated evidence path entries."""
    evidence_paths = []
    p_id = local_response.get("peer_id", PEER_NAME)
    for target in local_response.get("targets", []):
        if isinstance(target, dict):
            evidence_path = {
                "peer_id": p_id,
                "path": target.get("path", "unknown"),
            }
            if "ml_score" in target:
                evidence_path["ml_score"] = target["ml_score"]
            evidence_paths.append(evidence_path)
        else:
            evidence_paths.append(
                {
                    "peer_id": p_id,
                    "path": f"{local_response.get('drug_id', 'unknown')} -> {target}",
                }
            )
    return evidence_paths


def summarize_weights_lists(weights: dict) -> dict:
    """Summarize serialized list weights without materializing huge tensors."""
    summary: dict = {}
    for layer_name, tensor_values in weights.items():
        if not tensor_values:
            summary[layer_name] = {"shape": [0]}
        elif isinstance(tensor_values[0], list):
            summary[layer_name] = {
                "shape": [len(tensor_values), len(tensor_values[0])],
            }
        else:
            summary[layer_name] = {"shape": [len(tensor_values)]}
    return summary


def sanitize_peer_response_for_api(response: dict) -> dict:
    """Strip full weight tensors so browser/curl JSON stays small."""
    sanitized = {key: value for key, value in response.items() if key != "model_weights"}
    if "model_weights" in response and "model_weights_summary" not in sanitized:
        sanitized["model_weights_summary"] = summarize_weights_lists(response["model_weights"])
    return sanitized


# ---------------------------------------------------------------------------
# FedAvg aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_models(client_weights_list: list[dict]) -> dict:
    """Federated Averaging (FedAvg) over a list of serialized PyTorch state dicts.

    Each value in the state dict is a plain Python list (tensor serialized via
    ``state_dict_to_lists``).  We perform an element-wise mean across all peers
    and return the averaged state dict in the same list format.

    Args:
        client_weights_list: List of state dicts, each mapping layer name to a
                             nested list of floats.

    Returns:
        Averaged state dict (same structure), or empty dict if input is empty.
    """
    if not client_weights_list:
        return {}

    # Use first client's keys as reference
    reference = client_weights_list[0]
    aggregated: dict = {}

    for layer_name, ref_tensor in reference.items():
        # Collect the same layer from every peer
        all_tensors = []
        for weights in client_weights_list:
            t = weights.get(layer_name)
            if t is not None:
                all_tensors.append(t)

        if not all_tensors:
            aggregated[layer_name] = ref_tensor
            continue

        n = len(all_tensors)

        # 2-D tensor (e.g. embedding or linear weight matrix)
        if isinstance(ref_tensor[0], list):
            rows = len(ref_tensor)
            cols = len(ref_tensor[0])
            avg = [
                [
                    sum(all_tensors[p][r][c] for p in range(n)) / n
                    for c in range(cols)
                ]
                for r in range(rows)
            ]
        else:
            # 1-D tensor (bias / output layer)
            avg = [
                sum(all_tensors[p][i] for p in range(n)) / n
                for i in range(len(ref_tensor))
            ]

        aggregated[layer_name] = avg

    return aggregated


def summarize_state_dict_lists(state_dict: dict) -> dict:
    """Convert an aggregated list-based state dict into human-readable shape info."""
    summary: dict = {}
    for layer_name, tensor_values in state_dict.items():
        if not tensor_values:
            summary[layer_name] = {"shape": [0]}
        elif isinstance(tensor_values[0], list):
            summary[layer_name] = {
                "shape": [len(tensor_values), len(tensor_values[0])],
            }
        else:
            summary[layer_name] = {"shape": [len(tensor_values)]}
    return summary


def sanitize_raw_responses_for_api(raw_responses: list[dict]) -> list[dict]:
    """Strip model_weights from every response in the list, replacing with shape summary."""
    return [sanitize_peer_response_for_api(r) for r in raw_responses]


# ---------------------------------------------------------------------------
# Membership endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping() -> dict:
    """Health-check / heartbeat target. Returns this peer's live identity and DHT node ID."""
    return {
        "status": "alive",
        "peer_id": PEER_NAME,
        "node_id": NODE_ID,
        "node_id_hex_prefix": NODE_ID_HEX,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/add_peer")
def add_peer(url: str) -> dict:
    """Manually register a peer URL into this node's membership list.

    Args:
        url: Base URL of the peer to register, e.g. ``http://localhost:8002``.
    """
    url = url.rstrip("/")
    if url in KNOWN_PEERS:
        return {
            "result": "already_known",
            "url": url,
            "peer_info": KNOWN_PEERS[url],
        }

    KNOWN_PEERS[url] = {
        "last_seen": None,
        "status": "active",
        "node_id": None,  # Will be populated on next heartbeat cycle
    }
    print(f"[Membership] [+] New peer added: {url} (total known: {len(KNOWN_PEERS)})")
    return {
        "result": "added",
        "url": url,
        "total_known_peers": len(KNOWN_PEERS),
    }


@app.get("/peers")
def list_peers() -> dict:
    """Return the current peer membership list with status and DHT node IDs."""
    return {
        "self": PEER_NAME,
        "self_node_id": NODE_ID,
        "self_node_id_hex_prefix": NODE_ID_HEX,
        "total_known": len(KNOWN_PEERS),
        "peers": KNOWN_PEERS,
    }


@app.post("/bootstrap")
def bootstrap(peer_url: str) -> dict:
    """Network-entry endpoint: register a new peer and share this node's routing table.

    A new peer contacts exactly one existing peer via this endpoint.  The
    existing peer:
    1. Registers the newcomer in its own KNOWN_PEERS (Step A).
    2. Returns its full routing table to the newcomer so it can discover
       the rest of the network (Step B).

    Args:
        peer_url: Base URL of the newcomer, e.g. ``http://localhost:8003``.
    """
    peer_url = peer_url.rstrip("/")
    my_url = f"http://localhost:{PEER_PORT}"

    # Step A: Register the newcomer if not already known
    if peer_url == my_url:
        return {"result": "self", "known_peers": KNOWN_PEERS}

    if peer_url not in KNOWN_PEERS:
        KNOWN_PEERS[peer_url] = {
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "node_id": None,  # Heartbeat will resolve this
        }
        print(f"[Bootstrap] [+] New peer joined: {peer_url} (total known: {len(KNOWN_PEERS)})")
    else:
        # Refresh last_seen and ensure active
        KNOWN_PEERS[peer_url]["last_seen"] = datetime.now(timezone.utc).isoformat()
        KNOWN_PEERS[peer_url]["status"] = "active"
        print(f"[Bootstrap] [~] Existing peer re-joined: {peer_url}")

    # Step B: Return full routing table to the newcomer
    return {
        "result": "welcome",
        "introducer": PEER_NAME,
        "known_peers": KNOWN_PEERS,
    }


# ---------------------------------------------------------------------------
# DHT Routing — XOR distance math & closest-peer lookup
# ---------------------------------------------------------------------------

def calculate_xor_distance(id1: int, id2: int) -> int:
    """Compute Kademlia-style XOR distance between two 256-bit node IDs.

    Lower result means the two nodes are 'closer' in the DHT ring.
    """
    return id1 ^ id2


@app.get("/closest_peers")
def closest_peers(target_id: str, limit: int = 2) -> dict:
    """Return the ``limit`` peers closest to ``target_id`` by XOR distance.

    Args:
        target_id: The target DHT key as a decimal integer string or
                   a 0x-prefixed / bare hex string.
        limit:     How many closest peers to return (default 2).

    The response includes this node itself as a candidate so the caller
    always gets a complete Kademlia k-bucket view.
    """
    # --- Parse target_id (accept decimal int string or hex string) ----------
    try:
        if isinstance(target_id, str) and target_id.startswith("0x"):
            target_int = int(target_id, 16)
        else:
            # Try decimal first, fall back to bare hex
            try:
                target_int = int(target_id)
            except ValueError:
                target_int = int(target_id, 16)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"target_id '{target_id}' is not a valid integer or hex string. "
                "Pass a decimal integer or a hex string (optionally prefixed with 0x)."
            ),
        )

    # --- Build candidate list (self + active known peers) -------------------
    candidates: list[dict] = []

    # Include self
    candidates.append({
        "url": "self",
        "peer_id": PEER_NAME,
        "node_id": NODE_ID,
        "node_id_hex_prefix": NODE_ID_HEX,
        "xor_distance": calculate_xor_distance(NODE_ID, target_int),
        "status": "active",
    })

    # Include active known peers that have a resolved node_id
    for url, info in KNOWN_PEERS.items():
        if info.get("status") != "active":
            continue
        peer_node_id = info.get("node_id")
        if peer_node_id is None:
            continue
        candidates.append({
            "url": url,
            "peer_id": info.get("peer_id", "unknown"),
            "node_id": peer_node_id,
            "node_id_hex_prefix": hex(peer_node_id)[:18],
            "xor_distance": calculate_xor_distance(peer_node_id, target_int),
            "status": "active",
        })

    # --- Sort by XOR distance (ascending = closest first) -------------------
    candidates.sort(key=lambda c: c["xor_distance"])
    closest = candidates[:limit]

    return {
        "target_id": target_int,
        "limit": limit,
        "total_candidates": len(candidates),
        "closest_peers": closest,
    }


# ---------------------------------------------------------------------------
# CRDT helper functions
# ---------------------------------------------------------------------------

def merge_crdt_event(update_id: str, payload: dict) -> bool:
    """Merge a remote event into the local CRDT ledger using LWW strategy.

    Args:
        update_id: Globally unique identifier for the FL round event.
        payload:   Dict containing at minimum ``timestamp`` (float),
                   ``status`` (str), and ``client_id`` (str).

    Returns:
        ``True``  if the event was newly added or overwrote a stale entry.
        ``False`` if the existing entry is newer-or-equal (event rejected).
    """
    incoming_ts = payload.get("timestamp", 0.0)

    if update_id not in CRDT_LEDGER:
        CRDT_LEDGER[update_id] = payload
        return True

    # LWW: only overwrite when the incoming timestamp is strictly newer
    if incoming_ts > CRDT_LEDGER[update_id].get("timestamp", 0.0):
        CRDT_LEDGER[update_id] = payload
        return True

    return False  # stale or duplicate — reject


def check_if_duplicate(update_id: str) -> bool:
    """Return True if update_id is already committed in the local CRDT ledger."""
    entry = CRDT_LEDGER.get(update_id)
    if entry is None:
        return False
    return entry.get("status") == "update_committed"


# ---------------------------------------------------------------------------
# CRDT endpoints
# ---------------------------------------------------------------------------

@app.post("/crdt_sync")
def crdt_sync(incoming_ledger: dict) -> dict:
    """Merge an incoming CRDT ledger from a remote peer (push-pull gossip receiver).

    Accepts a JSON dict mapping update_id -> event payload and applies
    the LWW merge strategy to each entry.  Returns the **local ledger** so
    the caller can pull-merge any events it had not yet seen.

    Returns:
        Merge summary plus the full local CRDT_LEDGER for the caller to pull.
    """
    merged = 0
    ignored = 0

    for update_id, payload in incoming_ledger.items():
        if not isinstance(payload, dict):
            ignored += 1
            continue
        if merge_crdt_event(update_id, payload):
            merged += 1
        else:
            ignored += 1

    print(
        f"[CRDT] Sync received {len(incoming_ledger)} events: "
        f"{merged} merged, {ignored} ignored. Ledger size: {len(CRDT_LEDGER)}"
    )
    return {
        "result": "sync_complete",
        "received": len(incoming_ledger),
        "merged": merged,
        "ignored": ignored,
        "ledger_size": len(CRDT_LEDGER),
        "ledger": CRDT_LEDGER,   # Return own ledger so caller can pull-merge
    }


@app.get("/crdt_state")
def crdt_state() -> dict:
    """Return the full local CRDT ledger for inspection or gossip seeding.

    This replaces the old SQLite-backed /audit data view.
    """
    return {
        "peer_id": PEER_NAME,
        "ledger_size": len(CRDT_LEDGER),
        "ledger": CRDT_LEDGER,
    }


# ---------------------------------------------------------------------------
# Retrieval endpoints
# ---------------------------------------------------------------------------

@app.get("/local_retrieve")
def local_retrieve(
    drug_id: str,
    task_type: str = "classification",
    include_weights: bool = False,
) -> dict:
    """Run the local PyTorch model on this peer's private graph partition."""
    return run_local_retrieve(drug_id, task_type=task_type, include_weights=include_weights)


class DHTRetrieveRequest(BaseModel):
    drug_id: str
    task_type: str = "classification"
    query_id: str
    ttl: int
    visited_peers: list[str] = []


class GlobalModelPayload(BaseModel):
    """Payload sent by the initiator to each participant after FedAvg."""
    query_id: str
    drug_id: str
    task_type: str = "classification"
    global_weights: dict          # Raw FedAvg weight arrays (list-of-lists format)
    local_metrics: dict           # The original per-peer metrics for comparison


async def dht_retrieve_internal(
    drug_id: str,
    task_type: str,
    query_id: str,
    ttl: int,
    visited_peers: list[str]
) -> list[dict]:
    """Internal DHT query propagation logic."""
    my_url = f"http://localhost:{PEER_PORT}"

    # Step A (Local Work)
    local_response = run_local_retrieve(
        drug_id,
        task_type=task_type,
        include_weights=True,
    )

    # Log/Commit to local CRDT ledger
    update_id = local_response.get("update_id", str(uuid.uuid4()))
    if not check_if_duplicate(update_id):
        crdt_payload = {
            "status": "update_committed",
            "timestamp": time.time(),
            "client_id": PEER_NAME,
        }
        merge_crdt_event(update_id, crdt_payload)
        print(f"[CRDT] Committed update_id={update_id[:8]}... ledger_size={len(CRDT_LEDGER)}")

    # Add own base URL to visited list
    updated_visited = list(visited_peers)
    if my_url not in updated_visited:
        updated_visited.append(my_url)

    # Step B (Propagation)
    forward_results = []
    if ttl > 0:
        # Calculate target ID from drug_id SHA-256 hash
        target_id_hex = hashlib.sha256(drug_id.encode("utf-8")).hexdigest()
        target_id_int = int(target_id_hex, 16)

        # Build candidates list (self + active known peers)
        candidates = []
        for url, info in KNOWN_PEERS.items():
            if info.get("status") != "active":
                continue
            peer_node_id = info.get("node_id")
            if peer_node_id is None:
                continue
            candidates.append({
                "url": url,
                "node_id": peer_node_id,
                "xor_distance": calculate_xor_distance(peer_node_id, target_id_int)
            })

        # Sort candidates by XOR distance
        candidates.sort(key=lambda c: c["xor_distance"])

        # Filter out already visited peers
        to_forward = [c for c in candidates if c["url"] not in updated_visited]

        # Pick top 2 closest remaining peers as initial targets
        targets = to_forward[:2]
        # Keep the rest as fallbacks (index 2 onwards) for fault tolerance
        fallback_pool = list(to_forward[2:])

        if targets:
            print(f"[DHT] {PEER_NAME} forwarding query {query_id[:8]} to: {[t['url'] for t in targets]} (ttl={ttl})")

            # Add all target URLs we are querying to visited_peers to prevent loopback/cross-querying
            next_visited = list(updated_visited)
            for t in targets:
                if t["url"] not in next_visited:
                    next_visited.append(t["url"])

            async def forward_to_peer(peer_url: str) -> list | dict:
                """Forward DHT query to a single peer; mark it offline on failure."""
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{peer_url}/dht_retrieve",
                            json={
                                "drug_id": drug_id,
                                "task_type": task_type,
                                "query_id": query_id,
                                "ttl": ttl - 1,
                                "visited_peers": next_visited
                            },
                            timeout=30.0
                        )
                        if response.status_code == 200:
                            return response.json()
                        else:
                            print(f"[DHT] Failed to query {peer_url}, status code: {response.status_code}")
                            mark_peer_offline(peer_url)
                            return {"status": "failed", "url": peer_url}
                except Exception as e:
                    print(f"[DHT] Error querying {peer_url}: {type(e).__name__}")
                    mark_peer_offline(peer_url)
                    return {"status": "failed", "url": peer_url}

            tasks = [forward_to_peer(t["url"]) for t in targets]
            gathered = list(await asyncio.gather(*tasks))

            # --- Dynamic Routing Fallback -----------------------------------
            # For every target that failed, try the next closest peer in the
            # fallback pool until we either succeed or run out of candidates.
            final_results = []
            for result in gathered:
                if isinstance(result, dict) and result.get("status") == "failed":
                    # This target failed — try fallback peers one by one
                    recovered = False
                    while fallback_pool:
                        fallback = fallback_pool.pop(0)
                        fallback_url = fallback["url"]
                        if fallback_url in next_visited:
                            continue
                        print(
                            f"[Fault Tolerance] Falling back to {fallback_url} "
                            f"after failure in query {query_id[:8]}"
                        )
                        next_visited.append(fallback_url)
                        fallback_result = await forward_to_peer(fallback_url)
                        if isinstance(fallback_result, list):
                            final_results.extend(fallback_result)
                            recovered = True
                            break
                        # fallback also failed — loop continues to next candidate
                    if not recovered:
                        print(f"[Fault Tolerance] No fallback peers left for query {query_id[:8]}.")
                elif isinstance(result, list):
                    final_results.extend(result)

            forward_results.extend(final_results)

    # Step C (Bubble Up)
    return [local_response] + forward_results


@app.post("/dht_retrieve")
async def dht_retrieve(request: DHTRetrieveRequest) -> list[dict]:
    """Internal DHT query propagation endpoint."""
    print(f"[DHT] Received /dht_retrieve for drug={request.drug_id} (ttl={request.ttl})")
    return await dht_retrieve_internal(
        drug_id=request.drug_id,
        task_type=request.task_type,
        query_id=request.query_id,
        ttl=request.ttl,
        visited_peers=request.visited_peers
    )


@app.post("/receive_global_model")
async def receive_global_model(payload: GlobalModelPayload) -> dict:
    """Accept the FedAvg global model from the initiator and evaluate it locally.

    Workflow:
    1. Reconstruct a ``LinkPredictor`` from the received weight arrays.
    2. Re-create the *exact* local holdout set used during the original training
       run (same ``hash(drug_id)`` seed, same RNG sequence).
    3. Evaluate the global model on that holdout set.
    4. Log a side-by-side comparison with the original local metrics.
    """
    if G is None:
        return {"status": "skipped", "reason": "graph not loaded"}

    drug_id = payload.drug_id
    task_type = payload.task_type
    query_id = payload.query_id

    # -----------------------------------------------------------------------
    # Reconstruct the global model from received weights
    # -----------------------------------------------------------------------
    global_model = LinkPredictor(task_type=task_type)
    try:
        state_dict = lists_to_state_dict(payload.global_weights)
        global_model.load_state_dict(state_dict)
    except Exception as exc:
        print(f"[FedAvg Dissemination] ERROR loading global weights for {drug_id}: {exc}")
        return {"status": "error", "reason": str(exc)}

    # -----------------------------------------------------------------------
    # Recreate the exact holdout set (same seed as training)
    # -----------------------------------------------------------------------
    seed = abs(hash(drug_id)) % (2**32)
    rng = random.Random(seed)

    all_edges = list(G.edges())
    if not all_edges or NUM_NODES == 0:
        return {"status": "skipped", "reason": "graph has no edges"}

    rng.shuffle(all_edges)
    holdout_positive = all_edges[POSITIVE_TRAIN_EDGES: POSITIVE_TRAIN_EDGES + HOLDOUT_EDGES]

    negative_pool = _sample_negative_edges(POSITIVE_TRAIN_EDGES + HOLDOUT_EDGES, rng)
    holdout_negative = negative_pool[POSITIVE_TRAIN_EDGES:]

    # -----------------------------------------------------------------------
    # Evaluate the global model on the local holdout
    # -----------------------------------------------------------------------
    # Re-seed RNG to match the exact state that _evaluate_model expects
    eval_rng = random.Random(seed)
    global_metrics = _evaluate_model(
        global_model, holdout_positive, holdout_negative, task_type, eval_rng
    )

    # -----------------------------------------------------------------------
    # Log the before-vs-after comparison
    # -----------------------------------------------------------------------
    local_m = payload.local_metrics
    if task_type == "regression":
        old_val = local_m.get("r2", "N/A")
        new_val = global_metrics.get("r2", "N/A")
        metric_label = "R²"
    else:
        old_val = local_m.get("f1_score", "N/A")
        new_val = global_metrics.get("f1_score", "N/A")
        metric_label = "F1"

    print(
        f"[FedAvg Dissemination] {PEER_NAME} | drug={drug_id} | query={query_id[:8]} | "
        f"Local {metric_label}: {old_val} -> Global {metric_label}: {new_val}"
    )

    return {
        "status": "evaluated",
        "peer_id": PEER_NAME,
        "drug_id": drug_id,
        "query_id": query_id,
        "local_metrics": local_m,
        "global_metrics": global_metrics,
    }


@app.get("/global_retrieve")
async def global_retrieve(
    drug_id: str,
    task_type: str = "classification",
    ttl: int = 2,
) -> dict:
    """Initiate a federated DHT query, aggregate results via FedAvg, return summary."""
    query_id = str(uuid.uuid4())
    print(f"[FedAvg] Initiating global query={query_id[:8]} for drug={drug_id}")

    # -----------------------------------------------------------------------
    # Step 1: DHT propagation — collect raw responses from all reachable peers
    # -----------------------------------------------------------------------
    raw_responses = await dht_retrieve_internal(
        drug_id=drug_id,
        task_type=task_type,
        query_id=query_id,
        ttl=ttl,
        visited_peers=[]
    )
    # Deduplicate responses by peer_id — fallback routing can cause the same
    # peer to respond via two different paths; keep the first occurrence only.
    seen_peer_ids: set = set()
    deduped: list[dict] = []
    for resp in raw_responses:
        pid = resp.get("peer_id", "unknown")
        if pid not in seen_peer_ids:
            seen_peer_ids.add(pid)
            deduped.append(resp)
    raw_responses = deduped

    # -----------------------------------------------------------------------
    # Step 2: Collect inputs from raw responses
    # -----------------------------------------------------------------------
    client_weights_list: list[dict] = []
    peer_metrics: dict = {}
    evidence_paths: list = []
    available_peers: list = []
    missing_peers: list = []
    confidence_scores: list[float] = []
    local_update_id = None

    for resp in raw_responses:
        p_id = resp.get("peer_id", "unknown")
        status = resp.get("status", "unknown")

        if status in ("success", "not_found"):
            available_peers.append(p_id)
        else:
            missing_peers.append(p_id)

        # Collect model weights for FedAvg
        weights = resp.get("model_weights")
        if weights:
            client_weights_list.append(weights)

        # Collect per-peer metrics
        m = resp.get("metrics")
        if m:
            peer_metrics[p_id] = m

        # Collect evidence paths (drug->target paths)
        evidence_paths.extend(_targets_to_evidence_paths(resp))

        # Collect confidence scores
        conf = resp.get("local_confidence")
        if conf is not None:
            confidence_scores.append(float(conf))

        # Track local node's update_id for CRDT reference
        if resp.get("peer_id") == PEER_NAME:
            local_update_id = resp.get("update_id")

    # Completeness: how many peers responded vs total ACTIVE known peers (self + active)
    active_known = sum(1 for info in KNOWN_PEERS.values() if info.get("status") == "active")
    total_network_size = 1 + active_known  # self + active known peers
    completeness_score = f"{len(available_peers)}/{total_network_size}"

    # -----------------------------------------------------------------------
    # Step 3: FedAvg — element-wise average of all collected weights
    # -----------------------------------------------------------------------
    fedavg_weights = aggregate_models(client_weights_list)
    global_aggregated_model = summarize_state_dict_lists(fedavg_weights)
    print(
        f"[FedAvg] Aggregated {len(client_weights_list)} weight sets. "
        f"Layers: {list(global_aggregated_model.keys())[:3]}..."
    )

    # -----------------------------------------------------------------------
    # Step 4: Global confidence — average across all responding peers
    # -----------------------------------------------------------------------
    global_confidence = (
        round(sum(confidence_scores) / len(confidence_scores), 4)
        if confidence_scores else 0.0
    )

    # -----------------------------------------------------------------------
    # Step 5: Federated metrics — average per-metric across all peers
    # -----------------------------------------------------------------------
    federated_metrics: dict = {}
    if peer_metrics:
        if task_type == "classification":
            metric_keys = ["precision", "recall", "f1_score", "top_50_precision"]
        else:
            metric_keys = ["mse", "r2"]

        for key in metric_keys:
            vals = [
                m[key] for m in peer_metrics.values()
                if key in m and m[key] is not None
            ]
            if vals:
                federated_metrics[key] = round(sum(vals) / len(vals), 4)

    print(
        f"[FedAvg] Federated metrics ({task_type}): {federated_metrics} "
        f"(global_confidence={global_confidence})"
    )

    # -----------------------------------------------------------------------
    # Step 6: Sanitize raw responses (strip heavy weight arrays)
    # -----------------------------------------------------------------------
    sanitized_responses = sanitize_raw_responses_for_api(raw_responses)

    # -----------------------------------------------------------------------
    # Step 7: Disseminate global model to participant peers (fire-and-forget)
    # -----------------------------------------------------------------------
    my_url = f"http://localhost:{PEER_PORT}"

    # Build a lookup: peer_id -> original metrics (for comparison at receiver)
    peer_id_to_metrics = {p_id: m for p_id, m in peer_metrics.items()}

    # Build a lookup: peer_id -> peer base URL
    # KNOWN_PEERS keys are URLs; we need peer_id from the ping node_id info.
    # However peer_id is stored in raw_responses alongside the URL info.
    # We construct a mapping from peer_id -> URL using KNOWN_PEERS + raw_responses.
    peer_id_to_url: dict = {}
    for resp in raw_responses:
        p_id = resp.get("peer_id", "")
        # Try matching against KNOWN_PEERS by scanning for a matching response
        # The initiator itself is always at my_url — skip it.
        if p_id and p_id != PEER_NAME:
            # Infer URL: check each known peer's /ping-cached peer_id if available,
            # otherwise fall back to the port convention used in tests.
            for url, info in KNOWN_PEERS.items():
                # Match by node_id that was resolved during heartbeat if possible;
                # otherwise accept any active known peer URL that we haven't mapped yet.
                if info.get("status") == "active" and url not in peer_id_to_url.values():
                    peer_id_to_url[p_id] = url
                    break

    async def _broadcast_global_model(peer_url: str, peer_id: str) -> None:
        """Fire-and-forget: POST the global model to one participant peer."""
        original_metrics = peer_id_to_metrics.get(peer_id, {})
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{peer_url}/receive_global_model",
                    json={
                        "query_id": query_id,
                        "drug_id": drug_id,
                        "task_type": task_type,
                        "global_weights": fedavg_weights,   # Raw list-of-lists weights
                        "local_metrics": original_metrics,
                    },
                    timeout=60.0,
                )
            print(
                f"[FedAvg Dissemination] Sent global model to {peer_id} @ {peer_url} "
                f"for query={query_id[:8]}"
            )
        except Exception as exc:
            print(
                f"[FedAvg Dissemination] WARNING: Could not reach {peer_id} @ {peer_url}: "
                f"{type(exc).__name__}"
            )

    # Schedule broadcasts as background tasks — do not await them
    for p_id, p_url in peer_id_to_url.items():
        if p_url != my_url:  # Never send to self
            asyncio.create_task(_broadcast_global_model(p_url, p_id))

    dissemination_targets = list(peer_id_to_url.keys())
    print(
        f"[FedAvg Dissemination] Broadcasting global model to {len(dissemination_targets)} "
        f"peer(s): {dissemination_targets}"
    )

    return {
        "query": drug_id,
        "task_type": task_type,
        "query_id": query_id,
        "initiator_peer": PEER_NAME,
        "completeness_score": completeness_score,
        "global_confidence": global_confidence,
        "available_peers": available_peers,
        "missing_peers": missing_peers,
        "evidence_paths_count": len(evidence_paths),
        "evidence_paths": evidence_paths,
        "peer_link_prediction_metrics": peer_metrics,
        "federated_link_prediction_metrics": federated_metrics,
        "global_aggregated_model": global_aggregated_model,
        "dissemination_targets": dissemination_targets,
        "raw_responses": sanitized_responses,
        "routing_mode": "dht_propagation",
        "crdt_update_id": local_update_id,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a hybrid P2P peer node.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--name", type=str, default="Peer_1")
    parser.add_argument(
        "--bootstrap",
        type=str,
        default=None,
        help="URL of an existing peer to bootstrap from (e.g. http://localhost:8001)",
    )
    args = parser.parse_args()

    PEER_NAME = args.name
    PEER_PORT = args.port
    if args.bootstrap:
        BOOTSTRAP_PEER = args.bootstrap.rstrip("/")
        print(f"[Bootstrap] Will join network via {BOOTSTRAP_PEER}")

    # Generate stable DHT node ID from peer identity
    NODE_ID, NODE_ID_HEX = _generate_node_id(PEER_NAME, args.port)
    print(f"[DHT] {PEER_NAME} node_id_hex_prefix={NODE_ID_HEX} (256-bit SHA-256 of '{PEER_NAME}:{args.port}')")

    load_peer_graph(args.file, peer_name=PEER_NAME)

    print(f"[Peer Node] Starting {PEER_NAME} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
