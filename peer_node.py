"""Hybrid P2P peer: local graph client and federated query initiator."""

import argparse
import asyncio
import hashlib
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import networkx as nx
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

G = None
PEER_NAME = "Unknown"
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
# Background heartbeat
# ---------------------------------------------------------------------------

async def heartbeat_loop() -> None:
    """Continuously ping every known peer every HEARTBEAT_INTERVAL_SECONDS seconds.

    Only logs state *changes* (online->offline or offline->online) to avoid
    flooding the console during normal operation.
    """
    print("[Heartbeat] Background task started.")
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
                except (httpx.TimeoutException, httpx.ConnectError, Exception):
                    KNOWN_PEERS[url]["status"] = "offline"
                    if previous_status != "offline":
                        print(f"[Heartbeat] [OFFLINE] Peer went OFFLINE: {url}")


# ---------------------------------------------------------------------------
# FastAPI lifespan: start/stop the heartbeat background task
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    task = asyncio.create_task(heartbeat_loop())
    print("[Lifespan] Heartbeat task created.")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print("[Lifespan] Heartbeat task stopped.")


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
    for target in local_response.get("targets", []):
        if isinstance(target, dict):
            evidence_path = {
                "peer_id": PEER_NAME,
                "path": target.get("path", "unknown"),
            }
            if "ml_score" in target:
                evidence_path["ml_score"] = target["ml_score"]
            evidence_paths.append(evidence_path)
        else:
            evidence_paths.append(
                {
                    "peer_id": PEER_NAME,
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


@app.get("/global_retrieve")
def global_retrieve(
    drug_id: str,
    task_type: str = "classification",
) -> dict:
    """Initiate a federated query from this peer (local-only until DHT routing lands)."""
    query_id = str(uuid.uuid4())
    local_response = run_local_retrieve(
        drug_id,
        task_type=task_type,
        include_weights=True,
    )
    sanitized_local = sanitize_peer_response_for_api(local_response)

    status = local_response.get("status", "unknown")
    available_peers = [PEER_NAME] if status in ("success", "not_found") else []
    missing_peers = [] if available_peers else [PEER_NAME]
    completeness_score = f"{len(available_peers)}/1"
    evidence_paths = _targets_to_evidence_paths(local_response)
    peer_metrics = local_response.get("metrics")
    local_confidence = float(local_response.get("local_confidence", 0.0))
    weights_summary = sanitized_local.get("model_weights_summary", {})

    return {
        "query": drug_id,
        "task_type": task_type,
        "query_id": query_id,
        "initiator_peer": PEER_NAME,
        "completeness_score": completeness_score,
        "retrieval_confidence_score": f"{int(local_confidence * 100)}%",
        "available_peers": available_peers,
        "available_clients": available_peers,
        "missing_peers": missing_peers,
        "missing_clients": missing_peers,
        "evidence_paths_count": len(evidence_paths),
        "evidence_paths": evidence_paths,
        "peer_link_prediction_metrics": {PEER_NAME: peer_metrics} if peer_metrics else {},
        "client_link_prediction_metrics": {PEER_NAME: peer_metrics} if peer_metrics else {},
        "global_aggregated_model": weights_summary,
        "raw_responses": [sanitized_local],
        "routing_mode": "local_only",
        "note": "P2P DHT propagation not yet implemented; only initiator peer queried.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a hybrid P2P peer node.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--name", type=str, default="Peer_1")
    args = parser.parse_args()

    PEER_NAME = args.name

    # Generate stable DHT node ID from peer identity
    NODE_ID, NODE_ID_HEX = _generate_node_id(PEER_NAME, args.port)
    print(f"[DHT] {PEER_NAME} node_id_hex_prefix={NODE_ID_HEX} (256-bit SHA-256 of '{PEER_NAME}:{args.port}')")

    load_peer_graph(args.file, peer_name=PEER_NAME)

    print(f"[Peer Node] Starting {PEER_NAME} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
