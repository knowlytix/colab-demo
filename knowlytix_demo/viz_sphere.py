"""Build an interactive 3D sphere visualization of a GMS multi-hop query.

VENDORED from the GMS upstream `demo/viz/sphere_demo.py` (commit a4112c6).
Changes from upstream, applied during vendoring:
  1. dev-namespace imports rewritten to the public `knowlytix.*` namespace
     (`gms.config`/`gms.graph.gkg` -> `knowlytix.core.*`; `eval.scorers` ->
     `knowlytix.benchmark.eval.scorers`).
  2. one compat shim in `run_query`: `score_path_validity` is post-rc31, so it
     falls back to an equivalent triple-membership check on older wheels.
Re-sync from upstream by re-copying and re-applying these.

Usage:
    python -m demo.viz.sphere_demo \\
        --store cap_experiments/ch04_ds_policy_cap_lib \\
        --source data_engineer \\
        --relations has_role \\
        --out demo.html

The visualization is a 3D projection of the high-dimensional unit hypersphere.
Entity embeddings live on S^{m-1}; we project to R^3 via PCA and renormalize
onto S^2 for display. Geodesic arcs draw relations; a highlighted polyline
draws the composed multi-hop transport with the retrieved tail circled.

This is an EXPLANATORY view, not an exact metric chart. Per-hop confidence,
geodesic distance, and the retrieved tail are computed in the original
high-dimensional space; only the layout is projected.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

from knowlytix.core.config import GeometryConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedStore:
    model: GeometricKnowledgeGraph
    entity_names: list[str]
    relation_names: list[str]
    entity_to_idx: dict[str, int]
    relation_to_idx: dict[str, int]
    V_high: torch.Tensor          # (n, m) entity v-embeddings on S^{m-1}
    R_mats: torch.Tensor          # (R, m, m) relation rotors
    triples: list[tuple[str, str, str]]
    enm: dict                     # raw enm.json contents (may be empty)
    store_path: Path


def load_store(store_path: str | Path) -> LoadedStore:
    """Load a saved GMS store (adapter + model + triples + optional enm)."""
    store_path = Path(store_path)
    adapter = json.loads((store_path / "adapter.json").read_text())
    dims = json.loads((store_path / "model_dims.json").read_text())

    state = torch.load(store_path / "model.pt", map_location="cpu", weights_only=True)
    m, d_v = state["dual_emb.P_v"].shape
    d_u = state["dual_emb.P_u"].shape[1]
    cfg = GeometryConfig(d_v=d_v, d_u=d_u, m=m, d=m)

    model = GeometricKnowledgeGraph(
        num_entities=dims["num_entities"],
        num_relations=dims["num_relations"],
        cfg=cfg,
        cap_enabled=dims.get("cap_enabled", False),
        cap_rho_max=dims.get("cap_rho_max", 1.5707963267948966),
        cap_use_diag=dims.get("cap_use_diag", True),
    )
    model.load_state_dict(state)
    model.eval()

    entity_to_idx = adapter["entity_to_idx"]
    relation_to_idx = adapter["relation_to_idx"]
    entity_names = [None] * len(entity_to_idx)
    for name, idx in entity_to_idx.items():
        entity_names[idx] = name
    relation_names = [None] * len(relation_to_idx)
    for name, idx in relation_to_idx.items():
        relation_names[idx] = name

    with torch.no_grad():
        V_high = model.dual_emb.project_v(torch.arange(model.num_entities))
        R_mats = torch.stack([r.get_rotor() for r in model.relation_rotors], dim=0)

    triples_raw = json.loads((store_path / "triples.json").read_text())
    triples: list[tuple[str, str, str]] = [tuple(t) for t in triples_raw]

    enm_path = store_path / "enm.json"
    enm = json.loads(enm_path.read_text()) if enm_path.exists() else {}

    return LoadedStore(
        model=model,
        entity_names=entity_names,
        relation_names=relation_names,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        V_high=V_high.detach(),
        R_mats=R_mats.detach(),
        triples=triples,
        enm=enm,
        store_path=store_path,
    )


# ---------------------------------------------------------------------------
# Multi-hop trace
# ---------------------------------------------------------------------------

@dataclass
class HopResult:
    """One sequential cap-aware retrieval step."""
    name: str           # nearest tail entity name
    dist: float         # geodesic distance to its v-embedding (cap_distance if cap_enabled)
    rho: float          # calibrated admissibility radius rho_r for this relation
    ratio: float        # dist / rho (lower is more confident; <=1.0 is admissible)
    verdict: str        # "admissible" | "borderline" | "inadmissible"
    rotor: str          # relation name


@dataclass
class Candidate:
    name: str
    dist: float
    ratio: float        # dist / rho_r for that hop


@dataclass
class QueryTrace:
    source: str
    relations: list[str]
    # Sequence of length L+1: source v-embedding followed by cap-center (or
    # plain rotor output for non-cap stores) after each hop. (m,)-tensors on S^{m-1}.
    hops_high: list[torch.Tensor]
    # Per-hop plausibility result.
    nearest_per_hop: list[HopResult]
    # Final retrieved entity = last hop's nearest.
    retrieved: str
    retrieved_dist: float
    retrieved_rho: float
    retrieved_ratio: float
    retrieved_verdict: str
    # Per-hop triple existence check: is (h_k, r_k, t_k) a real triple in the
    # store? Each True means the hop landed on an actual edge that exists in
    # the graph, independent of which fork the bridge labeled as ground truth.
    path_valid_per_hop: list[bool] = field(default_factory=list)
    # path_valid = all per-hop edges exist in the store
    path_valid: bool = False
    # Top-k candidates at the final hop.
    top_k: list[Candidate] = field(default_factory=list)


def _verdict_from_ratio(ratio: float) -> str:
    # ratio = dist / rho_r ; admissible iff dist <= rho ; we add a soft band
    # for borderline cases between rho and 1.25 * rho.
    if ratio <= 1.0:
        return "admissible"
    if ratio <= 1.25:
        return "borderline"
    return "inadmissible"


def run_query(store: LoadedStore, source: str, relations: list[str],
              top_k: int = 5) -> QueryTrace:
    """Sequential cap-aware multi-hop retrieval against calibrated thresholds.

    For each hop k the current head h_{k-1} (the source for k=0, the previous
    hop's chosen tail for k>0) and relation r_k define a cap center via
    ``model.cap_center(h, r)`` (or the plain rotor output for non-cap stores).
    The nearest entity by ``cap_distance`` becomes the hop's tail. Each tail
    carries its own admissibility verdict relative to the per-relation
    calibrated radius ``rho_r`` returned by ``model.cap_radii()``.
    """
    if source not in store.entity_to_idx:
        raise KeyError(f"source entity {source!r} not in store")
    bad = [r for r in relations if r not in store.relation_to_idx]
    if bad:
        raise KeyError(f"unknown relation(s): {bad}")

    model = store.model
    cap_on = bool(getattr(model, "cap_enabled", False))
    rho_all = model.cap_radii().detach() if cap_on else None

    src_idx = store.entity_to_idx[source]
    hops_high: list[torch.Tensor] = [store.V_high[src_idx].clone()]
    nearest_per_hop: list[HopResult] = []
    retrieved_chain: list[tuple[str, str, str]] = []
    top_k_candidates: list[Candidate] = []

    cur_h_idx = src_idx
    cur_h_name = source
    with torch.no_grad():
        for k, r_name in enumerate(relations):
            r_idx = store.relation_to_idx[r_name]
            h_t = torch.tensor([cur_h_idx])
            r_t = torch.tensor([r_idx])

            if cap_on:
                # cap_center returns the renormalized, scaled (Lambda_r * R_r v_h);
                # cap_distance from this center to every tail gives the per-hop
                # plausibility score, calibrated against rho_r.
                center = model.cap_center(h_t, r_t)[0]
                dists = model.score_all_tails_cap(h_t, r_t)[0]
                rho = float(rho_all[r_idx].item())
            else:
                # Non-cap fallback: geodesic distance to R_r v_h, no calibrated
                # rho. We default rho=pi/2 (the natural sphere half-radius) so
                # downstream code is still well-defined.
                v = store.V_high[cur_h_idx]
                R = store.R_mats[r_idx]
                center = R @ v
                center = center / center.norm()
                dists = model.score_all_tails(h_t, r_t)[0]
                rho = float(torch.tensor(torch.pi / 2.0).item())

            hops_high.append(center.clone())

            i_top = int(torch.argmin(dists).item())
            dist_top = float(dists[i_top].item())
            ratio = dist_top / rho if rho > 0 else float("inf")
            tail_name = store.entity_names[i_top]
            nearest_per_hop.append(HopResult(
                name=tail_name,
                dist=dist_top,
                rho=rho,
                ratio=ratio,
                verdict=_verdict_from_ratio(ratio),
                rotor=r_name,
            ))
            retrieved_chain.append((cur_h_name, r_name, tail_name))

            if k == len(relations) - 1:
                # Top-k for the last hop, sorted by distance (ascending).
                order = torch.argsort(dists)
                for i in order[:top_k]:
                    j = int(i.item())
                    d = float(dists[j].item())
                    top_k_candidates.append(Candidate(
                        name=store.entity_names[j],
                        dist=d,
                        ratio=d / rho if rho > 0 else float("inf"),
                    ))
            else:
                cur_h_idx = i_top  # advance head for the next hop
                cur_h_name = tail_name

    # Canonical path-validity check. score_path_validity lands in
    # knowlytix.benchmark.eval.scorers in GMS builds newer than rc31; fall back
    # to the equivalent triple-membership check on older wheels (compat shim
    # added during vendoring — see module header).
    triple_set = {tuple(t) for t in store.triples}
    try:
        from knowlytix.benchmark.eval.scorers import score_path_validity
        path_valid = score_path_validity(retrieved_chain, triple_set).correct
    except ImportError:
        path_valid = all(hop in triple_set for hop in retrieved_chain)
    last = nearest_per_hop[-1]
    return QueryTrace(
        source=source,
        relations=list(relations),
        hops_high=hops_high,
        nearest_per_hop=nearest_per_hop,
        retrieved=last.name,
        retrieved_dist=last.dist,
        retrieved_rho=last.rho,
        retrieved_ratio=last.ratio,
        retrieved_verdict=last.verdict,
        path_valid_per_hop=[hop in triple_set for hop in retrieved_chain],
        path_valid=path_valid,
        top_k=top_k_candidates,
    )


# ---------------------------------------------------------------------------
# Projection R^m -> S^2
# ---------------------------------------------------------------------------

def fit_projector(V_high: torch.Tensor) -> tuple[Callable[[torch.Tensor], np.ndarray], np.ndarray]:
    """Fit a 3D PCA projector and return (project_fn, V_3d_unit_sphere).

    The projector takes a (m,) or (B, m) torch tensor and returns an
    (B, 3) numpy array renormalized onto S^2.
    """
    X = V_high.detach().cpu().numpy()
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)
    Y_unit = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    def project(v: torch.Tensor) -> np.ndarray:
        arr = v.detach().cpu().numpy()
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        y = pca.transform(arr)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        return y[0] if single else y

    return project, Y_unit


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def slerp(p: np.ndarray, q: np.ndarray, n: int = 32) -> np.ndarray:
    """Spherical linear interpolation between unit vectors p and q on S^2."""
    p = p / (np.linalg.norm(p) + 1e-12)
    q = q / (np.linalg.norm(q) + 1e-12)
    dot = float(np.clip(np.dot(p, q), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        # Coincident — return tiny line at p.
        return np.tile(p, (n, 1))
    if dot < -1.0 + 1e-7:
        # Antipodal — pick an arbitrary great circle via an auxiliary axis.
        aux = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(p, aux)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ts = np.linspace(0.0, 1.0, n)
        out = np.empty((n, 3))
        for i, t in enumerate(ts):
            theta = np.pi * t
            out[i] = p * np.cos(theta) + axis * np.sin(theta)
        return out
    omega = np.arccos(dot)
    sin_o = np.sin(omega)
    ts = np.linspace(0.0, 1.0, n)
    a = np.sin((1 - ts) * omega) / sin_o
    b = np.sin(ts * omega) / sin_o
    return a[:, None] * p[None, :] + b[:, None] * q[None, :]


def _unit_sphere_mesh(resolution: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _enm_values_for_entity(enm: dict, entity_name: str, limit: int = 4) -> list[tuple[str, float]]:
    """Return up to `limit` (key, value) pairs from ENM whose key references this entity.

    Keys in cap_experiments-style stores look like
    ``"<section>::<entity>/<role>/<func>/<field>": <number>``.
    We match on the entity token to keep the lookup store-agnostic.
    """
    if not enm:
        return []
    out: list[tuple[str, float]] = []
    needle = f"::{entity_name}/"
    for k, v in enm.items():
        if not isinstance(v, (int, float)):
            continue
        if needle in k or k.endswith(f"::{entity_name}") or f"/{entity_name}/" in k:
            field = k.rsplit("/", 1)[-1]
            out.append((field, float(v)))
            if len(out) >= limit:
                break
    return out


def build_figure(
    store: LoadedStore,
    trace: QueryTrace,
    *,
    show_triples: bool = True,
    triple_alpha: float = 0.08,
    max_triple_arcs: int = 400,
    show_pointers: bool = True,
) -> go.Figure:
    project, V3 = fit_projector(store.V_high)

    fig = go.Figure()

    # Translucent unit sphere
    sx, sy, sz = _unit_sphere_mesh()
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        opacity=0.08,
        showscale=False,
        colorscale=[[0, "#888"], [1, "#888"]],
        hoverinfo="skip",
        name="S^2",
    ))

    # All entity nodes
    hover_specificity = None
    hover_confidence = None
    with torch.no_grad():
        all_idx = torch.arange(store.model.num_entities)
        try:
            hover_specificity = store.model.dual_emb.get_specificity(all_idx).cpu().numpy()
            hover_confidence = store.model.dual_emb.get_confidence(all_idx).cpu().numpy()
        except AttributeError:
            pass

    hover_text = []
    for i, name in enumerate(store.entity_names):
        bits = [f"<b>{name}</b>"]
        if hover_specificity is not None:
            bits.append(f"specificity: {hover_specificity[i]:.2f}")
        if hover_confidence is not None:
            bits.append(f"confidence: {hover_confidence[i]:.2f}")
        hover_text.append("<br>".join(bits))

    fig.add_trace(go.Scatter3d(
        x=V3[:, 0], y=V3[:, 1], z=V3[:, 2],
        mode="markers",
        marker=dict(size=2.4, color="#cbd5e1", opacity=0.45,
                    line=dict(color="#94a3b8", width=0.2)),
        text=hover_text,
        hoverinfo="text",
        name="entities (background)",
    ))

    # Background relation arcs (subset to avoid clutter)
    if show_triples and store.triples:
        ent_idx = store.entity_to_idx
        rel_to_color = _relation_palette(store.relation_names)
        arcs_by_rel: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {}
        for h, r, t in store.triples:
            if h not in ent_idx or t not in ent_idx:
                continue
            p = V3[ent_idx[h]]
            q = V3[ent_idx[t]]
            arcs_by_rel.setdefault(r, []).append((p, q, f"{h} —{r}→ {t}"))

        total = sum(len(v) for v in arcs_by_rel.values())
        if total > max_triple_arcs:
            keep = max_triple_arcs / total
        else:
            keep = 1.0
        rng = np.random.default_rng(0)

        for r_name, arcs in arcs_by_rel.items():
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for p, q, _label in arcs:
                if keep < 1.0 and rng.random() > keep:
                    continue
                pts = slerp(p, q, n=20)
                xs.extend([*pts[:, 0].tolist(), None])
                ys.extend([*pts[:, 1].tolist(), None])
                zs.extend([*pts[:, 2].tolist(), None])
            if not xs:
                continue
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=rel_to_color[r_name], width=1.5),
                opacity=triple_alpha,
                hoverinfo="skip",
                name=r_name,
                legendgroup="relations",
                showlegend=True,
            ))

    # Highlighted query path: per-hop arc + arrowhead + rotor label
    hop_points_3d = np.stack([project(v) for v in trace.hops_high], axis=0)
    n_hops = len(trace.relations)
    hop_colors = _hop_palette(n_hops)
    # Intermediate rotor-output positions (small crosses, no name label).
    inter_tip_xs, inter_tip_ys, inter_tip_zs, inter_tip_hover = [], [], [], []
    # Per-hop "nearest entity" positions at the actual entity's projected coord.
    near_xs, near_ys, near_zs, near_text, near_hover = [], [], [], [], []

    for k in range(n_hops):
        p = hop_points_3d[k]
        q = hop_points_3d[k + 1]
        pts = slerp(p, q, n=60)
        rel = trace.relations[k]
        hr = trace.nearest_per_hop[k]
        nearest_name = hr.name
        color = hop_colors[k]
        hover_label = (f"hop {k + 1}: <b>{rel}</b><br>"
                       f"tip nearest: {hr.name} "
                       f"(d={hr.dist:.3f}, \u03c1={hr.rho:.3f}, {hr.verdict})")
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color=color, width=10),
            opacity=0.95,
            name=f"hop {k + 1}: {rel}",
            hovertext=[hover_label] * pts.shape[0],
            hoverinfo="text",
            legendgroup=f"hop{k+1}",
        ))
        # Rotor label at the midpoint of the arc, large + bold for readability
        mid = pts[len(pts) // 2]
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode="text",
            text=[f"<b>R[{rel}]</b>"],
            textposition="middle center",
            textfont=dict(color=color, size=14, family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=f"hop{k+1}",
        ))
        # Arrowhead at the end of the arc
        tail_pt = pts[-3]
        head_pt = pts[-1]
        direction = head_pt - tail_pt
        nrm = float(np.linalg.norm(direction))
        if nrm > 1e-9:
            u, v_, w = (direction / nrm).tolist()
            fig.add_trace(go.Cone(
                x=[head_pt[0]], y=[head_pt[1]], z=[head_pt[2]],
                u=[u], v=[v_], w=[w],
                sizemode="absolute", sizeref=0.08, anchor="tip",
                showscale=False,
                colorscale=[[0, color], [1, color]],
                hoverinfo="skip",
                name=f"hop {k + 1} arrow",
                showlegend=False,
                legendgroup=f"hop{k+1}",
            ))
        # Intermediate transported-tip marker (the rotor output, not an entity)
        if k < n_hops - 1:
            inter_tip_xs.append(q[0])
            inter_tip_ys.append(q[1])
            inter_tip_zs.append(q[2])
            inter_tip_hover.append(
                f"<b>rotor output after hop {k+1}</b><br>"
                f"R[{rel}] · (previous tip)<br>"
                f"nearest entity: {hr.name} "
                f"(d={hr.dist:.3f}, \u03c1={hr.rho:.3f}, {hr.verdict})"
            )
        # Per-hop nearest entity at its actual sphere position, with a bold label.
        # Skip the final hop here — the retrieved entity gets its own prominent trace.
        if k < n_hops - 1 and nearest_name in store.entity_to_idx:
            anc = V3[store.entity_to_idx[nearest_name]]
            near_xs.append(anc[0])
            near_ys.append(anc[1])
            near_zs.append(anc[2])
            near_text.append(f"<b>{nearest_name}</b>")
            near_hover.append(
                f"<b>{hr.name}</b><br>"
                f"hop {k+1} (d={hr.dist:.3f}, \u03c1={hr.rho:.3f}, {hr.verdict})"
            )

    if inter_tip_xs:
        fig.add_trace(go.Scatter3d(
            x=inter_tip_xs, y=inter_tip_ys, z=inter_tip_zs,
            mode="markers",
            marker=dict(size=7, color="#10b981", symbol="cross",
                        line=dict(color="#065f46", width=1.5)),
            hovertext=inter_tip_hover,
            hoverinfo="text",
            name="rotor outputs (interior)",
        ))
    if near_xs:
        fig.add_trace(go.Scatter3d(
            x=near_xs, y=near_ys, z=near_zs,
            mode="markers+text",
            marker=dict(size=11, color="#06b6d4", symbol="circle",
                        line=dict(color="#0c4a6e", width=1.5)),
            text=near_text,
            textposition="top center",
            textfont=dict(color="#0b1020", size=13, family="Arial Black"),
            hovertext=near_hover,
            hoverinfo="text",
            name="intermediate entities",
        ))

    # Source marker
    src_xyz = hop_points_3d[0]
    fig.add_trace(go.Scatter3d(
        x=[src_xyz[0]], y=[src_xyz[1]], z=[src_xyz[2]],
        mode="markers+text",
        marker=dict(size=10, color="#fbbf24", symbol="diamond",
                    line=dict(color="#92400e", width=1.5)),
        text=[f"<b>{trace.source}</b>"],
        textposition="top center",
        textfont=dict(color="#0b1020", size=14, family="Arial Black"),
        hovertext=[f"<b>source</b>: {trace.source}"],
        hoverinfo="text",
        name="source",
    ))

    # Transported tip (final composed point)
    tip_xyz = hop_points_3d[-1]
    fig.add_trace(go.Scatter3d(
        x=[tip_xyz[0]], y=[tip_xyz[1]], z=[tip_xyz[2]],
        mode="markers",
        marker=dict(size=8, color="#10b981", symbol="cross",
                    line=dict(color="#065f46", width=1.5)),
        hovertext=[f"<b>transported tip</b><br>"
                   f"(R_{trace.relations[-1]} ... R_{trace.relations[0]})·v_src"],
        hoverinfo="text",
        name="transported tip",
    ))

    # Snap-to-nearest segment: transported tip → retrieved entity (the
    # final "find closest entity" step, drawn as a dashed geodesic).
    ret_idx = store.entity_to_idx[trace.retrieved]
    ret_xyz = V3[ret_idx]
    snap_pts = slerp(hop_points_3d[-1], ret_xyz, n=24)
    fig.add_trace(go.Scatter3d(
        x=snap_pts[:, 0], y=snap_pts[:, 1], z=snap_pts[:, 2],
        mode="lines",
        line=dict(color="#ef4444", width=6, dash="dash"),
        opacity=0.85,
        name="snap → nearest",
        hovertext=[f"snap to nearest entity: <b>{trace.retrieved}</b><br>"
                   f"d={trace.retrieved_dist:.3f}, \u03c1={trace.retrieved_rho:.3f}, "
                   f"{trace.retrieved_verdict}"] * snap_pts.shape[0],
        hoverinfo="text",
    ))
    # Arrowhead on the snap segment
    tail = snap_pts[-3]
    head = snap_pts[-1]
    direction = head - tail
    nrm = float(np.linalg.norm(direction))
    if nrm > 1e-9:
        u, v_, w = (direction / nrm).tolist()
        fig.add_trace(go.Cone(
            x=[head[0]], y=[head[1]], z=[head[2]],
            u=[u], v=[v_], w=[w],
            sizemode="absolute", sizeref=0.08, anchor="tip",
            showscale=False,
            colorscale=[[0, "#ef4444"], [1, "#ef4444"]],
            hoverinfo="skip", showlegend=False, name="snap arrow",
        ))

    # Retrieved entity
    fig.add_trace(go.Scatter3d(
        x=[ret_xyz[0]], y=[ret_xyz[1]], z=[ret_xyz[2]],
        mode="markers+text",
        marker=dict(size=14, color="rgba(239,68,68,0)",
                    line=dict(color="#ef4444", width=4),
                    symbol="circle-open"),
        text=[f"<b>{trace.retrieved}</b>"],
        textposition="bottom center",
        textfont=dict(color="#ef4444", size=14, family="Arial Black"),
        hovertext=[f"<b>retrieved</b>: {trace.retrieved}<br>"
                   f"d={trace.retrieved_dist:.3f}, \u03c1={trace.retrieved_rho:.3f}, "
                   f"{trace.retrieved_verdict}"],
        hoverinfo="text",
        name="retrieved",
    ))

    # Top-k candidate ring (smaller markers)
    if len(trace.top_k) > 1:
        xs, ys, zs, txt = [], [], [], []
        for cand in trace.top_k[1:]:
            if cand.name not in store.entity_to_idx:
                continue
            xyz = V3[store.entity_to_idx[cand.name]]
            xs.append(xyz[0])
            ys.append(xyz[1])
            zs.append(xyz[2])
            txt.append(f"{cand.name}<br>d={cand.dist:.3f}, d/\u03c1={cand.ratio:.2f}")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=6, color="rgba(239,68,68,0.35)",
                        line=dict(color="#ef4444", width=1)),
            hovertext=txt,
            hoverinfo="text",
            name="top-k candidates",
        ))

    # Off-sphere pointer nodes for exact numeric payloads (ENM).
    if show_pointers and store.enm:
        anchors_of_interest = {trace.source, trace.retrieved}
        for hr in trace.nearest_per_hop:
            anchors_of_interest.add(hr.name)
        for cand in trace.top_k:
            anchors_of_interest.add(cand.name)

        ptr_xs, ptr_ys, ptr_zs, ptr_text = [], [], [], []
        link_xs, link_ys, link_zs = [], [], []
        for ent_name in anchors_of_interest:
            if ent_name not in store.entity_to_idx:
                continue
            payloads = _enm_values_for_entity(store.enm, ent_name)
            if not payloads:
                continue
            anchor = V3[store.entity_to_idx[ent_name]]
            for i, (field, value) in enumerate(payloads):
                # Off-sphere position: along the entity's radial direction
                # at radius 1.18 + a small fan to spread multiple payloads.
                fan = 0.06 * i
                radial = anchor * (1.18 + fan)
                ptr_xs.append(radial[0])
                ptr_ys.append(radial[1])
                ptr_zs.append(radial[2])
                ptr_text.append(f"<b>{ent_name}</b>.{field} = {value:g}")
                # Dashed link from sphere anchor to payload
                link_xs += [anchor[0], radial[0], None]
                link_ys += [anchor[1], radial[1], None]
                link_zs += [anchor[2], radial[2], None]

        if ptr_xs:
            fig.add_trace(go.Scatter3d(
                x=link_xs, y=link_ys, z=link_zs,
                mode="lines",
                line=dict(color="#94a3b8", width=2, dash="dot"),
                hoverinfo="skip",
                name="pointer links",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=ptr_xs, y=ptr_ys, z=ptr_zs,
                mode="markers",
                marker=dict(size=5, color="#0f172a", symbol="square",
                            line=dict(color="#94a3b8", width=1)),
                hovertext=ptr_text,
                hoverinfo="text",
                name="numeric payloads",
            ))

    title = _format_title(trace)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            bgcolor="#fbfbfd",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#fbfbfd",
    )
    return fig


# ---------------------------------------------------------------------------
# Palette + title helpers
# ---------------------------------------------------------------------------

_RELATION_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _relation_palette(relation_names: list[str]) -> dict[str, str]:
    return {name: _RELATION_PALETTE[i % len(_RELATION_PALETTE)]
            for i, name in enumerate(relation_names)}


def _hop_palette(n_hops: int) -> list[str]:
    base = ["#f97316", "#a855f7", "#0ea5e9", "#22c55e", "#eab308"]
    return [base[i % len(base)] for i in range(max(n_hops, 1))]


def _format_title(trace: QueryTrace) -> str:
    chain = " → ".join([trace.source] + [f"[{r}]" for r in trace.relations] + ["?"])
    return (f"<b>{chain}</b><br>"
            f"<span style='font-size:0.85em'>retrieved: <b>{trace.retrieved}</b> "
            f"(d={trace.retrieved_dist:.3f}, \u03c1={trace.retrieved_rho:.3f}, "
            f"{trace.retrieved_verdict})</span>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a 3D sphere visualization of a GMS multi-hop query.")
    p.add_argument("--store", required=True, help="Path to a saved GMS store directory.")
    p.add_argument("--source", required=True, help="Source entity name.")
    p.add_argument("--relations", required=True,
                   help="Comma- or arrow-separated relation chain, e.g. 'has_role,escalates_to'.")
    p.add_argument("--out", default="gms_sphere_demo.html", help="Output HTML file.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-triples", action="store_true", help="Hide background triple arcs.")
    p.add_argument("--max-triple-arcs", type=int, default=400)
    p.add_argument("--triple-alpha", type=float, default=0.08)
    p.add_argument("--no-pointers", action="store_true",
                   help="Hide off-sphere numeric-payload pointer nodes.")
    args = p.parse_args(argv)

    rels = [r.strip() for r in args.relations.replace("->", ",").split(",") if r.strip()]
    if not rels:
        p.error("--relations must contain at least one relation")

    store = load_store(args.store)
    trace = run_query(store, args.source, rels, top_k=args.top_k)
    fig = build_figure(
        store, trace,
        show_triples=not args.no_triples,
        triple_alpha=args.triple_alpha,
        max_triple_arcs=args.max_triple_arcs,
        show_pointers=not args.no_pointers,
    )
    out_path = Path(args.out)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out_path}  (entities={len(store.entity_names)}, "
          f"hops={len(trace.relations)}, retrieved={trace.retrieved!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
