# app.py
# DRBS Simulator (Streamlit)
# Baseline (Static: length-only) vs DRBS (Risk-aware: hybrid cost)
# + Fair comparison: Baseline chosen without risk, but risk exposure is measured on the same scenario risk map
# + Experiment Log: store multiple scenarios and export as CSV
#
# Run:
#   streamlit run app.py
#
# Install:
#   pip install streamlit osmnx networkx folium streamlit-folium pandas numpy geopandas shapely

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import streamlit as st

import networkx as nx
import osmnx as ox
import folium
from streamlit_folium import st_folium


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="DRBS Simulator (Baseline vs DRBS)",
    layout="wide",
)

ox.settings.log_console = False
ox.settings.use_cache = True

# -----------------------------
# Constants
# -----------------------------
PLACE_NAME = "Bağcılar, Istanbul, Turkey"


# -----------------------------
# Data classes
# -----------------------------
@dataclass(frozen=True)
class ScenarioParams:
    structural_pct: float  # 0..1
    dynamic_pct: float     # 0..1
    intensity: float       # 0..1
    alpha: float           # >=0
    beta: float            # >=0
    gamma: float           # >=0

    @property
    def key(self) -> str:
        s = (
            f"{self.structural_pct:.4f}|{self.dynamic_pct:.4f}|{self.intensity:.4f}|"
            f"{self.alpha:.4f}|{self.beta:.4f}|{self.gamma:.4f}"
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class RouteResult:
    name: str
    path_nodes: List[int]
    distance_m: float
    eta_min: float
    sum_d_risk: float
    sum_v_risk: float
    safety_cost: float
    total_cost_length_m: float   # always meters (sum of edge lengths)
    total_cost_hybrid: float     # always unitless (sum of cost_drbs)


# -----------------------------
# Deterministic, repeatable risk
# -----------------------------
def _hash_to_unit_float(seed: str) -> float:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    x = int(h[:16], 16)
    return (x % (10**12)) / float(10**12)


def edge_risk_deterministic(
    u: int, v: int, k: int,
    structural_pct: float,
    dynamic_pct: float,
    intensity: float
) -> Tuple[float, float]:
    sel_d = _hash_to_unit_float(f"SEL_D|{u}|{v}|{k}")
    sel_v = _hash_to_unit_float(f"SEL_V|{u}|{v}|{k}")

    mag_d = _hash_to_unit_float(f"MAG_D|{u}|{v}|{k}")
    mag_v = _hash_to_unit_float(f"MAG_V|{u}|{v}|{k}")

    d = 0.0
    v_ = 0.0

    if sel_d < structural_pct:
        d = intensity * (0.4 + 0.6 * mag_d)

    if sel_v < dynamic_pct:
        v_ = intensity * (0.4 + 0.6 * mag_v)

    d = float(np.clip(d, 0.0, 1.0))
    v_ = float(np.clip(v_, 0.0, 1.0))
    return d, v_


# -----------------------------
# Graph loading & preparation
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_base_graph(place_name: str) -> nx.MultiDiGraph:
    G = ox.graph_from_place(place_name, network_type="drive", simplify=True)

    # Ensure lengths
    if not all("length" in data for _, _, _, data in G.edges(keys=True, data=True)):
        G = ox.distance.add_edge_lengths(G)

    # Speeds + travel times
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:
        for u, v, k, data in G.edges(keys=True, data=True):
            data["speed_kph"] = float(data.get("speed_kph", 30.0) or 30.0)
            length_m = float(data.get("length", 0.0) or 0.0)
            speed_mps = (data["speed_kph"] * 1000.0) / 3600.0
            data["travel_time"] = length_m / max(speed_mps, 0.1)

    for _, _, _, data in G.edges(keys=True, data=True):
        data["base_length"] = float(data.get("length", 0.0) or 0.0)

    return G


# Avoid hashing MultiDiGraph in Streamlit cache
@st.cache_resource(show_spinner=False)
def load_edges_gdf_for_viz(_G: nx.MultiDiGraph):
    return ox.graph_to_gdfs(_G, nodes=False, edges=True, fill_edge_geometry=True)


@st.cache_resource(show_spinner=False)
def load_place_bounds(place_name: str) -> List[Tuple[float, float]]:
    """
    Returns bounds as [(south, west), (north, east)] based on the place polygon (Bağcılar admin boundary).
    If any issue occurs, caller can fallback to graph bounds.
    """
    gdf = ox.geocode_to_gdf(place_name)
    minx, miny, maxx, maxy = gdf.total_bounds  # lon_min, lat_min, lon_max, lat_max
    return [(float(miny), float(minx)), (float(maxy), float(maxx))]


def compute_length_reference(G: nx.MultiDiGraph) -> float:
    lengths = [float(data.get("base_length", data.get("length", 0.0)) or 0.0)
               for _, _, _, data in G.edges(keys=True, data=True)]
    return float(np.max(lengths)) if lengths else 1.0


def prepare_scenario_graph(
    G_base: nx.MultiDiGraph,
    params: ScenarioParams,
    length_ref_m: float
) -> nx.MultiDiGraph:
    # Build fresh graph
    G = nx.MultiDiGraph()
    G.graph.update(G_base.graph)

    for n, nd in G_base.nodes(data=True):
        G.add_node(n, **dict(nd))

    for u, v, k, ed in G_base.edges(keys=True, data=True):
        data = dict(ed)

        d_risk, v_risk = edge_risk_deterministic(
            u, v, k,
            structural_pct=params.structural_pct,
            dynamic_pct=params.dynamic_pct,
            intensity=params.intensity
        )
        data["d_risk"] = d_risk
        data["v_risk"] = v_risk
        data["risk01"] = float(np.clip((d_risk + v_risk) / 2.0, 0.0, 1.0))

        length_m = float(data.get("base_length", data.get("length", 0.0)) or 0.0)
        Lhat = length_m / max(length_ref_m, 1.0)
        data["Lhat"] = float(Lhat)

        data["cost_drbs"] = float(params.alpha * Lhat + params.beta * d_risk + params.gamma * v_risk)

        G.add_edge(u, v, key=k, **data)

    return G


# -----------------------------
# Nearest node without scikit-learn
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_node_xy_arrays(_G: nx.MultiDiGraph):
    node_ids = np.array(list(_G.nodes), dtype=np.int64)
    xs = np.array([_G.nodes[n]["x"] for n in node_ids], dtype=np.float64)  # lon
    ys = np.array([_G.nodes[n]["y"] for n in node_ids], dtype=np.float64)  # lat
    return node_ids, xs, ys


def nearest_node_haversine(lat: float, lon: float, node_ids: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> int:
    R = 6371000.0
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    lat2 = np.deg2rad(ys)
    lon2 = np.deg2rad(xs)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist = R * c

    idx = int(np.argmin(dist))
    return int(node_ids[idx])


# -----------------------------
# MultiDiGraph helpers
# -----------------------------
def best_edge_key_and_data(G: nx.MultiDiGraph, u: int, v: int, weight_attr: str) -> Tuple[int, Dict[str, Any]]:
    candidates = G.get_edge_data(u, v)
    if not candidates:
        raise KeyError(f"No edge data between {u} and {v}")

    best_k, best_data, best_w = None, None, float("inf")
    for k, data in candidates.items():
        w = float(data.get(weight_attr, float("inf")))
        if w < best_w:
            best_w = w
            best_k = k
            best_data = data

    if best_k is None or best_data is None:
        raise ValueError(f"Could not pick best edge for {u}->{v} using '{weight_attr}'")

    return best_k, best_data


def compute_path(G: nx.MultiDiGraph, start_node: int, target_node: int, weight_attr: str) -> List[int]:
    try:
        return nx.shortest_path(G, start_node, target_node, weight=weight_attr)
    except nx.NetworkXNoPath as e:
        raise nx.NetworkXNoPath(f"No path found from Start to Target using '{weight_attr}'.") from e


def compute_metrics_along_path(
    G_metrics: nx.MultiDiGraph,
    path_nodes: List[int],
    name: str,
    route_edge_choice: str,  # "length" or "cost_drbs" to choose which parallel edge is assumed taken
    beta: float,
    gamma: float
) -> RouteResult:
    distance_m = 0.0
    eta_s = 0.0
    sum_d = 0.0
    sum_v = 0.0
    total_cost_len = 0.0
    total_cost_hyb = 0.0

    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        _, data = best_edge_key_and_data(G_metrics, a, b, weight_attr=route_edge_choice)

        length_m = float(data.get("base_length", data.get("length", 0.0)) or 0.0)
        travel_time_s = float(data.get("travel_time", 0.0) or 0.0)
        d = float(data.get("d_risk", 0.0) or 0.0)
        v = float(data.get("v_risk", 0.0) or 0.0)
        cost_drbs = float(data.get("cost_drbs", 0.0) or 0.0)

        distance_m += length_m
        eta_s += travel_time_s
        sum_d += d
        sum_v += v

        total_cost_len += length_m
        total_cost_hyb += cost_drbs

    eta_min = eta_s / 60.0
    safety_cost = beta * sum_d + gamma * sum_v

    return RouteResult(
        name=name,
        path_nodes=path_nodes,
        distance_m=float(distance_m),
        eta_min=float(eta_min),
        sum_d_risk=float(sum_d),
        sum_v_risk=float(sum_v),
        safety_cost=float(safety_cost),
        total_cost_length_m=float(total_cost_len),
        total_cost_hybrid=float(total_cost_hyb),
    )


# -----------------------------
# Visualization helpers
# -----------------------------
def risk_color(risk01: float) -> str:
    r = float(np.clip(risk01, 0.0, 1.0))
    if r < 0.33:
        return "#2ECC71"
    elif r < 0.66:
        return "#F39C12"
    else:
        return "#E74C3C"


def build_base_map(bounds: Optional[List[Tuple[float, float]]] = None) -> folium.Map:
    """
    If bounds provided, map will open covering the whole Bağcılar area (fit_bounds).
    bounds format: [(south, west), (north, east)]
    """
    if bounds is not None:
        (south, west), (north, east) = bounds
        center = ((south + north) / 2.0, (west + east) / 2.0)
        m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles="cartodbpositron")
        m.fit_bounds(bounds, padding=(10, 10))
        return m

    # fallback
    return folium.Map(location=(41.0395, 28.8563), zoom_start=13, control_scale=True, tiles="cartodbpositron")


def add_risk_overlay(m: folium.Map, edges_geojson: str) -> None:
    def style_function(feature):
        r = feature.get("properties", {}).get("risk01", 0.0)
        return {"color": risk_color(float(r)), "weight": 3, "opacity": 0.65}

    folium.GeoJson(edges_geojson, name="Risk Overlay", style_function=style_function, control=False).add_to(m)


def add_start_target_markers(
    m: folium.Map,
    start_latlon: Optional[Tuple[float, float]],
    target_latlon: Optional[Tuple[float, float]]
) -> None:
    if start_latlon is not None:
        folium.Marker(
            location=start_latlon,
            popup="Start",
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
    if target_latlon is not None:
        folium.Marker(
            location=target_latlon,
            popup="Target",
            tooltip="Target",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(m)


def add_route_to_map(m: folium.Map, G: nx.MultiDiGraph, path_nodes: List[int], color: str, name: str) -> None:
    try:
        ox.plot_route_folium(G, path_nodes, route_map=m, color=color, weight=6, opacity=0.9, popup_attribute=name)
    except Exception:
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
        folium.PolyLine(coords, color=color, weight=6, opacity=0.9, tooltip=name).add_to(m)


def make_metrics_dataframe(baseline: Optional[RouteResult], drbs: Optional[RouteResult]) -> pd.DataFrame:
    rows = []
    for r in [baseline, drbs]:
        if r is None:
            continue
        rows.append({
            "Method": r.name,
            "Total Distance (m)": round(r.distance_m, 2),
            "ETA (min)": round(r.eta_min, 2),
            "Σ Structural Risk (∑d)": round(r.sum_d_risk, 4),
            "Σ Dynamic Risk (∑v)": round(r.sum_v_risk, 4),
            "Safety Cost (β∑d + γ∑v)": round(r.safety_cost, 4),
            "Total Cost (length, m)": round(r.total_cost_length_m, 2),
            "Total Cost (hybrid)": round(r.total_cost_hybrid, 4),
        })

    df = pd.DataFrame(rows)
    if len(df) == 2:
        b = df.iloc[0]
        d = df.iloc[1]
        delta = {
            "Method": "Δ (DRBS - Baseline)",
            "Total Distance (m)": round(float(d["Total Distance (m)"] - b["Total Distance (m)"]), 2),
            "ETA (min)": round(float(d["ETA (min)"] - b["ETA (min)"]), 2),
            "Σ Structural Risk (∑d)": round(float(d["Σ Structural Risk (∑d)"] - b["Σ Structural Risk (∑d)"]), 4),
            "Σ Dynamic Risk (∑v)": round(float(d["Σ Dynamic Risk (∑v)"] - b["Σ Dynamic Risk (∑v)"]), 4),
            "Safety Cost (β∑d + γ∑v)": round(float(d["Safety Cost (β∑d + γ∑v)"] - b["Safety Cost (β∑d + γ∑v)"]), 4),
            "Total Cost (length, m)": round(float(d["Total Cost (length, m)"] - b["Total Cost (length, m)"]), 2),
            "Total Cost (hybrid)": round(float(d["Total Cost (hybrid)"] - b["Total Cost (hybrid)"]), 4),
        }
        df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)

    return df


def scenario_edges_geojson(edges_gdf_base: "pd.DataFrame", G_scenario: nx.MultiDiGraph) -> str:
    gdf = edges_gdf_base.copy()
    risk_vals = []
    for (u, v, k) in gdf.index:
        data = G_scenario.get_edge_data(u, v, k)
        if data is None:
            risk_vals.append(0.0)
        else:
            risk_vals.append(float(data.get("risk01", 0.0) or 0.0))
    gdf["risk01"] = risk_vals
    keep = gdf[["geometry", "risk01"]].copy()
    return keep.to_json()


# -----------------------------
# Session state init
# -----------------------------
def ss_init():
    defaults = {
        "start_latlon": None,
        "target_latlon": None,
        "start_node": None,
        "target_node": None,
        "selection_mode": "Select Start",
        "lock_st": False,

        "scenario_graph": None,
        "scenario_key": None,

        "baseline_result": None,
        "drbs_result": None,
        "last_error": None,

        "length_ref_m": None,

        "experiment_log": [],
        "scenario_label": "S1",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ss_init()

# -----------------------------
# Load base graph + cached arrays + bounds
# -----------------------------
with st.spinner("Loading OSM road network (cached after first run)..."):
    G_base = load_base_graph(PLACE_NAME)
    edges_gdf_base = load_edges_gdf_for_viz(G_base)
    node_ids_arr, xs_arr, ys_arr = get_node_xy_arrays(G_base)

    if st.session_state["length_ref_m"] is None:
        st.session_state["length_ref_m"] = compute_length_reference(G_base)

length_ref_m = st.session_state["length_ref_m"]

# Map bounds: prefer Bağcılar admin boundary; fallback to graph bounds
try:
    MAP_BOUNDS = load_place_bounds(PLACE_NAME)  # [(south, west), (north, east)]
except Exception:
    minx, miny, maxx, maxy = edges_gdf_base.total_bounds
    MAP_BOUNDS = [(float(miny), float(minx)), (float(maxy), float(maxx))]


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("Start / Target Selection")
st.session_state["selection_mode"] = st.sidebar.radio(
    "Click on the map to set:",
    options=["Select Start", "Select Target"],
    index=0 if st.session_state["selection_mode"] == "Select Start" else 1,
    disabled=st.session_state["lock_st"],
)

st.session_state["lock_st"] = st.sidebar.checkbox(
    "Lock Start/Target (prevent accidental changes)",
    value=bool(st.session_state["lock_st"]),
)

col_reset1, col_reset2 = st.sidebar.columns(2)
with col_reset1:
    if st.button("Reset Start/Target", use_container_width=True):
        st.session_state["start_latlon"] = None
        st.session_state["target_latlon"] = None
        st.session_state["start_node"] = None
        st.session_state["target_node"] = None
        st.session_state["baseline_result"] = None
        st.session_state["drbs_result"] = None

with col_reset2:
    if st.button("Clear Current Results", use_container_width=True):
        st.session_state["baseline_result"] = None
        st.session_state["drbs_result"] = None
        st.session_state["last_error"] = None

st.sidebar.divider()
st.sidebar.subheader("Scenario Parameters (Risk Generation)")

structural_pct = st.sidebar.slider("Structural risk edge ratio (structural_pct)", 0.0, 1.0, 0.25, 0.01)
dynamic_pct = st.sidebar.slider("Dynamic risk edge ratio (dynamic_pct)", 0.0, 1.0, 0.20, 0.01)
intensity = st.sidebar.slider("Risk intensity (intensity)", 0.0, 1.0, 0.70, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Cost Weights (DRBS)")
alpha = st.sidebar.number_input("α (normalized length weight)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
beta = st.sidebar.number_input("β (structural risk weight)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
gamma = st.sidebar.number_input("γ (dynamic risk weight)", min_value=0.0, value=1.0, step=0.1, format="%.2f")

params = ScenarioParams(
    structural_pct=float(structural_pct),
    dynamic_pct=float(dynamic_pct),
    intensity=float(intensity),
    alpha=float(alpha),
    beta=float(beta),
    gamma=float(gamma),
)

# Rebuild scenario graph only when params change
if st.session_state["scenario_key"] != params.key:
    with st.spinner("Applying scenario parameters (risk + DRBS edge costs)..."):
        st.session_state["scenario_graph"] = prepare_scenario_graph(G_base, params, length_ref_m)
        st.session_state["scenario_key"] = params.key

    # prevent mixing results across scenarios
    st.session_state["baseline_result"] = None
    st.session_state["drbs_result"] = None
    st.session_state["last_error"] = None

G_scn: nx.MultiDiGraph = st.session_state["scenario_graph"]

st.sidebar.divider()
st.sidebar.subheader("Experiment Log (Multi-Scenario)")
st.session_state["scenario_label"] = st.sidebar.text_input(
    "Scenario label (saved row)",
    value=st.session_state.get("scenario_label", "S1")
)

col_log1, col_log2 = st.sidebar.columns(2)
with col_log1:
    clear_log = st.button("Clear Log", use_container_width=True)

if clear_log:
    st.session_state["experiment_log"] = []

st.sidebar.caption(
    "Tip: Fix Start–Target once. For each scenario, compute Baseline + DRBS, then save it to the log."
)

st.sidebar.divider()
st.sidebar.subheader("Map color note")
st.sidebar.caption(
    "Road colors depend only on risk01 = (d_risk + v_risk)/2. "
    "They do NOT change when you tune β/γ."
)


# -----------------------------
# Map click handler
# -----------------------------
def handle_map_click(map_state: dict):
    if st.session_state["lock_st"]:
        return
    if not map_state:
        return
    last_clicked = map_state.get("last_clicked")
    if not last_clicked:
        return

    lat = float(last_clicked["lat"])
    lon = float(last_clicked["lng"])

    if st.session_state["selection_mode"] == "Select Start":
        st.session_state["start_latlon"] = (lat, lon)
        st.session_state["start_node"] = nearest_node_haversine(lat, lon, node_ids_arr, xs_arr, ys_arr)
    else:
        st.session_state["target_latlon"] = (lat, lon)
        st.session_state["target_node"] = nearest_node_haversine(lat, lon, node_ids_arr, xs_arr, ys_arr)


def require_start_target() -> bool:
    if st.session_state["start_node"] is None or st.session_state["target_node"] is None:
        st.warning("Please select both Start and Target on the map (use the sidebar).")
        return False
    return True


def add_current_run_to_log():
    b = st.session_state["baseline_result"]
    d = st.session_state["drbs_result"]
    if b is None or d is None:
        st.warning("Compute BOTH Baseline and DRBS first, then add to log.")
        return

    row = {
        "timestamp_unix": int(time.time()),
        "scenario": st.session_state["scenario_label"],
        "place": PLACE_NAME,
        "start_latlon": st.session_state["start_latlon"],
        "target_latlon": st.session_state["target_latlon"],
        "start_node": st.session_state["start_node"],
        "target_node": st.session_state["target_node"],
        "params.structural_pct": params.structural_pct,
        "params.dynamic_pct": params.dynamic_pct,
        "params.intensity": params.intensity,
        "params.alpha": params.alpha,
        "params.beta": params.beta,
        "params.gamma": params.gamma,
        # Baseline
        "baseline_distance_m": b.distance_m,
        "baseline_eta_min": b.eta_min,
        "baseline_sum_d": b.sum_d_risk,
        "baseline_sum_v": b.sum_v_risk,
        "baseline_safety_cost": b.safety_cost,
        "baseline_total_cost_length_m": b.total_cost_length_m,
        "baseline_total_cost_hybrid": b.total_cost_hybrid,
        # DRBS
        "drbs_distance_m": d.distance_m,
        "drbs_eta_min": d.eta_min,
        "drbs_sum_d": d.sum_d_risk,
        "drbs_sum_v": d.sum_v_risk,
        "drbs_safety_cost": d.safety_cost,
        "drbs_total_cost_length_m": d.total_cost_length_m,
        "drbs_total_cost_hybrid": d.total_cost_hybrid,
        # Deltas
        "delta_distance_m": d.distance_m - b.distance_m,
        "delta_eta_min": d.eta_min - b.eta_min,
        "delta_sum_d": d.sum_d_risk - b.sum_d_risk,
        "delta_sum_v": d.sum_v_risk - b.sum_v_risk,
        "delta_safety_cost": d.safety_cost - b.safety_cost,
        "delta_total_cost_length_m": d.total_cost_length_m - b.total_cost_length_m,
        "delta_total_cost_hybrid": d.total_cost_hybrid - b.total_cost_hybrid,
    }

    st.session_state["experiment_log"].append(row)

    # auto increment label (S1 -> S2 ...)
    try:
        if st.session_state["scenario_label"].startswith("S"):
            n = int(st.session_state["scenario_label"][1:])
            st.session_state["scenario_label"] = f"S{n+1}"
    except Exception:
        pass


# -----------------------------
# Main UI
# -----------------------------
st.title("DRBS Simulator – Baseline (Static) vs DRBS (Risk-Aware)")
st.caption(
    "This simulator uses a fixed OSM road network (Bağcılar) and a controlled Start–Target pair. "
    "Baseline ignores risk during routing; DRBS uses a hybrid risk-aware cost."
)

tab1, tab2, tab3 = st.tabs(["1) Baseline (Static)", "2) DRBS (Risk-Aware)", "3) Compare + Scenarios"])


# -----------------------------
# TAB 1: Baseline
# -----------------------------
with tab1:
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.subheader("Map (Click to select Start/Target)")

        # IMPORTANT: Open map covering whole Bağcılar
        m = build_base_map(bounds=MAP_BOUNDS)

        add_start_target_markers(m, st.session_state["start_latlon"], st.session_state["target_latlon"])

        if st.session_state["baseline_result"] is not None:
            add_route_to_map(m, G_base, st.session_state["baseline_result"].path_nodes, color="#1f77b4", name="Baseline Route")

        map_state = st_folium(m, height=580, width=None, key="map_baseline")
        handle_map_click(map_state)

    with right:
        st.subheader("Baseline (Static) – Run")
        st.write("**Baseline definition:** shortest path using only edge length (risk is NOT used for routing).")
        st.caption("For fair comparison, Baseline risk exposure (Σd, Σv, hybrid cost) is measured on the SAME scenario risk map.")

        run_baseline = st.button("Compute Baseline Route", type="primary", use_container_width=True)

        if run_baseline:
            st.session_state["last_error"] = None
            if require_start_target():
                with st.spinner("Computing Baseline shortest path (length-only)..."):
                    try:
                        # 1) Choose path WITHOUT risk (length-only) on base graph
                        path_nodes = compute_path(
                            G_base,
                            st.session_state["start_node"],
                            st.session_state["target_node"],
                            weight_attr="length",
                        )
                        # 2) Measure risk exposure ON scenario graph (same risks as DRBS)
                        res = compute_metrics_along_path(
                            G_metrics=G_scn,
                            path_nodes=path_nodes,
                            name="Baseline",
                            route_edge_choice="length",
                            beta=params.beta,
                            gamma=params.gamma,
                        )
                        st.session_state["baseline_result"] = res
                    except Exception as e:
                        st.session_state["last_error"] = str(e)

        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])

        if st.session_state["baseline_result"] is not None:
            r = st.session_state["baseline_result"]
            st.success("Baseline route computed.")
            st.metric("Total Distance (m)", f"{r.distance_m:.1f}")
            st.metric("ETA (min)", f"{r.eta_min:.2f}")
            st.metric("Σ Structural Risk (∑d)", f"{r.sum_d_risk:.4f}")
            st.metric("Σ Dynamic Risk (∑v)", f"{r.sum_v_risk:.4f}")
            st.metric("Safety Cost (β∑d + γ∑v)", f"{r.safety_cost:.4f}")
            st.metric("Total Cost (length, m)", f"{r.total_cost_length_m:.1f}")
            st.metric("Total Cost (hybrid)", f"{r.total_cost_hybrid:.4f}")
        else:
            st.caption("No Baseline result yet.")


# -----------------------------
# TAB 2: DRBS
# -----------------------------
with tab2:
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.subheader("Map (Risk Overlay + DRBS Route)")

        # IMPORTANT: Open map covering whole Bağcılar
        m = build_base_map(bounds=MAP_BOUNDS)

        try:
            edges_json = scenario_edges_geojson(edges_gdf_base, G_scn)
            add_risk_overlay(m, edges_json)
        except Exception as e:
            st.warning(f"Risk overlay could not be rendered (routes still work). Details: {e}")

        add_start_target_markers(m, st.session_state["start_latlon"], st.session_state["target_latlon"])

        if st.session_state["drbs_result"] is not None:
            add_route_to_map(m, G_scn, st.session_state["drbs_result"].path_nodes, color="#8e44ad", name="DRBS Route")

        map_state = st_folium(m, height=580, width=None, key="map_drbs")
        handle_map_click(map_state)

    with right:
        st.subheader("DRBS (Risk-Aware) – Run")
        st.write("**DRBS definition:** shortest path over hybrid edge cost:")
        st.code("cost_drbs = α·Lhat + β·d_risk + γ·v_risk", language="text")
        st.caption(f"Length normalization uses Lhat = length / max_edge_length, max_edge_length = {length_ref_m:.1f} m.")

        run_drbs = st.button("Compute DRBS Route", type="primary", use_container_width=True)

        if run_drbs:
            st.session_state["last_error"] = None
            if require_start_target():
                with st.spinner("Computing DRBS shortest path (hybrid cost)..."):
                    try:
                        path_nodes = compute_path(
                            G_scn,
                            st.session_state["start_node"],
                            st.session_state["target_node"],
                            weight_attr="cost_drbs",
                        )
                        res = compute_metrics_along_path(
                            G_metrics=G_scn,
                            path_nodes=path_nodes,
                            name="DRBS",
                            route_edge_choice="cost_drbs",
                            beta=params.beta,
                            gamma=params.gamma,
                        )
                        st.session_state["drbs_result"] = res
                    except Exception as e:
                        st.session_state["last_error"] = str(e)

        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])

        if st.session_state["drbs_result"] is not None:
            r = st.session_state["drbs_result"]
            st.success("DRBS route computed.")
            st.metric("Total Distance (m)", f"{r.distance_m:.1f}")
            st.metric("ETA (min)", f"{r.eta_min:.2f}")
            st.metric("Σ Structural Risk (∑d)", f"{r.sum_d_risk:.4f}")
            st.metric("Σ Dynamic Risk (∑v)", f"{r.sum_v_risk:.4f}")
            st.metric("Safety Cost (β∑d + γ∑v)", f"{r.safety_cost:.4f}")
            st.metric("Total Cost (length, m)", f"{r.total_cost_length_m:.1f}")
            st.metric("Total Cost (hybrid)", f"{r.total_cost_hybrid:.4f}")
        else:
            st.caption("No DRBS result yet.")


# -----------------------------
# TAB 3: Compare + Scenarios
# -----------------------------
with tab3:
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.subheader("Compare Map (Baseline + DRBS on the Same Map)")

        # IMPORTANT: Open map covering whole Bağcılar
        m = build_base_map(bounds=MAP_BOUNDS)

        try:
            edges_json = scenario_edges_geojson(edges_gdf_base, G_scn)
            add_risk_overlay(m, edges_json)
        except Exception:
            pass

        add_start_target_markers(m, st.session_state["start_latlon"], st.session_state["target_latlon"])

        if st.session_state["baseline_result"] is not None:
            add_route_to_map(m, G_base, st.session_state["baseline_result"].path_nodes, color="#1f77b4", name="Baseline Route")
        if st.session_state["drbs_result"] is not None:
            add_route_to_map(m, G_scn, st.session_state["drbs_result"].path_nodes, color="#8e44ad", name="DRBS Route")

        map_state = st_folium(m, height=580, width=None, key="map_compare")
        handle_map_click(map_state)

    with right:
        st.subheader("Metrics Table (Baseline vs DRBS)")
        df = make_metrics_dataframe(st.session_state["baseline_result"], st.session_state["drbs_result"])
        if df.empty:
            st.info("Compute both Baseline and DRBS routes to see the comparison table.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Multi-Scenario Comparison (Experiment Log)")

        add_to_log = st.button("Add current (Baseline + DRBS) to Scenario Log", use_container_width=True)
        if add_to_log:
            add_current_run_to_log()

        if st.session_state["experiment_log"]:
            log_df = pd.DataFrame(st.session_state["experiment_log"])
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            csv_bytes = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Scenario Log CSV",
                data=csv_bytes,
                file_name="drbs_experiment_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Scenario log is empty. Save runs here to compare multiple scenarios.")


st.markdown("---")
st.caption(
    "PoC note: Risk values are deterministic (hash-based) for repeatability. "
    "Baseline uses length-only routing; DRBS uses the hybrid cost. "
    "For fair comparison, Baseline risk exposure is measured on the same scenario risk map."
)
