#!/usr/bin/env python3
"""Build and inspect DK, FI, NO and SE with local PyPSA-Earth defaults.

Place beside Snakefile and run in the checkout's conda/pixi environment:
    pixi run python nordic_networks.py
    pixi run python nordic_networks.py --skip-build       # redraw/recalculate only
    pixi run python nordic_networks.py --dry-run          # inspect the build jobs

Extra plotting dependency: folium. Other dependencies are from PyPSA-Earth.
Outputs: networks/{country}/base.nc; results/{country}/network_inspection/;
results/NORDIC/network_inspection/ (combined maps and country comparisons).
Country networks are built separately; the combined map is an overlay, not an
electrically merged Nordic network. Duplicate cross-border records may remain.

The driver executes unmodified native clean_osm_data/build_osm_network/
base_network rule bodies, selected from the local Snakefile, in a small
Snakemake workflow. It does not run retrieval, clustering, generation/load
attachment or optimization. Existing country inputs are required.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

COUNTRIES = ("DK", "FI", "NO", "SE")
COUNTRY_NAMES = {"DK": "Denmark", "FI": "Finland", "NO": "Norway", "SE": "Sweden"}
NA = ["NULL", "", "N/A", "NAN", "NaN", "nan", "Nan", "n/a", "null"]


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--countries", nargs="+", choices=COUNTRIES, default=list(COUNTRIES))
    p.add_argument("--reuse-existing", action="store_true",
                   help="Reuse existing base.nc files; build countries that have no base.nc")
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def native_rule(text, name, root):
    """Read the actual local rule, rather than reimplementing its parameters."""
    match = re.search(rf"(?ms)^rule {re.escape(name)}:\n.*?(?=^\S|\Z)", text)
    if not match:
        raise RuntimeError(f"Cannot find a conventional native rule {name!r} in {root / 'Snakefile'}.")
    rule = match.group(0)
    # Snakemake resolves script paths relative to the generated workflow file.
    rule = re.sub(r'(?m)^(\s*)["\'](scripts/[^"\']+)["\']\s*$',
                  lambda m: m[1] + repr(str(root / m[2])), rule)
    return rule


def build(root, out, cores, dry_run, country):
    import yaml
    config_file = root / "config.default.yaml"
    if not config_file.is_file() or not (root / "Snakefile").is_file():
        raise FileNotFoundError("Put nordic_networks.py in the PyPSA-Earth repository root, beside Snakefile.")
    cfg = yaml.safe_load(config_file.read_text())
    cfg["countries"] = [country]
    cfg.setdefault("run", {})["name"] = country
    # All electrical, cleaning, topology and snapshot assumptions stay at defaults.
    config_path = out / f"config.{country}.defaults.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    resources = root / "resources" / country
    stages = [
        ("raw", [resources / "osm/raw" / f"all_raw_{x}.geojson"
                 for x in ("cables", "generators", "lines", "substations")],
         ["clean_osm_data", "build_osm_network", "base_network"]),
        ("clean", [resources / "osm/clean" / f"all_clean_{x}.geojson"
                   for x in ("generators", "lines", "substations")],
         ["build_osm_network", "base_network"]),
        ("built", [resources / "base_network" / f"all_{x}_build_network.csv"
                   for x in ("buses", "lines", "transformers", "converters")],
         ["base_network"]),
    ]
    for stage, files, rules in stages:
        if all(x.is_file() for x in files):
            break
    else:
        missing = "\n".join(f"{name}: " + ", ".join(str(x) for x in files if not x.is_file())
                            for name, files, _ in stages)
        raise FileNotFoundError("No complete input stage is available:\n" + missing)
    needed_shapes = ["country_shapes.geojson", "offshore_shapes.geojson"]
    if stage == "raw":
        needed_shapes += ["extended_country_shape.geojson"]
    for name in needed_shapes:
        if not (resources / "shapes" / name).is_file():
            raise FileNotFoundError(resources / "shapes" / name)
    if stage != "raw":
        print(f"Starting with existing {stage} data; upstream cleaning assumptions cannot be re-applied.")
    text = (root / "Snakefile").read_text()
    workflow = out / f"workflow.{country}.smk"
    header = (f"import sys\nsys.path.insert(0, {str(root / 'scripts')!r})\n"
              f"configfile: {str(config_path)!r}\nRDIR = '{country}/'\n\n")
    workflow.write_text(header + "\n\n".join(native_rule(text, r, root) for r in rules))
    cmd = [sys.executable, "-m", "snakemake", "--snakefile", str(workflow),
           "--directory", str(root), "--cores", str(cores), "--forceall",
           "--printshellcmds"]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(f"networks/{country}/base.nc")
    print(f"[{country}] Native build from {stage}; rules: {', '.join(rules)}", flush=True)
    with (out / "build.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"Native Snakemake build failed. See {out / 'build.log'}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    (out / "build_provenance.json").write_text(json.dumps(
        {"repository": str(root), "commit": commit, "input_stage": stage,
         "rules": rules, "command": cmd, "defaults": str(config_file)}, indent=2))


def read_table(path, id_col):
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=NA)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    return frame.set_index(id_col)


def truth(series):
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def collect(n, root, country):
    folder = root / f"resources/{country}/base_network"
    original_buses = read_table(folder / "all_buses_build_network.csv", "bus_id")
    original_lines = read_table(folder / "all_lines_build_network.csv", "line_id")
    if original_buses.empty or original_lines.empty:
        raise ValueError("Constructed OSM bus/line CSVs are required for station IDs and line geometries.")
    buses = n.buses.copy()
    buses.index = buses.index.astype(str)
    metadata = original_buses.reindex(buses.index)
    if "station_id" not in metadata:
        raise ValueError("The bus CSV has no station_id; cannot calculate station hosting reliably.")
    buses["station_id"] = [str(s) if pd.notna(s) else f"bus:{i}"
                           for i, s in metadata.station_id.items()]
    buses["tag_area"] = pd.to_numeric(metadata.get("tag_area"), errors="coerce")
    buses["area_backed"] = buses.tag_area.fillna(0).gt(0)
    buses["dc"] = buses.carrier.astype(str).str.upper().eq("DC")
    buses["v_nom"] = pd.to_numeric(buses.v_nom, errors="raise")
    records = []
    fallback_geometry = 0
    for component, table in [("Line", n.lines), ("Link", n.links)]:
        for key, row in table.iterrows():
            key = str(key)
            if key not in original_lines.index:
                continue  # Transformers and converters are not transmission line lengths.
            old = original_lines.loc[key]
            b0, b1 = str(row.bus0), str(row.bus1)
            if b0 not in buses.index or b1 not in buses.index:
                raise ValueError(f"{component} {key}: missing endpoint {b0}/{b1}")
            geometry = None
            for value in (row.get("geometry"), old.get("geometry")):
                if isinstance(value, str) and value.strip():
                    try:
                        candidate = wkt.loads(value)
                        if not candidate.is_empty and candidate.geom_type in ("LineString", "MultiLineString"):
                            geometry = candidate
                            break
                    except Exception:
                        pass
            if geometry is None:
                fallback_geometry += 1
                geometry = LineString([(buses.at[b0, "x"], buses.at[b0, "y"]),
                                       (buses.at[b1, "x"], buses.at[b1, "y"])])
            voltage = row.get("v_nom", float("nan"))
            if pd.isna(voltage) or float(voltage) <= 0:
                voltage = float(old.voltage) / 1000
            is_dc = str(old.get("dc", "false")).lower() in ("true", "1")
            circuits = pd.to_numeric(old.get("circuits"), errors="coerce")
            planned = str(old.get("under_construction", "false")).lower() in ("true", "1")
            records.append(dict(line_id=key, component=component, bus0=b0, bus1=b1,
                                station0=buses.at[b0, "station_id"], station1=buses.at[b1, "station_id"],
                                voltage_kv=float(voltage), system="DC" if is_dc else "AC",
                                status="under_construction" if planned else "existing",
                                model_length_km=float(row.get("length", float(old.length) / 1000)),
                                circuits=circuits, geometry=geometry))
    if not records:
        raise ValueError("No base-network line IDs match the constructed OSM lines.")
    lines = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    # Geodesic lengths avoid high-latitude Web Mercator length distortion.
    geod = Geod(ellps="WGS84")
    lines["geometry_length_km"] = lines.geometry.map(lambda x: geod.geometry_length(x) / 1000)
    lines["model_circuit_km"] = lines.model_length_km * lines.circuits
    lines["layer"] = lines.apply(lambda x: f"{x.system} {x.voltage_kv:g} kV", axis=1)
    print(f"Retained {len(buses)} buses and {len(lines)} transmission-line records; "
          f"{fallback_geometry} geometries use straight endpoint fallbacks.")
    return buses, lines, fallback_geometry


def graph_metrics(label, graph):
    count = graph.number_of_nodes()
    components = list(nx.connected_components(graph))
    return dict(network=label, nodes=count, edges=graph.number_of_edges(),
                mean_degree=2 * graph.number_of_edges() / count if count else 0,
                components=len(components), isolates=nx.number_of_isolates(graph),
                largest_component_nodes=max(map(len, components), default=0))


def statistics(buses, lines, out, title):
    lengths = lines.groupby(["system", "voltage_kv", "status"], as_index=False).agg(
        line_records=("line_id", "size"), model_length_km=("model_length_km", "sum"),
        geometry_length_km=("geometry_length_km", "sum"),
        model_circuit_km=("model_circuit_km", lambda s: s.sum(min_count=1)),
        unknown_circuit_records=("circuits", lambda s: s.isna().sum()))
    lengths.to_csv(out / "lengths_by_voltage.csv", index=False)
    hosting = []
    for scope, subset in [("all_model_station_groups", buses),
                          ("positive_mapped_area_station_groups", buses[
                              buses.station_id.isin(buses.loc[buses.area_backed, "station_id"])])]:
        denominator = subset.station_id.nunique()
        for (system, voltage), group in subset.groupby(["dc", "v_nom"]):
            count = group.station_id.nunique()
            hosting.append(dict(scope=scope, system="DC" if system else "AC", voltage_kv=voltage,
                                stations_hosting=count, total_stations=denominator,
                                percent=100 * count / denominator if denominator else float("nan")))
    hosting = pd.DataFrame(hosting)
    hosting.to_csv(out / "substation_voltage_hosting.csv", index=False)
    station_rows = []
    for sid, group in buses.groupby("station_id"):
        station_rows.append(dict(station_id=sid, lon=group.x.mean(), lat=group.y.mean(),
                                 buses="|".join(map(str, group.index)),
                                 voltages="; ".join(sorted(set(f"{'DC' if r.dc else 'AC'} {r.v_nom:g} kV"
                                                              for _, r in group.iterrows()))),
                                 positive_mapped_area=bool(group.area_backed.any())))
    stations = pd.DataFrame(station_rows)
    stations.to_csv(out / "stations.csv", index=False)
    buses.to_csv(out / "buses.csv", index_label="bus_id")
    lines.drop(columns="geometry").to_csv(out / "lines.csv", index=False)
    lines.to_file(out / "lines.geojson", driver="GeoJSON")

    # Physical-style station graph: contract voltage buses using native station_id;
    # omit internal station edges. Per-voltage graphs use voltage-specific buses.
    graphs = {}
    ac = lines[lines.system.eq("AC")]
    ac_buses = buses[~buses.dc]
    for label, minimum, maximum in [("AC_all_station_graph", 0, float("inf")),
                                    ("AC_regional_lt220_station_graph", 0, 220),
                                    ("AC_EHV_ge220_station_graph", 220, float("inf"))]:
        subset = ac[(ac.voltage_kv >= minimum) & (ac.voltage_kv < maximum)]
        node_buses = ac_buses[(ac_buses.v_nom >= minimum) & (ac_buses.v_nom < maximum)]
        graph = nx.Graph()
        graph.add_nodes_from(node_buses.station_id.unique())
        graph.add_edges_from((r.station0, r.station1) for _, r in subset.iterrows()
                             if r.station0 != r.station1)
        graphs[label] = graph
    for (system, voltage), subset in lines.groupby(["system", "voltage_kv"]):
        graph = nx.Graph()
        mask = buses.v_nom.eq(voltage) & buses.dc.eq(system == "DC")
        graph.add_nodes_from(buses.index[mask])
        graph.add_edges_from((r.bus0, r.bus1) for _, r in subset.iterrows() if r.bus0 != r.bus1)
        graphs[f"{system}_{voltage:g}kV_bus_graph"] = graph
    summaries, distributions, node_rows = [], [], []
    for label, graph in graphs.items():
        summaries.append(graph_metrics(label, graph))
        counts = pd.Series([d for _, d in graph.degree()], dtype=int).value_counts().sort_index()
        total = graph.number_of_nodes()
        for degree, count in counts.items():
            distributions.append(dict(network=label, degree=int(degree), count=int(count),
                                      probability=count / total,
                                      ccdf=counts[counts.index >= degree].sum() / total))
        node_rows.extend(dict(network=label, node_id=node, degree=degree) for node, degree in graph.degree())
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "graph_summary.csv", index=False)
    distribution = pd.DataFrame(distributions)
    distribution.to_csv(out / "degree_distributions.csv", index=False)
    pd.DataFrame(node_rows).to_csv(out / "node_degrees.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in list(graphs)[:3]:
        data = distribution[distribution.network.eq(label)]
        axes[0].plot(data.degree, data.probability, "o-", ms=4, label=label.replace("_station_graph", ""))
        data = data[data.degree >= 1]
        axes[1].semilogy(data.degree, data.ccdf, "o-", ms=4)
    axes[0].set(xlabel="Degree k", ylabel="P(K = k)")
    axes[1].set(xlabel="Degree k", ylabel="P(K ≥ k)")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"{title} — simple AC station graphs (no degree-two suppression)")
    fig.tight_layout()
    fig.savefig(out / "degree_distribution.png", dpi=220)
    plt.close(fig)
    print("\nLINE LENGTHS\n", lengths.round(3).to_string(index=False))
    print("\nVOLTAGE HOSTING\n", hosting.round(3).to_string(index=False))
    print("\nGRAPH SUMMARY\n", summary.round(3).to_string(index=False))
    return stations


def maps(n, buses, lines, stations, root, out, countries, title, colors):
    import folium
    from matplotlib.lines import Line2D
    from folium.plugins import Fullscreen, MeasureControl, MousePosition
    from branca.element import Element
    levels = (lines[["layer", "system", "voltage_kv"]].drop_duplicates()
              .sort_values(["system", "voltage_kv"]).layer.tolist())
    m = folium.Map(location=[65, 14], zoom_start=5, tiles=None, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics and the GIS User Community",
                     name="Satellite imagery").add_to(m)
    fields = ["country", "line_id", "component", "bus0", "bus1", "station0", "station1", "voltage_kv",
              "system", "status", "model_length_km", "geometry_length_km", "circuits"]
    for (country, level), subset in lines.groupby(["country", "layer"], sort=True):
        layer = folium.FeatureGroup(name=f"{country} — {level}", show=True)
        folium.GeoJson(json.loads(subset[fields + ["geometry"]].to_json()),
                       style_function=lambda feature, color=colors[level]: {
                           "color": color, "weight": 2.3, "opacity": .85,
                           "dashArray": "6 4" if feature["properties"]["system"] == "DC" else None},
                       highlight_function=lambda _: {"weight": 6, "opacity": 1},
                       tooltip=folium.GeoJsonTooltip(fields=["country", "line_id", "voltage_kv", "system"]),
                       popup=folium.GeoJsonPopup(fields=fields)).add_to(layer)
        layer.add_to(m)
    for area_backed, label, visible in [(True, "Station groups with mapped area", True),
                                       (False, "Other station groups / line-end nodes", False)]:
        layer = folium.FeatureGroup(name=label, show=visible)
        for row in stations[stations.positive_mapped_area.eq(area_backed)].itertuples():
            popup = (f"<b>Station {html.escape(str(row.station_id))}</b><br>"
                     f"Country: {html.escape(row.country)}<br>Voltages: {html.escape(row.voltages)}<br>Bus IDs: {html.escape(row.buses)}<br>"
                     f"Positive mapped area: {row.positive_mapped_area}")
            folium.CircleMarker([row.lat, row.lon], radius=3, color="#222222" if area_backed else "#888888",
                                weight=1, fill=True, fill_opacity=.7,
                                tooltip=f"Station {row.station_id}: {row.voltages}",
                                popup=folium.Popup(popup, max_width=450)).add_to(layer)
        layer.add_to(m)
    connectors = folium.FeatureGroup(name="Model transformers and converters", show=False)
    for component, table in [("Transformer", n.transformers), ("Link", n.links)]:
        for key, row in table.iterrows():
            if component == "Link" and str(key) in set(lines.line_id):
                continue
            b0, b1 = str(row.bus0), str(row.bus1)
            if b0 in buses.index and b1 in buses.index:
                folium.PolyLine([[buses.at[b0, "y"], buses.at[b0, "x"]],
                                 [buses.at[b1, "y"], buses.at[b1, "x"]]], color="#555555",
                                weight=1, dash_array="3 3",
                                tooltip=f"{component} {html.escape(str(key))}: {b0}–{b1}").add_to(connectors)
    connectors.add_to(m)
    legend = "".join(f'<div><span style="color:{colors[level]}">━━</span> {html.escape(level)}</div>'
                     for level in levels)
    m.get_root().html.add_child(Element('<div style="position:fixed;bottom:25px;left:15px;z-index:9999;'
                                        'background:white;padding:10px;border:1px solid #aaa;max-height:55vh;'
                                        f'overflow:auto;font-size:12px"><b>{html.escape(title)} — PyPSA-Earth</b>' + legend + '</div>'))
    bounds = lines.total_bounds
    m.fit_bounds([[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]])
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="kilometers").add_to(m)
    MousePosition().add_to(m)
    m.save(str(out / "network_interactive.html"))

    # Local country shape makes the static output independent of map tile services.
    projection = "EPSG:3035"
    fig, ax = plt.subplots(figsize=(9, 12))
    for country in countries:
        shape = gpd.read_file(root / f"resources/{country}/shapes/country_shapes.geojson")
        if shape.crs is None:
            shape = shape.set_crs(4326)
        shape.to_crs(projection).plot(ax=ax, facecolor="#f2f2ee", edgecolor="#aaaaaa", linewidth=.35)
    projected = lines.to_crs(projection)
    for level in levels:
        projected[projected.layer.eq(level)].plot(ax=ax, color=colors[level], linewidth=.7,
                                                 linestyle="--" if level.startswith("DC") else "-")
    # Show all built buses; station groups can contain slightly offset voltage buses.
    points = gpd.GeoDataFrame(buses, geometry=gpd.points_from_xy(buses.x, buses.y), crs=4326)
    points.to_crs(projection).plot(ax=ax, markersize=1, color="#333333", alpha=.5)
    bx0, by0, bx1, by1 = projected.total_bounds
    ax.set_xlim(bx0 - 40000, bx1 + 40000)
    ax.set_ylim(by0 - 40000, by1 + 40000)
    ax.legend(handles=[Line2D([0], [0], color=colors[level], lw=2,
                              ls="--" if level.startswith("DC") else "-", label=level)
                       for level in levels], loc="upper left", fontsize=8, frameon=True)
    ax.set_title(f"{title} — PyPSA-Earth\nTransmission lines by nominal voltage")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "network_by_voltage.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "network_by_voltage.pdf", bbox_inches="tight")
    plt.close(fig)


def write_notes(root, out, country, fallbacks):
    notes = f"""{COUNTRY_NAMES[country]} network inspection
=========================
Network: {root / f"networks/{country}/base.nc"}
Build configuration: config.{country}.defaults.yaml; native rule copies: workflow.{country}.smk.

Interpretation
--------------
* This is the unclustered base grid, before generation/load attachment or OPF.
* Defaults come from this checkout's config.default.yaml, not config.yaml.
  Only countries=[{country}] and run.name={country} are changed. No NVE repair logic is added.
* The native cleaning/grouping/line-parameter assumptions can change with version.
* A station is a native station_id group, which can combine nearby sites and
  include added line ends or junctions. Missing station IDs use individual bus IDs.
* Hosting is reported for all station groups AND groups containing positive
  mapped substation area. Positive area is a polygon-backed proxy, not a complete
  inventory of physical substations: mapped point substations may have zero area.
  The denominator is all stations in the selected scope, across AC/DC voltages;
  hosting percentages can sum above 100 because stations host multiple voltages.
* Lengths include every retained transmission record once; parallel records are
  not deduplicated into unique geographic corridors. Transformers/converters are
  excluded. Model lengths are the native PyPSA lengths; geometry lengths are WGS84
  geodesic lengths, which can differ after snapping/grouping. Circuit-km multiplies
  native length by the constructed OSM circuit count; unknown counts stay unknown.
  Existing and under-construction records are tabulated separately.
* Degree graphs are undirected and simple: parallel records become one edge,
  self-loops are omitted, isolates remain, degree-two nodes are not suppressed.
  Degrees include retained under-construction transmission records, like the map.
* AC station graphs contract all voltage buses with a common station_id and omit
  internal station connections. DC lines and transformers/converters are excluded.
  Regional is <220 kV; EHV is >=220 kV. Per-voltage AC/DC graphs use individual
  electrical buses and transmission lines at that voltage (no transformer edges).
* Degree CSVs contain empirical PMF and P(K>=k), including degree-zero entries.
  These distributions describe the constructed topology, not a cleaned NVE-style
  graph with every non-station degree-two junction suppressed.
* Maps use retained model line geometries with constructed-CSV fallback. Straight
  endpoint fallback count: {fallbacks}. Station markers use group-mean coordinates.
* HTML requires internet for Leaflet resources and background tiles. Static maps
  use the downloaded country shape and do not require background tile downloads.
* Each country is built independently. The Nordic map overlays those networks;
  station/bus IDs are prefixed by country and no cross-border topology is inferred.
  Cross-border assets can occur in more than one country extract. Country length
  tables are retained separately; summing them is not a deduplicated Nordic total.
* DK includes all Denmark in the supplied extract (including DK1 and DK2); this
  is a four-country geographic view, not a Nordic synchronous-area selection.
"""
    (out / "README.txt").write_text(notes)


def voltage_colors(items):
    levels = (pd.concat([lines[["layer", "system", "voltage_kv"]] for _, _, lines, _ in items])
              .drop_duplicates().sort_values(["system", "voltage_kv"]).layer.tolist())
    cmap = plt.get_cmap("tab10" if len(levels) <= 10 else "tab20" if len(levels) <= 20 else "turbo")
    return {level: mpl.colors.to_hex(cmap(i if len(levels) <= 20 else i / max(len(levels) - 1, 1)))
            for i, level in enumerate(levels)}


def combine_for_maps(items):
    """Namespace each graph. Overlay without inventing border connectivity."""
    from types import SimpleNamespace
    all_buses, all_lines, all_stations = [], [], []
    connectors = {"links": [], "transformers": []}
    for country, n, buses, lines, stations in items:
        prefix = country + ":"
        buses = buses.copy()
        buses.index = prefix + buses.index.astype(str)
        buses["station_id"] = prefix + buses.station_id.astype(str)
        lines = lines.copy()
        for col in ("line_id", "bus0", "bus1", "station0", "station1"):
            lines[col] = prefix + lines[col].astype(str)
        stations = stations.copy()
        stations["station_id"] = prefix + stations.station_id.astype(str)
        stations["buses"] = stations.buses.map(lambda value: "|".join(prefix + b for b in str(value).split("|")))
        for attr in connectors:
            table = getattr(n, attr).copy()
            table.index = prefix + table.index.astype(str)
            for col in ("bus0", "bus1"):
                table[col] = prefix + table[col].astype(str)
            connectors[attr].append(table)
        all_buses.append(buses)
        all_lines.append(lines)
        all_stations.append(stations)
    n = SimpleNamespace(**{key: pd.concat(tables) for key, tables in connectors.items()})
    buses = pd.concat(all_buses)
    lines = gpd.GeoDataFrame(pd.concat(all_lines, ignore_index=True), geometry="geometry", crs=4326)
    stations = pd.concat(all_stations, ignore_index=True)
    if not buses.index.is_unique or not stations.station_id.is_unique:
        raise ValueError("Country-prefixed identifiers are not unique")
    return n, buses, lines, stations


def comparisons(root, out, countries):
    """Compare country metrics; deliberately do not sum overlapping extracts."""
    tables = {}
    for filename in ["lengths_by_voltage.csv", "substation_voltage_hosting.csv", "graph_summary.csv",
                     "degree_distributions.csv", "node_degrees.csv"]:
        frames = []
        for country in countries:
            table = pd.read_csv(root / f"results/{country}/network_inspection" / filename)
            table.insert(0, "country", country)
            frames.append(table)
        tables[filename] = pd.concat(frames, ignore_index=True)
        tables[filename].to_csv(out / filename, index=False)
    lengths = tables["lengths_by_voltage.csv"]
    lengths.pivot_table(index=["system", "voltage_kv", "status"], columns="country",
                        values="model_length_km", aggfunc="sum").to_csv(out / "length_comparison_km.csv")
    hosting = tables["substation_voltage_hosting.csv"]
    hosting.pivot_table(index=["scope", "system", "voltage_kv"], columns="country",
                        values="percent", aggfunc="first").to_csv(out / "hosting_comparison_percent.csv")
    summary = tables["graph_summary.csv"]
    country_summary = []
    for country in countries:
        row = summary[(summary.country == country) & (summary.network == "AC_all_station_graph")].iloc[0]
        result = {"country": country}
        result.update({"ac_station_" + col: row[col] for col in
                       ["nodes", "edges", "mean_degree", "components", "isolates", "largest_component_nodes"]})
        for system in ["AC", "DC"]:
            for status in ["existing", "under_construction"]:
                subset = lengths[(lengths.country == country) & (lengths.system == system)
                                 & (lengths.status == status)]
                result[f"{system}_{status}_model_length_km"] = subset.model_length_km.sum()
        country_summary.append(result)
    pd.DataFrame(country_summary).to_csv(out / "country_comparison.csv", index=False)
    distribution = tables["degree_distributions.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    labels = [("AC_all_station_graph", "All AC"),
              ("AC_regional_lt220_station_graph", "Regional AC (<220 kV)"),
              ("AC_EHV_ge220_station_graph", "EHV AC (≥220 kV)")]
    for ax, (network, title) in zip(axes, labels):
        for country in countries:
            data = distribution[(distribution.country == country) & (distribution.network == network)
                                & (distribution.degree >= 1)]
            if not data.empty:
                ax.semilogy(data.degree, data.ccdf, "o-", ms=3, label=country)
        ax.set(xlabel="Degree k", ylabel="P(K ≥ k)", title=title)
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle("Country comparison — simple station graphs")
    fig.tight_layout()
    fig.savefig(out / "degree_comparison.png", dpi=220)
    fig.savefig(out / "degree_comparison.pdf")
    plt.close(fig)


def load_dependencies():
    global pd, gpd, nx, plt, mpl, wkt, LineString, Geod, pypsa
    try:
        import pandas as pd
        import geopandas as gpd
        import networkx as nx
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from shapely import wkt
        from shapely.geometry import LineString
        from pyproj import Geod
        import pypsa
        import folium
    except ImportError as exc:
        raise RuntimeError("Run with 'pixi run python nordic_networks.py'. "
                           "If Folium is missing, use 'pixi add folium'.") from exc


def main():
    args = arguments()
    root = args.root.expanduser().resolve()
    countries = list(dict.fromkeys(args.countries))
    if args.cores < 1:
        raise ValueError("--cores must be at least 1")
    if args.skip_build and args.dry_run:
        raise ValueError("Use --dry-run without --skip-build")
    if not args.dry_run:
        load_dependencies()
    combined_out = root / "results/NORDIC/network_inspection"
    combined_out.mkdir(parents=True, exist_ok=True)
    items = []
    for country in countries:
        out = root / f"results/{country}/network_inspection"
        out.mkdir(parents=True, exist_ok=True)
        network_path = root / f"networks/{country}/base.nc"
        print(f"\n{'=' * 12} {COUNTRY_NAMES[country]} ({country}) {'=' * 12}", flush=True)
        reuse = args.reuse_existing and network_path.is_file()
        if not args.skip_build and not reuse:
            build(root, out, args.cores, args.dry_run, country)
        elif reuse:
            print(f"Reusing {network_path}")
        if args.dry_run:
            continue
        if not network_path.is_file():
            raise FileNotFoundError(f"{network_path}: run without --skip-build to construct this country.")
        n = pypsa.Network(str(network_path))
        buses, lines, fallbacks = collect(n, root, country)
        buses["country"] = country
        lines["country"] = country
        stations = statistics(buses, lines, out, COUNTRY_NAMES[country])
        stations["country"] = country
        stations.to_csv(out / "stations.csv", index=False)
        write_notes(root, out, country, fallbacks)
        items.append((country, n, buses, lines, stations))
    if args.dry_run:
        return
    colors = voltage_colors([(c, b, l, s) for c, _, b, l, s in items])
    for country, n, buses, lines, stations in items:
        out = root / f"results/{country}/network_inspection"
        maps(n, buses, lines, stations, root, out, [country], COUNTRY_NAMES[country], colors)
    n, buses, lines, stations = combine_for_maps(items)
    maps(n, buses, lines, stations, root, combined_out, countries,
         "Norden — separate country networks", colors)
    lines.to_file(combined_out / "lines.geojson", driver="GeoJSON")
    buses.to_csv(combined_out / "buses.csv", index_label="bus_id")
    stations.to_csv(combined_out / "stations.csv", index=False)
    comparisons(root, combined_out, countries)
    (combined_out / "README.txt").write_text(
        "Nordic comparison: " + ", ".join(countries) + "\n\n"
        "Country builds use the local checkout's defaults and are independent.\n"
        "The combined maps/GeoJSON overlay those builds with country-prefixed IDs.\n"
        "They do not create a merged electrical model, resolve border connectivity,\n"
        "or deduplicate assets that occur in multiple extracts. Do not treat the sum\n"
        "of country lengths as a unique Nordic network length. DK includes DK1/DK2.\n\n"
        "The comparison CSVs retain the source country on every row. Hosting\n"
        "denominators are country-specific. Missing pivot entries mean that a level\n"
        "is absent from that table, not an assumed measured zero. Regional <220 kV;\n"
        "EHV >=220 kV. Degree comparisons use simple AC station graphs including\n"
        "isolates and retained under-construction lines, without junction suppression.\n"
        "See each country's README.txt for length, hosting and degree definitions.\n"
        "Identical AC/DC voltage labels use identical colors in every map.\n"
        "Build logs/configuration/provenance are under each country's output folder.\n"
        "--reuse-existing keeps existing base.nc files and does not re-apply defaults\n"
        "to them; --skip-build requires every selected country's network to exist.\n")
    print(f"\nFinished {', '.join(countries)}.\nCountry outputs: results/<country>/network_inspection/"
          f"\nCombined maps and comparisons: {combined_out}")
    if not args.no_browser:
        webbrowser.open((combined_out / "network_interactive.html").as_uri())


if __name__ == "__main__":
    main()
