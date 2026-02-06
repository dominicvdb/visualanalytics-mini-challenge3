import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json as json_lib
    import marimo as mo
    from networkx.readwrite import json_graph
    from datetime import datetime
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import Counter
    return Counter, alt, datetime, json_graph, json_lib, mo, nx, pd, plt


@app.cell
def _(json_lib, mo, pd):
    with open("data/MC3_schema.json", "r") as f:
        schema = json_lib.load(f)

    rows = []
    for node_type, node_info in schema["schema"]["nodes"].items():
        for sub_type, details in node_info["sub_types"].items():
            row = {
                "type": node_type,
                "sub_type": sub_type,
                "description": details.get("description", ""),
                "count": details.get("count", ""),
            }
            rows.append(row)

    df_schema = pd.DataFrame(rows)
    mo.ui.table(df_schema)
    return


@app.cell
def _(json_graph, json_lib):
    # Load graph
    with open("data/MC3_graph.json", "r") as g:
        json_data = json_lib.load(g)

    G = json_graph.node_link_graph(json_data, edges="edges")

    # Look at a few Communication nodes
    for node, attrs in G.nodes(data=True):
        if attrs.get("sub_type") == "Communication":
            print(node, attrs)
            break  # just one to start
    return (G,)


@app.cell
def _(G, datetime, pd):
    comms = []
    for n, a in G.nodes(data=True):
        if a.get("sub_type") == "Communication":
            ts = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
            comms.append({
                "node_id": n,
                "timestamp": ts,
                "hour": ts.hour,
                "date": ts.date(),
                "day_of_week": ts.strftime("%A"),
                "content": a.get("content", ""),
            })

    df_comms = pd.DataFrame(comms)
    df_comms
    return (df_comms,)


@app.cell
def _(df_comms):
    df_comms['timestamp'].min()
    return


@app.cell
def _(alt, df_comms):
    chart_hourly = alt.Chart(df_comms).mark_bar().encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("count()", title="Number of Messages"),
    ).properties(
        title="Messages by Hour of Day",
        width=600,
        height=300
    )

    chart_hourly
    return


@app.cell
def _(G, df_comms, mo):
    # Pick the first communication node
    first_comm = df_comms["node_id"].iloc[0]

    # Get all edges connected to this node (both in and out)
    in_edges = [(u, G.edges[u, first_comm]) for u in G.predecessors(first_comm)]
    out_edges = [(v, G.edges[first_comm, v]) for v in G.successors(first_comm)]

    mo.md(f"""
    ### Communication Node: `{first_comm}`

    **Content:** {G.nodes[first_comm].get('content', '')}

    **Timestamp:** {G.nodes[first_comm].get('timestamp', '')}

    **Incoming edges (who sends to this node):**
    {in_edges}

    **Outgoing edges (where this node points to):**
    {out_edges}
    """)
    return (first_comm,)


@app.cell
def _(G, first_comm, nx, plt):
    # Build a subgraph of this node and its neighbors
    neighbors = list(G.predecessors(first_comm)) + list(G.successors(first_comm))
    sub_nodes = [first_comm] + neighbors
    subgraph = G.subgraph(sub_nodes)

    # Color nodes by type
    color_map = {
        "Entity": "#FF8C8C",
        "Event": "#5B9BD5",
        "Relationship": "#FFB347",
    }
    colors = [color_map.get(subgraph.nodes[n].get("type", ""), "gray") for n in subgraph.nodes]

    # Labels: use name or label
    labels = {n: subgraph.nodes[n].get("name", subgraph.nodes[n].get("label", n)) for n in subgraph.nodes}

    # Edge labels
    edge_labels = {(u, v): d.get("type", "") for u, v, d in subgraph.edges(data=True)}

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(subgraph, seed=42, k=1)
    nx.draw(subgraph, pos, ax=ax, with_labels=True, labels=labels,
            node_color=colors, node_size=500, font_size=9, arrows=True)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=8, label_pos=0.3)
    ax.set_title(f"Neighborhood of {first_comm}")
    plt.tight_layout()
    fig
    return


@app.cell
def _(df_comms, mo):
    comm_slider = mo.ui.slider(
        start=0,
        stop=len(df_comms) - 1,
        step=1,
        value=0,
        label="Communication index"
    )
    comm_number = mo.ui.number(
        start=0,
        stop=len(df_comms) - 1,
        step=1,
        value=0,
        label="Or type index:"
    )
    return (comm_number,)


@app.cell
def _(G, comm_number, df_comms, mo, nx, plt):
    idx = comm_number.value
    selected_comm = df_comms["node_id"].iloc[idx]

    # Build subgraph
    nbrs = list(G.predecessors(selected_comm)) + list(G.successors(selected_comm))
    sub = G.subgraph([selected_comm] + nbrs)

    # Colors
    cmap = {"Entity": "#FF8C8C", "Event": "#5B9BD5", "Relationship": "#FFB347"}
    node_colors = [cmap.get(sub.nodes[nd].get("type", ""), "gray") for nd in sub.nodes]

    # Labels
    node_labels = {nd: sub.nodes[nd].get("name", sub.nodes[nd].get("label", nd)) for nd in sub.nodes}

    # Edge labels
    e_labels = {}
    for u, v, d in sub.edges(data=True):
        if d.get("type"):
            e_labels[(u, v)] = d["type"]
        elif sub.nodes[u].get("type") == "Entity" and sub.nodes[v].get("type") == "Event":
            e_labels[(u, v)] = "participates_in"
        else:
            e_labels[(u, v)] = ""

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    layout = nx.spring_layout(sub, seed=42, k=2)
    nx.draw(sub, layout, ax=ax2, with_labels=True, labels=node_labels,
            node_color=node_colors, node_size=1500, font_size=9, arrows=True)
    nx.draw_networkx_edge_labels(sub, layout, edge_labels=e_labels, ax=ax2,
                                  font_size=8, label_pos=0.3)
    ax2.set_title(f"[{idx}] {selected_comm} — {G.nodes[selected_comm].get('timestamp', '')}")
    plt.tight_layout()

    mo.vstack([
        fig2,
        mo.md(f"**Content:** {G.nodes[selected_comm].get('content', '')}"),
        comm_number
    ])
    return


@app.cell
def _(df_comms, mo):
    unique_dates = sorted(df_comms["date"].unique())

    day_slider = mo.ui.slider(
        start=0,
        stop=len(unique_dates) - 1,
        step=1,
        value=0,
        label="Day",
        full_width=True,
        show_value=True
    )

    hour_slider = mo.ui.slider(
        start=8,
        stop=14,
        step=1,
        value=8,
        label="Hour",
        full_width=True,
        show_value=True
    )

    mo.vstack([day_slider, hour_slider])
    return day_slider, hour_slider, unique_dates


@app.cell
def _(Counter, G, df_comms, nx):
    # Build the full person-to-person graph across ALL communications
    all_edges = Counter()
    for c_id in df_comms["node_id"]:
        snd = [u for u in G.predecessors(c_id) if G.nodes[u].get("type") == "Entity"]
        rcv = [v for v in G.successors(c_id) if G.nodes[v].get("type") == "Entity"]
        for sn in snd:
            for rn in rcv:
                sn_name = G.nodes[sn].get("name", sn)
                rn_name = G.nodes[rn].get("name", rn)
                all_edges[(sn_name, rn_name)] += 1

    full_P = nx.DiGraph()
    for (sn, rn), cnt in all_edges.items():
        full_P.add_edge(sn, rn, weight=cnt)

    fixed_pos = nx.spring_layout(full_P, seed=42, k=3)
    return fixed_pos, full_P


@app.cell
def _(
    Counter,
    G,
    day_slider,
    df_comms,
    fixed_pos,
    full_P,
    hour_slider,
    mo,
    nx,
    plt,
    unique_dates,
):
    selected_date = unique_dates[day_slider.value]
    selected_hour = hour_slider.value

    filtered2 = df_comms[(df_comms["date"] == selected_date) & (df_comms["hour"] == selected_hour)]

    edge_counts = Counter()
    for cid in filtered2["node_id"]:
        s_list = [u for u in G.predecessors(cid) if G.nodes[u].get("type") == "Entity"]
        r_list = [v for v in G.successors(cid) if G.nodes[v].get("type") == "Entity"]
        for s in s_list:
            for r in r_list:
                s_name = G.nodes[s].get("name", s)
                r_name = G.nodes[r].get("name", r)
                edge_counts[(s_name, r_name)] += 1

    fig4, ax4 = plt.subplots(figsize=(12, 8))

    # Always draw all nodes in fixed positions
    nx.draw_networkx_nodes(full_P, fixed_pos, ax=ax4,
                            node_color="#FF8C8C", node_size=1200, alpha=0.3)
    nx.draw_networkx_labels(full_P, fixed_pos, ax=ax4, font_size=8, alpha=0.4)

    # Draw active edges on top
    if len(edge_counts) > 0:
        active_P = nx.DiGraph()
        for (s, r), count in edge_counts.items():
            active_P.add_edge(s, r, weight=count)

        active_nodes = set(active_P.nodes)
        nx.draw_networkx_nodes(full_P, fixed_pos, ax=ax4,
                                nodelist=[n for n in full_P.nodes if n in active_nodes],
                                node_color="#FF8C8C", node_size=1200, alpha=1.0)
        nx.draw_networkx_labels(full_P, fixed_pos, ax=ax4,
                                 labels={n: n for n in active_nodes}, font_size=8, alpha=1.0)

        weights4 = [active_P.edges[e]["weight"] * 2 for e in active_P.edges]
        nx.draw_networkx_edges(active_P, fixed_pos, ax=ax4,
                                width=weights4, edge_color="#5B9BD5", arrows=True)
        el4 = {(u, v): d["weight"] for u, v, d in active_P.edges(data=True)}
        nx.draw_networkx_edge_labels(active_P, fixed_pos, edge_labels=el4, ax=ax4, font_size=8, label_pos = 0.3)

    ax4.set_title(f"{selected_date} — Hour {selected_hour}:00  |  {len(filtered2)} messages")
    plt.tight_layout()

    mo.vstack([
        fig4,
        mo.md(f"**Date:** {selected_date}  |  **Hour:** {selected_hour}:00  |  **Messages:** {len(filtered2)}"),
        day_slider,
        hour_slider
    ])
    return


@app.cell
def _(
    Counter,
    G,
    day_slider,
    df_comms,
    fixed_pos,
    full_P,
    hour_slider,
    json_lib,
    unique_dates,
):
    selected_date2 = unique_dates[day_slider.value]
    selected_hour2 = hour_slider.value

    filtered3 = df_comms[(df_comms["date"] == selected_date2) & (df_comms["hour"] == selected_hour2)]

    edge_counts2 = Counter()
    for c in filtered3["node_id"]:
        sd = [u for u in G.predecessors(c) if G.nodes[u].get("type") == "Entity"]
        rv = [v for v in G.successors(c) if G.nodes[v].get("type") == "Entity"]
        for p1 in sd:
            for p2 in rv:
                p1_name = G.nodes[p1].get("name", p1)
                p2_name = G.nodes[p2].get("name", p2)
                edge_counts2[(p1_name, p2_name)] += 1

    nodes_data = [
        {"id": node, "x": fixed_pos[node][0], "y": fixed_pos[node][1]}
        for node in full_P.nodes
    ]

    active_set = set()
    for (p1, p2) in edge_counts2:
        active_set.add(p1)
        active_set.add(p2)

    edges_data = [
        {"source": p1, "target": p2, "weight": w}
        for (p1, p2), w in edge_counts2.items()
    ]

    nodes_json = json_lib.dumps(nodes_data)
    edges_json = json_lib.dumps(edges_data)
    active_json = json_lib.dumps(list(active_set))
    return (
        active_json,
        edges_json,
        filtered3,
        nodes_json,
        selected_date2,
        selected_hour2,
    )


@app.cell
def _(
    active_json,
    day_slider,
    edges_json,
    filtered3,
    hour_slider,
    mo,
    nodes_json,
    selected_date2,
    selected_hour2,
):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
    </head>
    <body style="margin:0;">
    <svg id="graph" width="900" height="600"></svg>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        const activeSet = new Set({active_json});

        const width = 900;
        const height = 600;

        const svg = d3.select("#graph");

        const xScale = d3.scaleLinear()
            .domain(d3.extent(nodes, d => d.x))
            .range([80, width - 80]);
        const yScale = d3.scaleLinear()
            .domain(d3.extent(nodes, d => d.y))
            .range([80, height - 80]);

        svg.append("defs").append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 0 10 10")
            .attr("refX", 8)
            .attr("refY", 5)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M 0 0 L 10 5 L 0 10 Z")
            .attr("fill", "#5B9BD5");

        const nodeMap = new Map(nodes.map(n => [n.id, n]));

        // Check if reverse edge exists for curving
        const edgeSet = new Set(edges.map(e => e.source + "|" + e.target));

        function arcPath(d) {{
            const x1 = xScale(nodeMap.get(d.source).x);
            const y1 = yScale(nodeMap.get(d.source).y);
            const x2 = xScale(nodeMap.get(d.target).x);
            const y2 = yScale(nodeMap.get(d.target).y);
    
            const dx = x2 - x1;
            const dy = y2 - y1;
            const dist = Math.sqrt(dx * dx + dy * dy);
    
            const hasBidi = edgeSet.has(d.target + "|" + d.source);
            const offset = hasBidi ? 30 : 0;
    
            // Midpoint + perpendicular offset
            const mx = (x1 + x2) / 2 + (-dy / dist) * offset;
            const my = (y1 + y2) / 2 + (dx / dist) * offset;
    
            return "M" + x1 + "," + y1 +
                   " Q" + mx + "," + my +
                   " " + x2 + "," + y2;
        }}

        svg.selectAll(".edge")
            .data(edges)
            .enter()
            .append("path")
            .attr("d", arcPath)
            .attr("fill", "none")
            .attr("stroke", "#5B9BD5")
            .attr("stroke-width", d => d.weight * 2)
            .attr("marker-end", "url(#arrow)")
            .attr("opacity", 0.8);

        // Edge weight labels along the arc
        svg.selectAll(".edge-label")
            .data(edges)
            .enter()
            .append("text")
            .each(function(d) {{
                const x1 = xScale(nodeMap.get(d.source).x);
                const y1 = yScale(nodeMap.get(d.source).y);
                const x2 = xScale(nodeMap.get(d.target).x);
                const y2 = yScale(nodeMap.get(d.target).y);
                const mx = (x1 + x2) / 2;
                const my = (y1 + y2) / 2;

                // Offset label perpendicular to edge
                const dx = x2 - x1;
                const dy = y2 - y1;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const hasBidi = edgeSet.has(d.target + "|" + d.source);
                const offset = hasBidi ? 12 : 5;
                const nx = -dy / dist * offset;
                const ny = dx / dist * offset;

                d3.select(this)
                    .attr("x", mx + nx)
                    .attr("y", my + ny)
                    .text(d.weight)
                    .attr("font-size", "11px")
                    .attr("fill", "#333")
                    .attr("text-anchor", "middle");
            }});

        svg.selectAll("circle")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", 14)
            .attr("fill", "#FF8C8C")
            .attr("opacity", d => activeSet.has(d.id) ? 1.0 : 0.2)
            .attr("stroke", d => activeSet.has(d.id) ? "#333" : "none")
            .attr("stroke-width", 1.5);

        svg.selectAll(".label")
            .data(nodes)
            .enter()
            .append("text")
            .attr("x", d => xScale(d.x))
            .attr("y", d => yScale(d.y) - 20)
            .text(d => d.id)
            .attr("font-size", "11px")
            .attr("fill", d => activeSet.has(d.id) ? "#000" : "#aaa")
            .attr("text-anchor", "middle");

        svg.selectAll("circle")
            .on("mouseover", function(event, d) {{
                d3.select(this).attr("r", 20).attr("fill", "#ff5555");
            }})
            .on("mouseout", function(event, d) {{
                d3.select(this).attr("r", 14).attr("fill", "#FF8C8C")
                    .attr("opacity", activeSet.has(d.id) ? 1.0 : 0.2);
            }});
    </script>
    </body>
    </html>
    """

    d3_graph = mo.iframe(html_content, width="100%", height="620px")

    mo.vstack([
        mo.md(f"### Network — {selected_date2} at {selected_hour2}:00 — {len(filtered3)} messages"),
        d3_graph,
        day_slider,
        hour_slider
    ])
    return


if __name__ == "__main__":
    app.run()
