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
    df_comms['timestamp'].max()
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


@app.cell
def _(G, df_comms, json_lib, pd):
    comm_details = []
    for c_node in df_comms["node_id"]:
        attrs_c = G.nodes[c_node]
        ts_c = attrs_c["timestamp"]

        # Find sender and receiver entities
        sender = None
        receiver = None
        for pred in G.predecessors(c_node):
            if G.nodes[pred].get("type") == "Entity":
                edge_data = G.edges[pred, c_node]
                if edge_data.get("type") == "sent":
                    sender = pred
        for succ in G.successors(c_node):
            if G.nodes[succ].get("type") == "Entity":
                edge_data = G.edges[c_node, succ]
                if edge_data.get("type") == "received":
                    receiver = succ

        if sender and receiver:
            comm_details.append({
                "node_id": c_node,
                "timestamp": ts_c,
                "sender_name": G.nodes[sender].get("name", sender),
                "sender_type": G.nodes[sender].get("sub_type", ""),
                "receiver_name": G.nodes[receiver].get("name", receiver),
                "receiver_type": G.nodes[receiver].get("sub_type", ""),
            })

    df_details = pd.DataFrame(comm_details)
    df_details["ts"] = pd.to_datetime(df_details["timestamp"])
    df_details["date_str"] = df_details["ts"].dt.date.astype(str)
    df_details["hour_float"] = df_details["ts"].dt.hour + df_details["ts"].dt.minute / 60

    # Convert Timestamp to string before JSON serialization
    df_details["ts_str"] = df_details["ts"].astype(str)
    details_json = json_lib.dumps(
        df_details.drop(columns=["ts"]).to_dict(orient="records")
    )
    print(f"{len(df_details)} communications with sender/receiver info")
    df_details.head()
    return details_json, df_details


@app.cell
def _(df_details, mo):
    mo.ui.table(df_details)
    return


@app.cell
def _(details_json, df_details, json_lib, mo):
    unique_dates_list = json_lib.dumps(sorted(df_details["date_str"].unique().tolist()))

    daily_chart = mo.iframe(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ margin: 0; font-family: sans-serif; }}
            .tooltip {{
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
                display: none;
                max-width: 300px;
            }}
        </style>
    </head>
    <body>
    <div id="chart"></div>
    <div class="tooltip" id="tooltip"></div>
    <script>
    try {{
        var data = {details_json};
        var dates = {unique_dates_list};

        var typeColors = {{
            "Person": "#7B68EE",
            "Organization": "#DC143C",
            "Vessel": "#00CED1",
            "Group": "#FF8C00",
            "Location": "#4169E1"
        }};

        var margin = {{top: 50, right: 30, bottom: 80, left: 80}};
        var rowHeight = 80;
        var summaryHeight = 80;
        var width = 1000 - margin.left - margin.right;
        var height = dates.length * rowHeight;
        var totalHeight = height + summaryHeight;
        var dotSize = 10;
        var dotGap = 2;
        var maxCols = 8;

        var svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", totalHeight + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var x = d3.scaleLinear()
            .domain([8, 16])
            .range([0, width]);

        var y = d3.scaleBand()
            .domain(dates)
            .range([0, height])
            .padding(0.1);

        var tickVals = [8,9,10,11,12,13,14,15];

        // X axis top
        svg.append("g")
            .call(d3.axisTop(x)
                .tickValues(tickVals)
                .tickFormat(function(d) {{ return d + ":00"; }}))
            .selectAll("text")
            .style("font-size", "11px");

        // Vertical grid lines
        tickVals.forEach(function(h) {{
            svg.append("line")
                .attr("x1", x(h)).attr("x2", x(h))
                .attr("y1", 0).attr("y2", totalHeight)
                .attr("stroke", "#eee").attr("stroke-width", 1);
        }});

        // Row backgrounds and day labels
        dates.forEach(function(date, i) {{
            svg.append("rect")
                .attr("x", 0).attr("y", y(date))
                .attr("width", width).attr("height", y.bandwidth())
                .attr("fill", i % 2 === 0 ? "#f9f9f9" : "#ffffff")
                .attr("stroke", "#eee");

            svg.append("text")
                .attr("x", -10)
                .attr("y", y(date) + y.bandwidth() / 2)
                .attr("text-anchor", "end")
                .attr("dominant-baseline", "middle")
                .style("font-size", "12px")
                .style("font-weight", "bold")
                .text(i + 1);
        }});

        // Per-day density
        dates.forEach(function(date) {{
            var dayData = data.filter(function(d) {{ return d.date_str === date; }});
            if (dayData.length === 0) return;

            var values = dayData.map(function(d) {{ return d.hour_float; }});
            var bins = d3.bin().domain([8, 15.5]).thresholds(25)(values);
            var maxCount = d3.max(bins, function(b) {{ return b.length; }});

            var areaY = d3.scaleLinear()
                .domain([0, maxCount || 1])
                .range([y(date) + y.bandwidth(), y(date) + y.bandwidth() * 0.2]);

            var area = d3.area()
                .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
                .y0(y(date) + y.bandwidth())
                .y1(function(b) {{ return areaY(b.length); }})
                .curve(d3.curveBasis);

            svg.append("path")
                .datum(bins)
                .attr("d", area)
                .attr("fill", "#5B9BD5")
                .attr("opacity", 0.15);
        }});

        // Group by date + hour
        var grouped = {{}};
        data.forEach(function(d) {{
            var hour = Math.floor(d.hour_float);
            var key = d.date_str + "|" + hour;
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(d);
        }});

        var tooltip = d3.select("#tooltip");

        // Draw split dots per bin
        Object.keys(grouped).forEach(function(key) {{
            var parts = key.split("|");
            var date = parts[0];
            var hour = parseInt(parts[1]);
            var items = grouped[key];

            var binLeft = x(hour) + 5;
            var rowTop = y(date);
            var bandHeight = y.bandwidth();

            items.forEach(function(d, idx) {{
                var col = idx % maxCols;
                var row = Math.floor(idx / maxCols);

                var px = binLeft + col * (dotSize + dotGap);
                var py = rowTop + row * (dotSize + dotGap);

                var g = svg.append("g")
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {{
                        d3.select(this).select(".outline").attr("stroke", "#333").attr("stroke-width", 2);
                        tooltip.style("display", "block")
                            .html("<strong>" + d.sender_name + " → " + d.receiver_name + "</strong><br/>"
                                + d.timestamp + "<br/>"
                                + "<span style='color:" + (typeColors[d.sender_type]||"#999") + "'>■</span> "
                                + d.sender_type + " → "
                                + "<span style='color:" + (typeColors[d.receiver_type]||"#999") + "'>■</span> "
                                + d.receiver_type)
                            .style("left", (event.pageX + 12) + "px")
                            .style("top", (event.pageY - 20) + "px");
                    }})
                    .on("mouseout", function() {{
                        d3.select(this).select(".outline").attr("stroke", "none");
                        tooltip.style("display", "none");
                    }});

                g.append("rect")
                    .attr("x", px).attr("y", py)
                    .attr("width", dotSize / 2).attr("height", dotSize)
                    .attr("fill", typeColors[d.sender_type] || "#999");

                g.append("rect")
                    .attr("x", px + dotSize / 2).attr("y", py)
                    .attr("width", dotSize / 2).attr("height", dotSize)
                    .attr("fill", typeColors[d.receiver_type] || "#999");

                g.append("rect")
                    .attr("class", "outline")
                    .attr("x", px).attr("y", py)
                    .attr("width", dotSize).attr("height", dotSize)
                    .attr("fill", "none").attr("rx", 1);
            }});
        }});

        // === Summary density at bottom ===
        var summaryTop = height + 10;

        // Background
        svg.append("rect")
            .attr("x", 0).attr("y", summaryTop)
            .attr("width", width).attr("height", summaryHeight)
            .attr("fill", "#f0f0f0")
            .attr("stroke", "#ddd");

        // Label
        svg.append("text")
            .attr("x", -10)
            .attr("y", summaryTop + summaryHeight / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "middle")
            .style("font-size", "11px")
            .style("font-weight", "bold")
            .text("All");

        // Overall density
        var allValues = data.map(function(d) {{ return d.hour_float; }});
        var allBins = d3.bin().domain([8, 15.5]).thresholds(40)(allValues);
        var allMax = d3.max(allBins, function(b) {{ return b.length; }});

        var summaryY = d3.scaleLinear()
            .domain([0, allMax || 1])
            .range([summaryTop + summaryHeight, summaryTop + 10]);

        var summaryArea = d3.area()
            .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
            .y0(summaryTop + summaryHeight)
            .y1(function(b) {{ return summaryY(b.length); }})
            .curve(d3.curveBasis);

        svg.append("path")
            .datum(allBins)
            .attr("d", summaryArea)
            .attr("fill", "#5B9BD5")
            .attr("opacity", 0.35);

        // Summary line on top of area
        var summaryLine = d3.line()
            .x(function(b) {{ return x((b.x0 + b.x1) / 2); }})
            .y(function(b) {{ return summaryY(b.length); }})
            .curve(d3.curveBasis);

        svg.append("path")
            .datum(allBins)
            .attr("d", summaryLine)
            .attr("fill", "none")
            .attr("stroke", "#5B9BD5")
            .attr("stroke-width", 2);

        // === Bottom X axis ===
        svg.append("g")
            .attr("transform", "translate(0," + (summaryTop + summaryHeight) + ")")
            .call(d3.axisBottom(x)
                .tickValues(tickVals)
                .tickFormat(function(d) {{ return d + ":00"; }}))
            .selectAll("text")
            .style("font-size", "11px");

        // Legend
        var legendData = Object.entries(typeColors);
        var legend = svg.append("g")
            .attr("transform", "translate(0," + (summaryTop + summaryHeight + 30) + ")");

        legendData.forEach(function(entry, i) {{
            legend.append("rect")
                .attr("x", i * 130).attr("y", 0)
                .attr("width", 12).attr("height", 12)
                .attr("fill", entry[1]).attr("rx", 2);
            legend.append("text")
                .attr("x", i * 130 + 18).attr("y", 10)
                .text(entry[0]).style("font-size", "11px");
        }});

    }} catch(e) {{
        document.getElementById("chart").innerHTML = "<pre style='color:red'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height=f"{(len(sorted(df_details['date_str'].unique().tolist())) * 80) + 280}px")
    daily_chart
    return


@app.cell
def _(df_details, json_lib):
    # Get all entities involved in communications with their types
    entity_types = {}
    for _, row in df_details.iterrows():
        entity_types[row["sender_name"]] = row["sender_type"]
        entity_types[row["receiver_name"]] = row["receiver_type"]

    # Arrange entities in a circle
    import math
    entity_names = sorted(entity_types.keys())
    num_entities = len(entity_names)
    circle_nodes = []
    for i, name in enumerate(entity_names):
        angle = (2 * math.pi * i / num_entities) - math.pi / 2
        circle_nodes.append({
            "id": name,
            "type": entity_types[name],
            "x": math.cos(angle),
            "y": math.sin(angle),
        })

    # Build edges per day per hour
    day_hour_edges = {}
    for _, row in df_details.iterrows():
        date_s = row["date_str"]
        hr = int(row["hour_float"])
        key = date_s + "|" + str(hr)
        if key not in day_hour_edges:
            day_hour_edges[key] = []
        day_hour_edges[key].append({
            "source": row["sender_name"],
            "target": row["receiver_name"],
        })

    # Aggregate: count per source-target per day-hour
    from collections import Counter as Ctr
    day_hour_agg = {}
    for key, edge_list in day_hour_edges.items():
        counts = Ctr()
        for e in edge_list:
            counts[(e["source"], e["target"])] += 1
        day_hour_agg[key] = [
            {"source": s, "target": t, "weight": w}
            for (s, t), w in counts.items()
        ]

    circle_json = json_lib.dumps(circle_nodes)
    day_hour_json = json_lib.dumps(day_hour_agg)
    dates_list_json = json_lib.dumps(sorted(df_details["date_str"].unique().tolist()))
    return circle_json, day_hour_json


@app.cell
def _(df_details, mo):
    day_slider_sm = mo.ui.slider(
        start=0,
        stop=len(sorted(df_details["date_str"].unique().tolist())) - 1,
        step=1,
        value=0,
        label="Day",
        full_width=True,
        show_value=True
    )
    day_slider_sm
    return (day_slider_sm,)


@app.cell
def _(circle_json, day_hour_json, day_slider_sm, df_details, mo):
    sm_dates = sorted(df_details["date_str"].unique().tolist())
    selected_day_sm = sm_dates[day_slider_sm.value]

    circle_multiples = mo.iframe(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ margin: 0; font-family: sans-serif; }}
            .tooltip {{
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
                display: none;
                max-width: 250px;
                z-index: 100;
            }}
            .entity-node {{
                cursor: pointer;
            }}
            .entity-node:hover circle {{
                stroke-width: 3;
            }}
        </style>
    </head>
    <body>
    <div id="chart"></div>
    <div class="tooltip" id="tooltip"></div>
    <script>
    try {{
        var nodes = {circle_json};
        var dayHourEdges = {day_hour_json};
        var selectedDay = "{selected_day_sm}";
        var hours = [8,9,10,11,12,13,14];

        var typeColors = {{
            "Person": "#4169E1",
            "Organization": "#DC143C",
            "Vessel": "#FF8C00",
            "Group": "#9932CC",
            "Location": "#2E8B57"
        }};

        var cols = 4;
        var cellSize = 300;
        var pad = 15;
        var rowCount = Math.ceil(hours.length / cols);
        var margin = {{top: 60, right: 20, bottom: 60, left: 20}};
        var totalW = cols * (cellSize + pad) + margin.left + margin.right;
        var totalH = rowCount * (cellSize + pad) + margin.top + margin.bottom;
        var radius = cellSize / 2 - 40;

        var nodeMap = new Map(nodes.map(function(n) {{ return [n.id, n]; }}));

        var svg = d3.select("#chart")
            .append("svg")
            .attr("width", totalW)
            .attr("height", totalH);

        var tooltip = d3.select("#tooltip");

        // Track selected entity
        var selectedEntity = null;

        // Title
        svg.append("text")
            .attr("x", totalW / 2)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .text("Communication Networks — Day " + (parseInt("{day_slider_sm.value}") + 1) + " (" + selectedDay + ")");

        // Store references for filtering
        var allEdgePaths = [];
        var allNodeGroups = [];
        var allLabels = [];

        hours.forEach(function(hr, i) {{
            var col = i % cols;
            var row = Math.floor(i / cols);
            var ox = margin.left + col * (cellSize + pad) + cellSize / 2;
            var oy = margin.top + row * (cellSize + pad) + cellSize / 2;

            var g = svg.append("g")
                .attr("transform", "translate(" + ox + "," + oy + ")");

            // Cell background
            g.append("circle")
                .attr("r", radius + 25)
                .attr("fill", "#fafafa")
                .attr("stroke", "#ddd");

            // Hour label
            g.append("text")
                .attr("y", -radius - 12)
                .attr("text-anchor", "middle")
                .style("font-size", "13px")
                .style("font-weight", "bold")
                .text(hr + ":00");

            // Get edges for this day+hour
            var key = selectedDay + "|" + hr;
            var edges = dayHourEdges[key] || [];

            var activeNodes = new Set();
            edges.forEach(function(e) {{
                activeNodes.add(e.source);
                activeNodes.add(e.target);
            }});

            var edgeSet = new Set(edges.map(function(e) {{ return e.source + "|" + e.target; }}));
            var maxW = d3.max(edges, function(e) {{ return e.weight; }}) || 1;

            // Arrow marker
            g.append("defs").append("marker")
                .attr("id", "arr-" + hr)
                .attr("viewBox", "0 0 10 10")
                .attr("refX", 12)
                .attr("refY", 5)
                .attr("markerWidth", 5)
                .attr("markerHeight", 5)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M 0 0 L 10 5 L 0 10 Z")
                .attr("fill", "#888");

            // Draw edges
            edges.forEach(function(e) {{
                var src = nodeMap.get(e.source);
                var tgt = nodeMap.get(e.target);
                if (!src || !tgt) return;

                var x1 = src.x * radius;
                var y1 = src.y * radius;
                var x2 = tgt.x * radius;
                var y2 = tgt.y * radius;

                var dx = x2 - x1;
                var dy = y2 - y1;
                var dist = Math.sqrt(dx * dx + dy * dy) || 1;
                var hasBidi = edgeSet.has(e.target + "|" + e.source);
                var offset = hasBidi ? 20 : 0;
                var mx = (x1 + x2) / 2 + (-dy / dist) * offset;
                var my = (y1 + y2) / 2 + (dx / dist) * offset;

                var strokeW = Math.max(0.5, (e.weight / maxW) * 3);

                var path = g.append("path")
                    .attr("d", "M" + x1 + "," + y1 + " Q" + mx + "," + my + " " + x2 + "," + y2)
                    .attr("fill", "none")
                    .attr("stroke", "#999")
                    .attr("stroke-width", strokeW)
                    .attr("opacity", 0.4)
                    .attr("marker-end", "url(#arr-" + hr + ")")
                    .attr("data-source", e.source)
                    .attr("data-target", e.target);

                allEdgePaths.push(path);
            }});

            // Draw nodes
            nodes.forEach(function(n) {{
                var isActive = activeNodes.has(n.id);
                var nx = n.x * radius;
                var ny = n.y * radius;

                var ng = g.append("g")
                    .attr("class", "entity-node")
                    .attr("transform", "translate(" + nx + "," + ny + ")")
                    .attr("data-entity", n.id)
                    .attr("data-hour", hr);

                ng.append("circle")
                    .attr("r", isActive ? 7 : 4)
                    .attr("fill", typeColors[n.type] || "#999")
                    .attr("stroke", "#333")
                    .attr("stroke-width", isActive ? 1.5 : 0.5)
                    .attr("opacity", isActive ? 1 : 0.25);

                allNodeGroups.push({{group: ng, id: n.id, hour: hr, active: isActive}});

                // Label for active nodes
                if (isActive) {{
                    var angle = Math.atan2(n.y, n.x);
                    var lx = Math.cos(angle) * 18;
                    var ly = Math.sin(angle) * 18;
                    var anchor = (angle > Math.PI / 2 || angle < -Math.PI / 2) ? "end" : "start";

                    var label = g.append("text")
                        .attr("x", nx + lx)
                        .attr("y", ny + ly)
                        .attr("text-anchor", anchor)
                        .attr("dominant-baseline", "middle")
                        .style("font-size", "7px")
                        .style("fill", "#333")
                        .attr("data-entity", n.id)
                        .text(n.id.length > 12 ? n.id.substring(0, 12) + "..." : n.id);

                    allLabels.push(label);
                }}
            }});

            // Message count
            var totalMsgs = edges.reduce(function(sum, e) {{ return sum + e.weight; }}, 0);
            g.append("text")
                .attr("y", radius + 18)
                .attr("text-anchor", "middle")
                .style("font-size", "10px")
                .style("fill", "#888")
                .text(totalMsgs + " messages");
        }});

        // Click handler on all entity nodes
        svg.selectAll(".entity-node").on("click", function(event) {{
            var clickedEntity = d3.select(this).attr("data-entity");

            if (selectedEntity === clickedEntity) {{
                // Deselect: reset everything
                selectedEntity = null;
                allEdgePaths.forEach(function(p) {{
                    p.attr("opacity", 0.4).attr("stroke", "#999");
                }});
                svg.selectAll(".entity-node circle")
                    .attr("opacity", function() {{
                        return 0.6;
                    }});
            }} else {{
                selectedEntity = clickedEntity;

                // Dim all edges
                allEdgePaths.forEach(function(p) {{
                    var src = p.attr("data-source");
                    var tgt = p.attr("data-target");
                    if (src === clickedEntity || tgt === clickedEntity) {{
                        p.attr("opacity", 0.8).attr("stroke", "#ff5555").attr("stroke-width", 2.5);
                    }} else {{
                        p.attr("opacity", 0.05).attr("stroke", "#ccc");
                    }}
                }});

                // Dim nodes not connected
                svg.selectAll(".entity-node circle")
                    .attr("opacity", 0.15);

                // Highlight clicked entity nodes
                svg.selectAll(".entity-node")
                    .filter(function() {{ return d3.select(this).attr("data-entity") === clickedEntity; }})
                    .select("circle")
                    .attr("opacity", 1)
                    .attr("stroke-width", 3)
                    .attr("stroke", "#ff5555");

                // Also highlight connected entities
                allEdgePaths.forEach(function(p) {{
                    var src = p.attr("data-source");
                    var tgt = p.attr("data-target");
                    if (src === clickedEntity) {{
                        svg.selectAll(".entity-node")
                            .filter(function() {{ return d3.select(this).attr("data-entity") === tgt; }})
                            .select("circle").attr("opacity", 0.8);
                    }}
                    if (tgt === clickedEntity) {{
                        svg.selectAll(".entity-node")
                            .filter(function() {{ return d3.select(this).attr("data-entity") === src; }})
                            .select("circle").attr("opacity", 0.8);
                    }}
                }});
            }}
        }});

        // Hover tooltip on nodes
        svg.selectAll(".entity-node")
            .on("mouseover", function(event) {{
                var ent = d3.select(this).attr("data-entity");
                tooltip.style("display", "block")
                    .html("<strong>" + ent + "</strong><br/>" + (nodeMap.get(ent).type))
                    .style("left", (event.pageX + 12) + "px")
                    .style("top", (event.pageY - 20) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.style("display", "none");
            }});

        // Legend
        var legendG = svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + (totalH - 40) + ")");

        var legendItems = Object.entries(typeColors);
        legendItems.forEach(function(entry, i) {{
            legendG.append("circle")
                .attr("cx", i * 130 + 8)
                .attr("cy", 0)
                .attr("r", 6)
                .attr("fill", entry[1]);
            legendG.append("text")
                .attr("x", i * 130 + 20)
                .attr("y", 4)
                .text(entry[0])
                .style("font-size", "11px");
        }});

        // Click instruction
        legendG.append("text")
            .attr("x", totalW - margin.right - margin.left - 20)
            .attr("y", 4)
            .attr("text-anchor", "end")
            .style("font-size", "10px")
            .style("fill", "#888")
            .text("Click an entity to filter. Click again to reset.");

    }} catch(e) {{
        document.getElementById("chart").innerHTML = "<pre style='color:red'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height="720px")

    mo.vstack([
        circle_multiples,
        mo.md(f"**Day {day_slider_sm.value + 1}** — {selected_day_sm}"),
        day_slider_sm
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
