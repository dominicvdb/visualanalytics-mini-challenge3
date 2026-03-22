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
    import networkx as nx
    from collections import Counter
    import time
    return Counter, alt, datetime, json_graph, json_lib, mo, nx, pd, time


@app.cell
def _(json_graph, json_lib):
    # Load graph
    with open("data/MC3_graph.json", "r") as g:
        json_data = json_lib.load(g)
    G = json_graph.node_link_graph(json_data, edges="edges")
    return (G,)


@app.cell
def _(G, datetime, pd):
    # Extract all communications
    comms = []
    for n, a in G.nodes(data=True):
        if a.get("sub_type") == "Communication":
            ts = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
            comms.append({
                "node_id": n,
                "timestamp": ts,
                "hour": ts.hour,
                "date": ts.date(),
                "content": a.get("content", ""),
            })
    df_comms = pd.DataFrame(comms)
    return (df_comms,)


@app.cell
def _(G, df_comms, json_lib, pd):
    # Extract sender/receiver details for each communication
    comm_details = []
    for c_node in df_comms["node_id"]:
        attrs_c = G.nodes[c_node]
        ts_c = attrs_c["timestamp"]
        content_c = attrs_c.get("content", "")

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
                "content": content_c,
                "sender_name": G.nodes[sender].get("name", sender),
                "sender_type": G.nodes[sender].get("sub_type", ""),
                "receiver_name": G.nodes[receiver].get("name", receiver),
                "receiver_type": G.nodes[receiver].get("sub_type", ""),
            })

    df_details = pd.DataFrame(comm_details)
    df_details["ts"] = pd.to_datetime(df_details["timestamp"])
    df_details["date_str"] = df_details["ts"].dt.date.astype(str)
    df_details["hour_float"] = df_details["ts"].dt.hour + df_details["ts"].dt.minute / 60
    df_details["ts_str"] = df_details["ts"].astype(str)

    print(f"{len(df_details)} communications with sender/receiver info")
    df_details.head()
    return (df_details,)


@app.cell
def _(mo):
    mo.md("## Step 1: Classify Message Intents with OpenAI")
    return


@app.cell
def _(json_lib, mo, time):
    from openai import OpenAI

    # Set your API key here
    client = OpenAI(api_key="YOUR_API_KEY_HERE")

    intent_labels = [
        "requesting permission",
        "reporting status",
        "coordinating activity",
        "issuing warning",
        "concealing information",
        "giving orders",
        "sharing observations",
        "making excuses",
        "negotiating",
        "socializing"
    ]

    labels_str = ", ".join(intent_labels)

    def classify_batch(messages):
        numbered = "\n".join([f"[{i}] {msg[:300]}" for i, msg in enumerate(messages)])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": f"""You are a message intent classifier.
For each numbered message, classify its PRIMARY intent into exactly one of these categories:
{labels_str}

Respond ONLY with a JSON array of objects, one per message, in order:
[{{"index": 0, "intent": "...", "confidence": 0.0-1.0}}, ...]
No other text."""},
                {"role": "user", "content": numbered}
            ]
        )

        try:
            result = json_lib.loads(response.choices[0].message.content)
            return result
        except:
            return [{"index": i, "intent": "unknown", "confidence": 0} for i in range(len(messages))]

    mo.md("OpenAI client ready. Run the next cell to classify messages.")
    return classify_batch, client, intent_labels


@app.cell
def _(classify_batch, df_details, json_lib, mo, pd, time):
    # Run intent classification in batches
    batch_size = 20
    all_intents = []

    messages_list = df_details["content"].tolist()

    for start in range(0, len(messages_list), batch_size):
        end = min(start + batch_size, len(messages_list))
        batch_msgs = messages_list[start:end]

        results = classify_batch(batch_msgs)

        for j, res in enumerate(results):
            row = df_details.iloc[start + j]
            all_intents.append({
                "node_id": row["node_id"],
                "sender_name": row["sender_name"],
                "receiver_name": row["receiver_name"],
                "sender_type": row["sender_type"],
                "receiver_type": row["receiver_type"],
                "date_str": row["date_str"],
                "hour_float": row["hour_float"],
                "timestamp": row["timestamp"],
                "intent": res.get("intent", "unknown"),
                "confidence": res.get("confidence", 0),
                "content": batch_msgs[j],
            })

        print(f"Processed {end}/{len(messages_list)}")
        time.sleep(0.5)

    df_intents = pd.DataFrame(all_intents)

    # Save to CSV so you don't have to re-run the API
    df_intents.to_csv("data/intents_classified.csv", index=False)
    print(f"Done! {len(df_intents)} messages classified. Saved to data/intents_classified.csv")
    df_intents.head()
    return (df_intents,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Visualizations
    
    If you already have a saved `intents_classified.csv`, uncomment the cell below 
    and comment out the API cell above to load from file instead of re-running the API.
    """)
    return


# @app.cell
# def _(pd):
#     # Load pre-classified intents from CSV
#     df_intents = pd.read_csv("data/intents_classified.csv")
#     print(f"Loaded {len(df_intents)} classified messages")
#     df_intents.head()
#     return (df_intents,)


@app.cell
def _(alt, df_intents, mo):
    # Intent distribution overview
    intent_bar = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("count()", title="Number of Messages"),
        y=alt.Y("intent:N", title="Intent", sort="-x"),
        color=alt.Color("intent:N", legend=None),
    ).properties(
        title="Overall Intent Distribution",
        width=600,
        height=300
    )

    mo.vstack([
        mo.md("### Intent Distribution"),
        intent_bar
    ])
    return


@app.cell
def _(alt, df_intents, mo):
    # Intent by entity — who does what?
    intent_entity = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("count()", title="Count"),
        y=alt.Y("sender_name:N", title="Entity", sort="-x"),
        color=alt.Color("intent:N", title="Intent"),
    ).properties(
        title="Message Intents by Sender",
        width=700,
        height=500
    )

    mo.vstack([
        mo.md("### Intent by Entity"),
        intent_entity
    ])
    return


@app.cell
def _(df_intents, mo):
    # Create intent filter dropdown
    unique_intents = ["All"] + sorted(df_intents["intent"].unique().tolist())

    intent_dropdown = mo.ui.dropdown(
        options=unique_intents,
        value="All",
        label="Filter by Intent"
    )
    return intent_dropdown, unique_intents


@app.cell
def _(df_intents, intent_dropdown, json_lib, mo):
    # Prepare data for D3 daily chart, filtered by intent
    selected_intent = intent_dropdown.value

    if selected_intent == "All":
        df_filtered = df_intents.copy()
    else:
        df_filtered = df_intents[df_intents["intent"] == selected_intent].copy()

    unique_dates_intent = json_lib.dumps(sorted(df_intents["date_str"].unique().tolist()))

    # Intent color map
    intent_colors = {
        "requesting permission": "#e6194b",
        "reporting status": "#3cb44b",
        "coordinating activity": "#ffe119",
        "issuing warning": "#f58231",
        "concealing information": "#911eb4",
        "giving orders": "#42d4f4",
        "sharing observations": "#f032e6",
        "making excuses": "#bfef45",
        "negotiating": "#fabed4",
        "socializing": "#469990",
        "unknown": "#999999"
    }

    # Add intent color to each record
    records = []
    for _, row in df_filtered.iterrows():
        records.append({
            "node_id": row["node_id"],
            "sender_name": row["sender_name"],
            "receiver_name": row["receiver_name"],
            "sender_type": row["sender_type"],
            "receiver_type": row["receiver_type"],
            "date_str": row["date_str"],
            "hour_float": row["hour_float"],
            "timestamp": row["timestamp"],
            "intent": row["intent"],
            "confidence": row["confidence"],
            "content": str(row["content"])[:150],
            "intent_color": intent_colors.get(row["intent"], "#999"),
        })

    filtered_json = json_lib.dumps(records)
    intent_colors_json = json_lib.dumps(intent_colors)
    msg_count = len(df_filtered)

    # Build the D3 chart
    daily_intent_chart = mo.iframe(f"""
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
                max-width: 350px;
            }}
        </style>
    </head>
    <body>
    <div id="chart"></div>
    <div class="tooltip" id="tooltip"></div>
    <script>
    try {{
        var data = {filtered_json};
        var allDates = {unique_dates_intent};
        var intentColors = {intent_colors_json};

        var typeColors = {{
            "Person": "#7B68EE",
            "Organization": "#DC143C",
            "Vessel": "#00CED1",
            "Group": "#FF8C00",
            "Location": "#4169E1"
        }};

        var margin = {{top: 50, right: 30, bottom: 100, left: 80}};
        var rowHeight = 80;
        var summaryHeight = 80;
        var width = 1100 - margin.left - margin.right;
        var height = allDates.length * rowHeight;
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
            .domain(allDates)
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
        allDates.forEach(function(date, i) {{
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
        allDates.forEach(function(date) {{
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

        // Draw dots — colored by INTENT instead of entity type
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
                        d3.select(this).select(".outline")
                            .attr("stroke", "#333").attr("stroke-width", 2);
                        tooltip.style("display", "block")
                            .html(
                                "<strong>" + d.sender_name + " → " + d.receiver_name + "</strong><br/>"
                                + d.timestamp + "<br/>"
                                + "<strong>Intent:</strong> " + d.intent 
                                + " (" + (d.confidence * 100).toFixed(0) + "%)<br/>"
                                + "<span style='color:" + (typeColors[d.sender_type]||"#999") + "'>■</span> "
                                + d.sender_type + " → "
                                + "<span style='color:" + (typeColors[d.receiver_type]||"#999") + "'>■</span> "
                                + d.receiver_type + "<br/>"
                                + "<em>" + d.content + "</em>"
                            )
                            .style("left", (event.pageX + 12) + "px")
                            .style("top", (event.pageY - 20) + "px");
                    }})
                    .on("mouseout", function() {{
                        d3.select(this).select(".outline").attr("stroke", "none");
                        tooltip.style("display", "none");
                    }});

                // Left half — sender entity type color
                g.append("rect")
                    .attr("x", px).attr("y", py)
                    .attr("width", dotSize / 2).attr("height", dotSize)
                    .attr("fill", typeColors[d.sender_type] || "#999");

                // Right half — receiver entity type color
                g.append("rect")
                    .attr("x", px + dotSize / 2).attr("y", py)
                    .attr("width", dotSize / 2).attr("height", dotSize)
                    .attr("fill", typeColors[d.receiver_type] || "#999");

                // Intent border color
                g.append("rect")
                    .attr("class", "outline")
                    .attr("x", px).attr("y", py)
                    .attr("width", dotSize).attr("height", dotSize)
                    .attr("fill", "none")
                    .attr("rx", 1)
                    .attr("stroke", d.intent_color)
                    .attr("stroke-width", 2);
            }});
        }});

        // === Summary density at bottom ===
        var summaryTop = height + 10;

        svg.append("rect")
            .attr("x", 0).attr("y", summaryTop)
            .attr("width", width).attr("height", summaryHeight)
            .attr("fill", "#f0f0f0")
            .attr("stroke", "#ddd");

        svg.append("text")
            .attr("x", -10)
            .attr("y", summaryTop + summaryHeight / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "middle")
            .style("font-size", "11px")
            .style("font-weight", "bold")
            .text("All");

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

        // Bottom X axis
        svg.append("g")
            .attr("transform", "translate(0," + (summaryTop + summaryHeight) + ")")
            .call(d3.axisBottom(x)
                .tickValues(tickVals)
                .tickFormat(function(d) {{ return d + ":00"; }}))
            .selectAll("text")
            .style("font-size", "11px");

        // Entity type legend
        var legendData = Object.entries(typeColors);
        var legend = svg.append("g")
            .attr("transform", "translate(0," + (summaryTop + summaryHeight + 25) + ")");

        legend.append("text")
            .attr("x", 0).attr("y", 0)
            .text("Entity Type:").style("font-size", "11px").style("font-weight", "bold");

        legendData.forEach(function(entry, i) {{
            legend.append("rect")
                .attr("x", 90 + i * 120).attr("y", -9)
                .attr("width", 12).attr("height", 12)
                .attr("fill", entry[1]).attr("rx", 2);
            legend.append("text")
                .attr("x", 90 + i * 120 + 16).attr("y", 0)
                .text(entry[0]).style("font-size", "10px");
        }});

        // Intent legend
        var intentLegendData = Object.entries(intentColors);
        var intentLegend = svg.append("g")
            .attr("transform", "translate(0," + (summaryTop + summaryHeight + 50) + ")");

        intentLegend.append("text")
            .attr("x", 0).attr("y", 0)
            .text("Intent (border):").style("font-size", "11px").style("font-weight", "bold");

        intentLegendData.forEach(function(entry, i) {{
            var col = i % 5;
            var row = Math.floor(i / 5);
            intentLegend.append("rect")
                .attr("x", 110 + col * 170).attr("y", -9 + row * 18)
                .attr("width", 10).attr("height", 10)
                .attr("fill", "none").attr("stroke", entry[1]).attr("stroke-width", 2);
            intentLegend.append("text")
                .attr("x", 110 + col * 170 + 14).attr("y", row * 18)
                .text(entry[0]).style("font-size", "9px");
        }});

    }} catch(e) {{
        document.getElementById("chart").innerHTML = "<pre style='color:red'>" + e.message + "\\n" + e.stack + "</pre>";
    }}
    </script>
    </body>
    </html>
    """, width="100%", height=f"{(len(sorted(df_intents['date_str'].unique().tolist())) * 80) + 350}px")

    mo.vstack([
        mo.md(f"### Daily Communication Patterns — Intent View ({selected_intent}) — {msg_count} messages"),
        intent_dropdown,
        daily_intent_chart,
    ])
    return


@app.cell
def _(alt, df_intents, mo):
    # Intent heatmap: date x intent
    intent_heatmap = alt.Chart(df_intents).mark_rect().encode(
        x=alt.X("date_str:O", title="Date"),
        y=alt.Y("intent:N", title="Intent"),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), title="Count"),
        tooltip=["date_str", "intent", "count()"]
    ).properties(
        title="Intent Frequency Over Time",
        width=700,
        height=350
    )

    mo.vstack([
        mo.md("### Intent Heatmap — Date × Intent"),
        intent_heatmap
    ])
    return


@app.cell
def _(alt, df_intents, mo):
    # Intent over hours — when do different intents happen?
    intent_hour = alt.Chart(df_intents).mark_bar().encode(
        x=alt.X("hours(timestamp):O", title="Hour of Day"),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("intent:N", title="Intent"),
    ).properties(
        title="Intent Distribution by Hour",
        width=700,
        height=350
    )

    mo.vstack([
        mo.md("### When do different intents occur?"),
        intent_hour
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
