import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # <center> Mini-Challenge 3 </center>

    ## **Team Members**

    1. AmanDeep Singh
    2. Dominic van den Bungelaar
    3. Kim Wilmink

    ## **Background**
    Over the past decade, the community of Oceanus has faced numerous transformations and challenges evolving from its fishing-centric origins. Following major crackdowns on illegal fishing activities, suspects have shifted investments into more regulated sectors such as the ocean tourism industry, resulting in growing tensions. This increased tourism has recently attracted the likes of international pop star Sailor Shift, who announced plans to film a music video on the island.

    Clepper Jessen, a former analyst at FishEye and now a seasoned journalist for the Hacklee Herald, has been keenly observing these rising tensions. Recently, he turned his attention towards the temporary closure of Nemo Reef. By listening to radio communications and utilizing his investigative tools, Clepper uncovered a complex web of expedited approvals and secretive logistics. These efforts revealed a story involving high-level Oceanus officials, Sailor Shift’s team, local influential families, and local conservationist group The Green Guardians, pointing towards a story of corruption and manipulation.

    Our task is to develop new and novel visualizations and visual analytics approaches to help Clepper get to the bottom of this story.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Tasks & Questions**

    Clepper diligently recorded all intercepted radio communications over the last two weeks. With the help of his intern, they have analyzed their content to identify important events and relationships between key players. The result is a knowledge graph describing the last two weeks on Oceanus. Clepper and his intern have spent a large amount of time generating this knowledge graph, and they would now like some assistance using it to answer the following questions.

    1. Clepper found that messages frequently came in at around the same time each day.
        - Develop a graph-based visual analytics approach to identify any daily temporal patterns in communications.
        - How do these patterns shift over the two weeks of observations?
        - Focus on a specific entity and use this information to determine who has influence over them.

    2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.
        - Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.
        - Are there groups that are more closely associated? If so, what are the topic areas that are predominant for each group?
              - For example, these groupings could be related to: Environmentalism (known associates of Green Guardians), Sailor Shift, and fishing/leisure vessels.

    3. It was noted by Clepper’s intern that some people and vessels are using pseudonyms to communicate.
        - Expanding upon your prior visual analytics, determine who is using pseudonyms to communicate, and what these pseudonyms are.
              - Some that Clepper has already identified include: “Boss”, and “The Lookout”, but there appear to be many more.
              - To complicate the matter, pseudonyms may be used by multiple people or vessels.
        - Describe how your visualizations make it easier for Clepper to identify common entities in the knowledge graph.
        - How does your understanding of activities change given your understanding of pseudonyms?

    4. Clepper suspects that Nadia Conti, who was formerly entangled in an illegal fishing scheme, may have continued illicit activity within Oceanus.

        - Through visual analytics, provide evidence that Nadia is, or is not, doing something illegal.
        - Summarize Nadia’s actions visually. Are Clepper’s suspicions justified?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Import the libraries
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import numpy as np
    from collections import defaultdict, Counter
    from datetime import datetime
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import networkx as nx
    return Counter, defaultdict, go, json, mo, np, nx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2.1: Understanding Interactions Between Vessels and People

    - *2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.*
        - *a. Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.*

    To answer these questions, I create interactive visualization to explore:
    1. **Communication Network** - Who talks to whom and how frequently?
    2. **Relationship Network** - The formal relationships such as (Colleagues, Operates, Reports, etc.)
    3. **Entity Profiles** - Analyse and dive deep into the individual actors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Preparation & Graph Creation

    ### 1.1 Loading and Structuring the Knowledge Graphs

    In this part I have loaded the `MC3_graph.json` file, extracted all nodes and edges and built a lookup dictionaries for efficient querying. I have categorized nodes into:
    - *Persons*
    - *Vessels*
    - *Organizations*
    - *Groups*
    - *Locations*

    which helps for filtering and analysis by entity type.
    """)
    return


@app.cell
def _(json):
    # At first we will load the knowledge graph data that is provided
    with open('data/MC3_graph.json', 'r') as f:
        graph_data = json.load(f)

    # Then we will build a lookup dictionaries
    nodes_by_id = {n['id']: n for n in graph_data['nodes']}

    # Then we will categorize all the entities
    persons = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Person'}
    vessels = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Vessel'}
    organizations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Organization'}
    groups = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Group'}
    locations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Location'}

    # In here we have listed all the entities of interest (especially for network analysis)
    all_entities = {**persons, **vessels, **organizations, **groups}
    entity_ids = set(all_entities.keys())

    print(f"Loaded: {len(persons)} persons, {len(vessels)} vessels, {len(organizations)} organizations, {len(groups)} groups")
    return all_entities, entity_ids, graph_data, nodes_by_id


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Building the Communication Structures & Extracted the Formal Relationships

    In this section I have extracted all nodes of subtypes `Communication` identified
    - **Senders** → via `sent` edges
    - **Receivers** via `received` edges

    Then I constructed a ***Communication Matrix*** which stores (**Timestamp, Content, Communication ID**). This matrix helps with creating ***network graph, heatmap, individual profiles and other statistics***

    After that, I have extracted the `Relationship` nodes and identified the connected entities while distinguishing between
    - **Bidirectional relationships (Collegues, Friends)**
    - **Directional relationships (Reports, Operates, etc)**

    and created a structured dataset with `type`, `entity1`, `entity2`, `bidirectional` and `rel_id` which helps with the structural network analysis.
    """)
    return


@app.cell
def _(defaultdict, entity_ids, graph_data):
    # I have build in here a edge lookup structures
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)
    for edge in graph_data['edges']:
        edges_to[edge['target']].append(edge)
        edges_from[edge['source']].append(edge)

    # After that I have extracted the Communication events
    comm_events = [n for n in graph_data['nodes'] if n.get('sub_type') == 'Communication']

    # Then I have build the communication matrix to answer the question like (who talks to whom)
    comm_matrix = defaultdict(lambda: defaultdict(list))  # This store the actual communications

    for comm in comm_events:
        comm_id = comm['id']
        timestamp = comm.get('timestamp', '')
        content = comm.get('content', '')

        # In here I find the senders (edges TO communication with type 'sent')
        senders = [e['source'] for e in edges_to[comm_id] if e.get('type') == 'sent']
        # and In here I find receivers (edges FROM communication with type 'received')
        receivers = [e['target'] for e in edges_from[comm_id] if e.get('type') == 'received']

        for sender in senders:
            for receiver in receivers:
                if sender in entity_ids or receiver in entity_ids:
                    comm_matrix[sender][receiver].append({
                        'timestamp': timestamp,
                        'content': content,
                        'comm_id': comm_id
                    })

    # After all that, I extracted all the Relationship nodes
    relationships = [n for n in graph_data['nodes'] if n['type'] == 'Relationship']

    # Then build the relationship data structure
    relationship_data = []
    for rel in relationships:
        rel_id = rel['id']
        rel_type = rel['sub_type']

        # I have got the connected entities
        sources = [e['source'] for e in edges_to[rel_id] if e['source'] in entity_ids]
        targets = [e['target'] for e in edges_from[rel_id] if e['target'] in entity_ids]

        # and for bidirectional relationships such as (Colleagues, Friends), both parties are my sources
        if rel_type in ['Colleagues', 'Friends']:
            if len(sources) >= 2:
                relationship_data.append({
                    'type': rel_type,
                    'entity1': sources[0],
                    'entity2': sources[1],
                    'bidirectional': True,
                    'rel_id': rel_id
                })
        else:
            # These are the directional relationships
            for s in sources:
                for t in targets:
                    relationship_data.append({
                        'type': rel_type,
                        'entity1': s,
                        'entity2': t,
                        'bidirectional': False,
                        'rel_id': rel_id
                    })

    print(f"Extracted {len(comm_events)} communications and {len(relationship_data)} relationships")
    return comm_matrix, relationship_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2. Communication Network**
    In this visualization, I will show the ***communication flow*** between all entities *(persons, vessels and organizations),* in other words **who talks to whom and how often**. The edge thickness shows the ***communication frequency.***
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    At first I have create an **interaction filtering** with
    - Entity Type Selection
    - Minimum Communication Threshold
    - Layout selection
    """)
    return


@app.cell
def _(mo):
    # At first I have created the controls for the network visualization
    node_type_filter = mo.ui.multiselect(
        options=['Person', 'Vessel', 'Organization', 'Group'],
        value=['Person', 'Vessel', 'Organization'],
        label="Show Entity Types:"
    )

    min_comm_slider = mo.ui.slider(
        start=1, stop=20, value=2, step=1,
        label="Minimum Communications to Show Edge:"
    )

    layout_select = mo.ui.dropdown(
        options=['spring', 'circular', 'kamada_kawai'],
        value='spring',
        label="Network Layout:"
    )

    mo.hstack([node_type_filter, min_comm_slider, layout_select], justify='start', gap=2)
    return layout_select, min_comm_slider, node_type_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following code, you can see that I have build a **directed NetworkX graph** with edge weight that are the number of messages, used with multiple layouts (Spring, Circular and Kamada-Kawai). For the visual encoding I have used the node size which corresponds to degree of (communication activity), node color for the entity type, the edge thickness for the message frequency.
    """)
    return


@app.cell
def _(
    all_entities,
    comm_matrix,
    go,
    layout_select,
    min_comm_slider,
    node_type_filter,
    nodes_by_id,
    nx,
):
    # At first I have build filtered communication network
    def build_comm_network(entity_types, min_comms, layout):
        # Filter entities by type
        filtered_entities = {
            eid: e for eid, e in all_entities.items() 
            if e.get('sub_type') in entity_types
        }

        # Then a created the NetworkX graph
        G = nx.DiGraph()

        # I have added the nodes
        for eid, entity in filtered_entities.items():
            G.add_node(eid, 
                      label=entity.get('name', eid),
                      sub_type=entity.get('sub_type'))

        # I have added the edges based on communications
        edge_weights = {}
        for sender in comm_matrix:
            if sender not in filtered_entities:
                continue
            for receiver, comms in comm_matrix[sender].items():
                if receiver not in filtered_entities:
                    continue
                weight = len(comms)
                if weight >= min_comms:
                    G.add_edge(sender, receiver, weight=weight)
                    edge_weights[(sender, receiver)] = weight

        # From here I have the Calculate layout of (spring, circular and kamada-kawai)
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        return G, pos, edge_weights

    G_comm, pos_comm, edge_weights_comm = build_comm_network(
        node_type_filter.value, 
        min_comm_slider.value,
        layout_select.value
    )

    # I have integrated the Plotly figure (for interaction)
    # Then I have manually added color mapping for entity types
    color_map = {
        'Person': '#4ECDC4',       # Teal
        'Vessel': '#FF6B6B',       # Coral red
        'Organization': '#95E1D3', # Mint
        'Group': '#F38181'         # Salmon
    }

    # In here I have the edge traces
    edge_traces = []
    for (u, v), weight in edge_weights_comm.items():
        x0, y0 = pos_comm[u]
        x1, y1 = pos_comm[v]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=min(weight / 2, 8),
                color='rgba(150, 150, 150, 0.4)'
            ),
            hoverinfo='text',
            text=f"{u} → {v}: {weight} messages",
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # In here I have the node traces (grouped by type so I can place it in the legend)
    node_traces = []
    for entity_type in node_type_filter.value:
        nodes_of_type = [n for n in G_comm.nodes() 
                        if nodes_by_id.get(n, {}).get('sub_type') == entity_type]

        if not nodes_of_type:
            continue

        x_vals = [pos_comm[n][0] for n in nodes_of_type]
        y_vals = [pos_comm[n][1] for n in nodes_of_type]

        # In here I calculated the node sizes based on degree
        sizes = [max(15, min(50, G_comm.degree(n) * 5)) for n in nodes_of_type]

        # Added the Hover text
        hover_texts = []
        for n in nodes_of_type:
            in_deg = G_comm.in_degree(n)
            out_deg = G_comm.out_degree(n)
            hover_texts.append(f"<b>{n}</b><br>Type: {entity_type}<br>Sent to: {out_deg} entities<br>Received from: {in_deg} entities")

        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=color_map[entity_type],
                line=dict(width=2, color='white')
            ),
            text=[n if len(n) < 15 else n[:12]+'...' for n in nodes_of_type],
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=hover_texts,
            name=entity_type
        )
        node_traces.append(trace)

    # Then finally I have combined all the traces
    fig_comm_network = go.Figure(data=edge_traces + node_traces)

    fig_comm_network.update_layout(
        title=dict(
            text='<b>Communication Network</b><br><sup>Who talks to whom? Edge thickness = message frequency</sup>',
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Show the Communication Network Graph
    fig_comm_network
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **3 Relationship Frequency Matrix (Heatmap)**

    This heatmap provides quite a detailed overview of the the ***communication intesity*** between the entities, the darker blue the cells are the more frequent the communication is.
    """)
    return


@app.cell
def _(mo):
    # I have added the Entity selection interaction filter for heatmap
    heatmap_type_filter = mo.ui.multiselect(
        options=['Person', 'Vessel', 'Organization'],
        value=['Person', 'Vessel'],
        label="Include Entity Types in Heatmap:"
    )
    heatmap_type_filter
    return (heatmap_type_filter,)


@app.cell
def _(all_entities, comm_matrix, go, heatmap_type_filter, np):
    # In here I create the communication count matrix for heatmap
    def build_heatmap_data(entity_types):
        # At first I Filter the entities
        filtered = [eid for eid, e in all_entities.items() 
                   if e.get('sub_type') in entity_types]

        # Then I sort them by entity type and then by name
        filtered = sorted(filtered, key=lambda x: (all_entities[x].get('sub_type', ''), x))

        # At last I build the matrix
        n = len(filtered)
        matrix = np.zeros((n, n))

        for i, sender in enumerate(filtered):
            for j, receiver in enumerate(filtered):
                if sender in comm_matrix and receiver in comm_matrix[sender]:
                    matrix[i][j] = len(comm_matrix[sender][receiver])

        return filtered, matrix

    entities_hm, matrix_hm = build_heatmap_data(heatmap_type_filter.value)

    # After that I create the labels with entity type indicators
    labels_hm = []
    for eid in entities_hm:
        etype = all_entities[eid].get('sub_type', '?')[0]  # This is for the First letter
        labels_hm.append(f"[{etype}] {eid}")

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_hm,
        x=labels_hm,
        y=labels_hm,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Messages: %{z}<extra></extra>'
    ))

    fig_heatmap.update_layout(
        title=dict(
            text='<b>Communication Frequency Matrix</b><br><sup>Row = Sender, Column = Receiver. [P]=Person, [V]=Vessel, [O]=Organization</sup>',
            x=0.5
        ),
        xaxis=dict(title='Receiver', tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(title='Sender', tickfont=dict(size=8)),
        height=800,
        width=900,
        margin=dict(l=150, r=50, t=100, b=150)
    )

    # Display the frequency heatmap matrix
    fig_heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **4. Formal Relationship Network** - TODO

    Except of the communications, entities also ahve ***formal relationships*** such as Colleagues, Operates, Report, Coordinates, etc and this network will display those structural connections and relationships.
    """)
    return


@app.cell
def _(mo):
    # I added the filter with all the relationship options
    rel_type_filter = mo.ui.multiselect(
        options=['Colleagues', 'Operates', 'Reports', 'Coordinates', 'Suspicious', 'Friends', 'Unfriendly', 'Jurisdiction', 'AccessPermission'],
        value=['Colleagues', 'Operates', 'Reports', 'Suspicious'],
        label="Show Relationship Types:"
    )
    rel_type_filter
    return (rel_type_filter,)


@app.cell
def _(
    Counter,
    all_entities,
    go,
    nodes_by_id,
    nx,
    rel_type_filter,
    relationship_data,
):
    # Build relationship network
    def build_rel_network(rel_types):
        G = nx.Graph()  # Undirected for visualization

        # Filter relationships
        filtered_rels = [r for r in relationship_data if r['type'] in rel_types]

        # Count relationship types between entity pairs
        edge_rels = Counter()
        for rel in filtered_rels:
            e1, e2 = rel['entity1'], rel['entity2']
            if e1 in all_entities and e2 in all_entities:
                key = tuple(sorted([e1, e2]))
                edge_rels[(key, rel['type'])] += 1

        # Add nodes and edges
        for (key, rel_type), count in edge_rels.items():
            e1, e2 = key
            G.add_node(e1, sub_type=all_entities[e1].get('sub_type'))
            G.add_node(e2, sub_type=all_entities[e2].get('sub_type'))

            if G.has_edge(e1, e2):
                G[e1][e2]['types'].append(rel_type)
            else:
                G.add_edge(e1, e2, types=[rel_type])

        return G

    G_rel = build_rel_network(rel_type_filter.value)

    # Layout
    if len(G_rel.nodes()) > 0:
        pos_rel = nx.spring_layout(G_rel, k=3, iterations=50, seed=42)
    else:
        pos_rel = {}

    # Color map for relationship types
    rel_color_map = {
        'Colleagues': '#2ECC71',     # Green
        'Operates': '#3498DB',       # Blue
        'Reports': '#9B59B6',        # Purple
        'Coordinates': '#F39C12',    # Orange
        'Suspicious': '#E74C3C',     # Red
        'Friends': '#1ABC9C',        # Teal
        'Unfriendly': '#C0392B',     # Dark red
        'Jurisdiction': '#34495E',   # Dark gray
        'AccessPermission': '#7F8C8D' # Gray
    }

    # Node type color map
    node_color_map = {
        'Person': '#4ECDC4',
        'Vessel': '#FF6B6B',
        'Organization': '#95E1D3',
        'Group': '#F38181'
    }

    # Create edge traces for each relationship type
    rel_edge_traces = []
    for rel_type in rel_type_filter.value:
        x_edges, y_edges = [], []
        for u, v, data in G_rel.edges(data=True):
            if rel_type in data['types']:
                x0, y0 = pos_rel[u]
                x1, y1 = pos_rel[v]
                x_edges.extend([x0, x1, None])
                y_edges.extend([y0, y1, None])

        if x_edges:
            trace = go.Scatter(
                x=x_edges, y=y_edges,
                mode='lines',
                line=dict(width=2, color=rel_color_map.get(rel_type, 'gray')),
                name=rel_type,
                hoverinfo='none'
            )
            rel_edge_traces.append(trace)

    # Node traces
    rel_node_traces = []
    for ntype in ['Person', 'Vessel', 'Organization', 'Group']:
        nodes_of_type = [n for n in G_rel.nodes() 
                        if nodes_by_id.get(n, {}).get('sub_type') == ntype]
        if not nodes_of_type:
            continue

        x_vals = [pos_rel[n][0] for n in nodes_of_type]
        y_vals = [pos_rel[n][1] for n in nodes_of_type]

        # Hover text with relationship info
        hover_texts = []
        for n in nodes_of_type:
            neighbors = list(G_rel.neighbors(n))
            rel_summary = []
            for nb in neighbors:
                types = G_rel[n][nb]['types']
                rel_summary.append(f"  • {nb}: {', '.join(types)}")
            hover_text = f"<b>{n}</b> ({ntype})<br>Relationships:<br>" + "<br>".join(rel_summary[:10])
            if len(rel_summary) > 10:
                hover_text += f"<br>  ...and {len(rel_summary)-10} more"
            hover_texts.append(hover_text)

        trace = go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_color_map[ntype],
                line=dict(width=2, color='white')
            ),
            text=nodes_of_type,
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=hover_texts,
            name=f'{ntype}s',
            legendgroup='nodes'
        )
        rel_node_traces.append(trace)

    fig_rel_network = go.Figure(data=rel_edge_traces + rel_node_traces)

    fig_rel_network.update_layout(
        title=dict(
            text='<b>Formal Relationship Network</b><br><sup>Structural connections between entities (colored by relationship type)</sup>',
            x=0.5
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig_rel_network
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **5. Individual Communication Profiles (Entity Dive Deep)**
    """)
    return


@app.cell
def _(all_entities, mo):
    entity_selector = mo.ui.dropdown(
        options=sorted(all_entities.keys()),
        value='Nadia Conti',
        label="Select Entity to Analyze:"
    )
    entity_selector
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## **6. Communication Timeline**
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **7. Key Statistics**
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Findings for Question 2.1**

    Based on the visual analytics executed above, several insights emerge about the interaction and relationships.
    1.
    2.
    3.
    4.
    5.

    (Continue with 2.2 For Community detection and topic analysis)
    """)
    return


if __name__ == "__main__":
    app.run()
