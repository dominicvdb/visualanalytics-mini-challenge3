import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mini-Challenge 3 - Question 2: Entity Interactions & Relationships

    **Team Members:** Singh AmanDeep, Kim Wilmink, Dominic van den Bungelaar

    ---

    ## Question 2.1: Understanding Interactions Between Vessels and People

    > *Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation. Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.*

    This analysis provides interactive visualizations to explore:
    1. **Communication Network** - Who talks to whom and how frequently
    2. **Relationship Network** - Formal relationships (Colleagues, Operates, Reports, etc.)
    3. **Entity Profiles** - Deep-dive into individual actors
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
    return Counter, defaultdict, datetime, go, json, make_subplots, mo, np, nx, pd, px


@app.cell
def _(json):
    # Load the knowledge graph data
    with open('MC3_graph.json', 'r') as f:
        graph_data = json.load(f)
    
    # Build lookup dictionaries
    nodes_by_id = {n['id']: n for n in graph_data['nodes']}
    
    # Categorize entities
    persons = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Person'}
    vessels = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Vessel'}
    organizations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Organization'}
    groups = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Group'}
    locations = {n['id']: n for n in graph_data['nodes'] if n.get('sub_type') == 'Location'}
    
    # All entities of interest (for network analysis)
    all_entities = {**persons, **vessels, **organizations, **groups}
    entity_ids = set(all_entities.keys())
    
    print(f"Loaded: {len(persons)} persons, {len(vessels)} vessels, {len(organizations)} organizations, {len(groups)} groups")
    return (
        all_entities,
        entity_ids,
        graph_data,
        groups,
        locations,
        nodes_by_id,
        organizations,
        persons,
        vessels,
    )


@app.cell
def _(defaultdict, entity_ids, graph_data, nodes_by_id):
    # Build edge lookup structures
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)
    for edge in graph_data['edges']:
        edges_to[edge['target']].append(edge)
        edges_from[edge['source']].append(edge)
    
    # Extract Communication events
    comm_events = [n for n in graph_data['nodes'] if n.get('sub_type') == 'Communication']
    
    # Build communication matrix (who talks to whom)
    comm_matrix = defaultdict(lambda: defaultdict(list))  # Store actual communications
    
    for comm in comm_events:
        comm_id = comm['id']
        timestamp = comm.get('timestamp', '')
        content = comm.get('content', '')
        
        # Find senders (edges TO communication with type 'sent')
        senders = [e['source'] for e in edges_to[comm_id] if e.get('type') == 'sent']
        # Find receivers (edges FROM communication with type 'received')
        receivers = [e['target'] for e in edges_from[comm_id] if e.get('type') == 'received']
        
        for sender in senders:
            for receiver in receivers:
                if sender in entity_ids or receiver in entity_ids:
                    comm_matrix[sender][receiver].append({
                        'timestamp': timestamp,
                        'content': content,
                        'comm_id': comm_id
                    })
    
    # Extract Relationship nodes
    relationships = [n for n in graph_data['nodes'] if n['type'] == 'Relationship']
    
    # Build relationship data structure
    relationship_data = []
    for rel in relationships:
        rel_id = rel['id']
        rel_type = rel['sub_type']
        
        # Get connected entities
        sources = [e['source'] for e in edges_to[rel_id] if e['source'] in entity_ids]
        targets = [e['target'] for e in edges_from[rel_id] if e['target'] in entity_ids]
        
        # For bidirectional relationships (Colleagues, Friends), both parties are sources
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
            # Directional relationships
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
    return (
        comm_events,
        comm_matrix,
        edges_from,
        edges_to,
        relationship_data,
        relationships,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Communication Network Overview

    This visualization shows the **communication flow** between all entities (persons, vessels, organizations). Edge thickness represents communication frequency.
    """)
    return


@app.cell
def _(mo):
    # Controls for the network visualization
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
    # Build filtered communication network
    def build_comm_network(entity_types, min_comms, layout):
        # Filter entities by type
        filtered_entities = {
            eid: e for eid, e in all_entities.items() 
            if e.get('sub_type') in entity_types
        }
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for eid, entity in filtered_entities.items():
            G.add_node(eid, 
                      label=entity.get('name', eid),
                      sub_type=entity.get('sub_type'))
        
        # Add edges based on communications
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
        
        # Calculate layout
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
    
    # Create Plotly figure
    # Color mapping for entity types
    color_map = {
        'Person': '#4ECDC4',      # Teal
        'Vessel': '#FF6B6B',      # Coral red
        'Organization': '#95E1D3', # Mint
        'Group': '#F38181'         # Salmon
    }
    
    # Edge traces
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
    
    # Node traces (grouped by type for legend)
    node_traces = []
    for entity_type in node_type_filter.value:
        nodes_of_type = [n for n in G_comm.nodes() 
                        if nodes_by_id.get(n, {}).get('sub_type') == entity_type]
        
        if not nodes_of_type:
            continue
            
        x_vals = [pos_comm[n][0] for n in nodes_of_type]
        y_vals = [pos_comm[n][1] for n in nodes_of_type]
        
        # Calculate node sizes based on degree
        sizes = [max(15, min(50, G_comm.degree(n) * 5)) for n in nodes_of_type]
        
        # Hover text
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
    
    # Combine all traces
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
    
    fig_comm_network
    return (
        G_comm,
        build_comm_network,
        color_map,
        edge_traces,
        edge_weights_comm,
        fig_comm_network,
        node_traces,
        pos_comm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Communication Frequency Matrix (Heatmap)

    This heatmap provides a detailed view of **communication intensity** between entities. Darker cells indicate more frequent communication.
    """)
    return


@app.cell
def _(mo):
    # Entity selection for heatmap
    heatmap_type_filter = mo.ui.multiselect(
        options=['Person', 'Vessel', 'Organization'],
        value=['Person', 'Vessel'],
        label="Include Entity Types in Heatmap:"
    )
    heatmap_type_filter
    return (heatmap_type_filter,)


@app.cell
def _(all_entities, comm_matrix, go, heatmap_type_filter, np):
    # Build communication count matrix for heatmap
    def build_heatmap_data(entity_types):
        # Filter entities
        filtered = [eid for eid, e in all_entities.items() 
                   if e.get('sub_type') in entity_types]
        
        # Sort by entity type then name
        filtered = sorted(filtered, key=lambda x: (all_entities[x].get('sub_type', ''), x))
        
        # Build matrix
        n = len(filtered)
        matrix = np.zeros((n, n))
        
        for i, sender in enumerate(filtered):
            for j, receiver in enumerate(filtered):
                if sender in comm_matrix and receiver in comm_matrix[sender]:
                    matrix[i][j] = len(comm_matrix[sender][receiver])
        
        return filtered, matrix
    
    entities_hm, matrix_hm = build_heatmap_data(heatmap_type_filter.value)
    
    # Create labels with entity type indicators
    labels_hm = []
    for eid in entities_hm:
        etype = all_entities[eid].get('sub_type', '?')[0]  # First letter
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
    
    fig_heatmap
    return (
        build_heatmap_data,
        entities_hm,
        fig_heatmap,
        labels_hm,
        matrix_hm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Formal Relationship Network

    Beyond communications, entities have **formal relationships** such as Colleagues, Operates, Reports, Coordinates, etc. This network shows these structural connections.
    """)
    return


@app.cell
def _(mo):
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
    return (
        G_rel,
        build_rel_network,
        fig_rel_network,
        node_color_map,
        pos_rel,
        rel_color_map,
        rel_edge_traces,
        rel_node_traces,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Entity Deep-Dive: Individual Communication Profiles

    Select an entity to see their complete communication profile - who they talk to, how often, and the content of their messages.
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
    return (entity_selector,)


@app.cell
def _(
    all_entities,
    comm_matrix,
    entity_selector,
    go,
    make_subplots,
    pd,
    relationship_data,
):
    selected_entity = entity_selector.value
    
    # Gather all communications involving this entity
    sent_to = {}
    received_from = {}
    
    # Messages sent BY selected entity
    if selected_entity in comm_matrix:
        for receiver, comms in comm_matrix[selected_entity].items():
            sent_to[receiver] = comms
    
    # Messages received BY selected entity
    for sender in comm_matrix:
        if selected_entity in comm_matrix[sender]:
            received_from[sender] = comm_matrix[sender][selected_entity]
    
    # Create subplot figure
    fig_entity = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Messages Sent by {selected_entity}',
            f'Messages Received by {selected_entity}'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Sent messages bar chart
    sent_data = sorted([(k, len(v)) for k, v in sent_to.items()], key=lambda x: -x[1])[:15]
    if sent_data:
        recipients, counts = zip(*sent_data)
        colors_sent = ['#FF6B6B' if all_entities.get(r, {}).get('sub_type') == 'Vessel' else '#4ECDC4' for r in recipients]
        fig_entity.add_trace(
            go.Bar(x=list(recipients), y=list(counts), marker_color=colors_sent, name='Sent'),
            row=1, col=1
        )
    
    # Received messages bar chart
    recv_data = sorted([(k, len(v)) for k, v in received_from.items()], key=lambda x: -x[1])[:15]
    if recv_data:
        senders, counts_r = zip(*recv_data)
        colors_recv = ['#FF6B6B' if all_entities.get(s, {}).get('sub_type') == 'Vessel' else '#4ECDC4' for s in senders]
        fig_entity.add_trace(
            go.Bar(x=list(senders), y=list(counts_r), marker_color=colors_recv, name='Received'),
            row=1, col=2
        )
    
    fig_entity.update_layout(
        title=dict(
            text=f'<b>Communication Profile: {selected_entity}</b><br><sup>Type: {all_entities.get(selected_entity, {}).get("sub_type", "Unknown")} | Teal = Person, Red = Vessel</sup>',
            x=0.5
        ),
        showlegend=False,
        height=400
    )
    
    fig_entity.update_xaxes(tickangle=45)
    
    # Build relationships summary
    entity_rels = [r for r in relationship_data 
                  if r['entity1'] == selected_entity or r['entity2'] == selected_entity]
    
    rel_summary_data = []
    for r in entity_rels:
        other = r['entity2'] if r['entity1'] == selected_entity else r['entity1']
        direction = '↔' if r['bidirectional'] else ('→' if r['entity1'] == selected_entity else '←')
        rel_summary_data.append({
            'Relationship': r['type'],
            'Direction': direction,
            'Other Entity': other,
            'Other Type': all_entities.get(other, {}).get('sub_type', 'Unknown')
        })
    
    rel_df = pd.DataFrame(rel_summary_data) if rel_summary_data else pd.DataFrame(columns=['Relationship', 'Direction', 'Other Entity', 'Other Type'])
    
    (fig_entity, rel_df)
    return (
        entity_rels,
        fig_entity,
        received_from,
        recv_data,
        rel_df,
        rel_summary_data,
        selected_entity,
        sent_data,
        sent_to,
    )


@app.cell
def _(mo, rel_df, selected_entity):
    mo.md(f"""
    ### Formal Relationships of {selected_entity}
    
    {mo.ui.table(rel_df) if len(rel_df) > 0 else "No formal relationships found."}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Communication Timeline

    Visualize when communications happen over the two-week observation period.
    """)
    return


@app.cell
def _(comm_events, go, pd):
    # Parse timestamps and create timeline data
    timeline_data = []
    for comm in comm_events:
        ts = comm.get('timestamp', '')
        if ts:
            timeline_data.append({
                'timestamp': pd.to_datetime(ts),
                'comm_id': comm['id']
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['date'] = timeline_df['timestamp'].dt.date
    timeline_df['hour'] = timeline_df['timestamp'].dt.hour
    
    # Daily message counts
    daily_counts = timeline_df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['count'],
        marker_color='#3498DB',
        name='Daily Messages'
    ))
    
    fig_timeline.update_layout(
        title=dict(
            text='<b>Communication Volume Over Time</b><br><sup>Daily message counts across the two-week observation period</sup>',
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        height=350,
        showlegend=False
    )
    
    fig_timeline
    return daily_counts, fig_timeline, timeline_data, timeline_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Key Statistics Summary
    """)
    return


@app.cell
def _(
    all_entities,
    comm_events,
    comm_matrix,
    mo,
    pd,
    relationship_data,
):
    # Calculate key statistics
    total_entities = len(all_entities)
    total_comms = len(comm_events)
    total_rels = len(relationship_data)
    
    # Most active communicators
    sent_counts = {}
    recv_counts = {}
    for sender in comm_matrix:
        for receiver, comms in comm_matrix[sender].items():
            sent_counts[sender] = sent_counts.get(sender, 0) + len(comms)
            recv_counts[receiver] = recv_counts.get(receiver, 0) + len(comms)
    
    total_activity = {k: sent_counts.get(k, 0) + recv_counts.get(k, 0) for k in set(sent_counts) | set(recv_counts)}
    top_active = sorted(total_activity.items(), key=lambda x: -x[1])[:10]
    
    top_active_df = pd.DataFrame(top_active, columns=['Entity', 'Total Messages'])
    top_active_df['Type'] = top_active_df['Entity'].apply(lambda x: all_entities.get(x, {}).get('sub_type', 'Unknown'))
    
    mo.hstack([
        mo.stat(value=total_entities, label="Total Entities"),
        mo.stat(value=total_comms, label="Communications"),
        mo.stat(value=total_rels, label="Relationships"),
    ])
    return (
        recv_counts,
        sent_counts,
        top_active,
        top_active_df,
        total_activity,
        total_comms,
        total_entities,
        total_rels,
    )


@app.cell
def _(mo, top_active_df):
    mo.md(f"""
    ### Most Active Communicators
    
    {mo.ui.table(top_active_df)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Key Findings for Question 2.1
    
    Based on the visual analytics above, several insights emerge about interactions and relationships:
    
    1. **Communication Hubs**: Certain entities (like `Mako`, `Reef Guardian`, `Remora`, `Neptune`) show high communication activity, suggesting they are central to operations.
    
    2. **Journalist Pair**: `Clepper Jensen` and `Miranda Jordan` have the highest bilateral communication (20+18 messages), consistent with their investigative collaboration.
    
    3. **Pseudonym Network**: Entities like `The Intern`, `Mrs. Money`, `The Lookout`, `Boss`, `The Middleman` form a distinct communication cluster, suggesting an organized network using code names.
    
    4. **Vessel-Organization Links**: Several vessels (Neptune, Remora, Mako) communicate with `V. Miesel Shipping`, suggesting shipping operations coordination.
    
    5. **Green Guardians Network**: Conservation vessels (`Reef Guardian`, `Horizon`, `Sentinel`, `EcoVigil`) are connected to the `Green Guardians` organization through formal relationships.
    
    *Continue to Question 2.2 for community detection and topic analysis.*
    """)
    return


if __name__ == "__main__":
    app.run()
