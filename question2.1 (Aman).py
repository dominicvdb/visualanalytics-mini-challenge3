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
    return Counter, defaultdict, go, json, make_subplots, mo, np, nx, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2.1: Understanding Interactions Between Vessels and People

    - *2. Clepper has noticed that people often communicate with (or about) the same people or vessels, and that grouping them together may help with the investigation.*
        - *a. Use visual analytics to help Clepper understand and explore the interactions and relationships between vessels and people in the knowledge graph.*

    To answer these questions, I create interactive visualization to explore:
    1. **Communication Network** - Who talks to whom and how frequently?
    2. **Communication Frequency Matrix** - Heatmap of message intensity between entities
    3. **Formal Relationship Network** - Structural relationships (Colleagues, Operates, Reports, etc.)
    4. **Entity Profiles** - Deep dive into individual actors
    5. **Communication Timeline** - Temporal patterns over the two-week period
    6. **Key Statistics** - Summary metrics and top communicators
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Preparation & Graph Creation

    ### 1.1 Loading and Structuring the Knowledge Graphs

    In this part I have loaded the `MC3_graph.json` file, extracted all nodes and edges and built a lookup dictionaries for efficient querying. I have categorized nodes into:
    - **Persons** (18 entities)
    - **Vessels** (15 entities)
    - **Organizations** (5 entities)
    - **Groups** (5 entities)
    - **Locations** (29 entities)

    which helps for filtering and analysis by entity type.
    """)
    return


@app.cell
def _(json):
    # At first we will load the knowledge graph data that is provided
    with open('data/MC3_graph.json', 'r') as f:
        graph_data = json.load(f)

    # Then we will build lookup dictionaries
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
    print(f"Total entities of interest: {len(all_entities)}")
    return all_entities, entity_ids, graph_data, nodes_by_id


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Building the Communication Structures & Extracting Formal Relationships

    In this section I have extracted all nodes of subtype `Communication` and identified:
    - **Senders** → via `sent` edges
    - **Receivers** → via `received` edges

    Then I constructed a ***Communication Matrix*** which stores (**Timestamp, Content, Communication ID**). This matrix helps with creating the ***network graph, heatmap, individual profiles and other statistics***.

    After that, I have extracted the `Relationship` nodes and identified the connected entities while distinguishing between:
    - **Bidirectional relationships** (Colleagues, Friends) - both parties are sources
    - **Directional relationships** (Reports, Operates, etc.) - one source, one target

    and created a structured dataset with `type`, `entity1`, `entity2`, `bidirectional` and `rel_id` which helps with the structural network analysis.
    """)
    return


@app.cell
def _(defaultdict, entity_ids, graph_data):
    # I have built edge lookup structures
    edges_to = defaultdict(list)
    edges_from = defaultdict(list)
    for edge in graph_data['edges']:
        edges_to[edge['target']].append(edge)
        edges_from[edge['source']].append(edge)

    # After that I have extracted the Communication events
    comm_events = [n for n in graph_data['nodes'] if n.get('sub_type') == 'Communication']

    # Then I have built the communication matrix to answer questions like (who talks to whom)
    comm_matrix = defaultdict(lambda: defaultdict(list))  # This stores the actual communications

    for comm in comm_events:
        _comm_id = comm['id']
        _timestamp = comm.get('timestamp', '')
        _content = comm.get('content', '')

        # In here I find the senders (edges TO communication with type 'sent')
        _senders = [e['source'] for e in edges_to[_comm_id] if e.get('type') == 'sent']
        # and in here I find receivers (edges FROM communication with type 'received')
        _receivers = [e['target'] for e in edges_from[_comm_id] if e.get('type') == 'received']

        for _sender in _senders:
            for _receiver in _receivers:
                if _sender in entity_ids or _receiver in entity_ids:
                    comm_matrix[_sender][_receiver].append({
                        'timestamp': _timestamp,
                        'content': _content,
                        'comm_id': _comm_id
                    })

    # After all that, I extracted all the Relationship nodes
    relationships = [n for n in graph_data['nodes'] if n['type'] == 'Relationship']

    # Then build the relationship data structure
    relationship_data = []
    for _rel in relationships:
        _rel_id = _rel['id']
        _rel_type = _rel['sub_type']

        # I have got the connected entities
        _sources = [e['source'] for e in edges_to[_rel_id] if e['source'] in entity_ids]
        _targets = [e['target'] for e in edges_from[_rel_id] if e['target'] in entity_ids]

        # For bidirectional relationships such as (Colleagues, Friends), both parties are sources
        if _rel_type in ['Colleagues', 'Friends']:
            if len(_sources) >= 2:
                relationship_data.append({
                    'type': _rel_type,
                    'entity1': _sources[0],
                    'entity2': _sources[1],
                    'bidirectional': True,
                    'rel_id': _rel_id
                })
        else:
            # These are the directional relationships
            for _s in _sources:
                for _t in _targets:
                    relationship_data.append({
                        'type': _rel_type,
                        'entity1': _s,
                        'entity2': _t,
                        'bidirectional': False,
                        'rel_id': _rel_id
                    })

    print(f"Extracted {len(comm_events)} communications and {len(relationship_data)} formal relationships")
    return comm_events, comm_matrix, relationship_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2. Communication Network**

    In this visualization, I show the ***communication flow*** between all entities (persons, vessels, and organizations). This answers the fundamental question: **who talks to whom and how often?**

    **Visual Encodings:**
    - **Node Size** = Communication activity (larger nodes = more messages sent/received)
    - **Edge Thickness** = Message frequency between two entities
    - **Node Color** = Entity type (Teal = Person, Coral Red = Vessel, Mint = Organization, Salmon = Group)
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
    In the following code, I have built a **directed NetworkX graph** with edge weights representing the number of messages. I used multiple layout algorithms (Spring, Circular, and Kamada-Kawai) to provide different perspectives. For the visual encoding I have used:
    - **Node size** = degree of communication activity
    - **Node color** = entity type
    - **Edge thickness** = message frequency
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
    # At first I have built filtered communication network
    def build_comm_network(entity_types, min_comms, layout):
        # Filter entities by type
        filtered_entities = {
            eid: e for eid, e in all_entities.items() 
            if e.get('sub_type') in entity_types
        }

        # Then I created the NetworkX graph
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

        # From here I calculate the layout (spring, circular and kamada-kawai)
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
    # Color mapping for entity types
    _color_map = {
        'Person': '#4ECDC4',       # Teal
        'Vessel': '#FF6B6B',       # Coral red
        'Organization': '#95E1D3', # Mint
        'Group': '#F38181'         # Salmon
    }

    # In here I have the edge traces
    _edge_traces = []
    for (_u, _v), _weight in edge_weights_comm.items():
        _x0, _y0 = pos_comm[_u]
        _x1, _y1 = pos_comm[_v]

        _edge_trace = go.Scatter(
            x=[_x0, _x1, None],
            y=[_y0, _y1, None],
            mode='lines',
            line=dict(
                width=min(_weight / 2, 8),
                color='rgba(150, 150, 150, 0.4)'
            ),
            hoverinfo='text',
            text=f"{_u} → {_v}: {_weight} messages",
            showlegend=False
        )
        _edge_traces.append(_edge_trace)

    # In here I have the node traces (grouped by type for the legend)
    _node_traces = []
    for _entity_type in node_type_filter.value:
        _nodes_of_type = [n for n in G_comm.nodes() 
                        if nodes_by_id.get(n, {}).get('sub_type') == _entity_type]

        if not _nodes_of_type:
            continue

        _x_vals = [pos_comm[n][0] for n in _nodes_of_type]
        _y_vals = [pos_comm[n][1] for n in _nodes_of_type]

        # In here I calculated the node sizes based on degree
        _sizes = [max(15, min(50, G_comm.degree(n) * 5)) for n in _nodes_of_type]

        # Added the hover text
        _hover_texts = []
        for _n in _nodes_of_type:
            _in_deg = G_comm.in_degree(_n)
            _out_deg = G_comm.out_degree(_n)
            _hover_texts.append(f"<b>{_n}</b><br>Type: {_entity_type}<br>Sent to: {_out_deg} entities<br>Received from: {_in_deg} entities")

        _trace = go.Scatter(
            x=_x_vals,
            y=_y_vals,
            mode='markers+text',
            marker=dict(
                size=_sizes,
                color=_color_map[_entity_type],
                line=dict(width=2, color='white')
            ),
            text=[n if len(n) < 15 else n[:12]+'...' for n in _nodes_of_type],
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=_hover_texts,
            name=_entity_type
        )
        _node_traces.append(_trace)

    # Then finally I have combined all the traces
    fig_comm_network = go.Figure(data=_edge_traces + _node_traces)

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
    ## **3. Communication Frequency Matrix (Heatmap)**

    This heatmap provides a detailed overview of the ***communication intensity*** between entities. The darker the cell color, the more frequent the communication between that sender-receiver pair.

    The matrix is organized with **[P] = Person, [V] = Vessel, [O] = Organization** prefixes for easy identification.
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
        # At first I filter the entities
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
    _labels_hm = []
    for _eid in entities_hm:
        _etype = all_entities[_eid].get('sub_type', '?')[0]  # First letter
        _labels_hm.append(f"[{_etype}] {_eid}")

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_hm,
        x=_labels_hm,
        y=_labels_hm,
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
    ## **4. Formal Relationship Network**

    Beyond communications, entities in Oceanus also have ***formal relationships*** such as Colleagues, Operates, Reports, Coordinates, and Suspicious. This network visualization displays those structural connections.

    In this section I have built an **undirected NetworkX graph** to visualize the formal relationships. The relationship types in the data include:

    - **Green** for Colleagues
    - **Blue** for Operates
    - **Purple** for Reports
    - **Orange** for Coordinates
    - **Red** for Suspicious

    The interactive filter allows selecting which relationship types to display, making it easier to focus on specific types of connections.
    """)
    return


@app.cell
def _(mo):
    # I added the filter with all the relationship options
    rel_type_filter = mo.ui.multiselect(
        options=['Colleagues', 'Operates', 'Reports', 'Coordinates', 'Suspicious', 'Friends', 'Unfriendly'],
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
        _edge_rels = Counter()
        for _rel in filtered_rels:
            _e1, _e2 = _rel['entity1'], _rel['entity2']
            if _e1 in all_entities and _e2 in all_entities:
                _key = tuple(sorted([_e1, _e2]))
                _edge_rels[(_key, _rel['type'])] += 1

        # Add nodes and edges
        for (_key, _rtype), _count in _edge_rels.items():
            _e1, _e2 = _key
            G.add_node(_e1, sub_type=all_entities[_e1].get('sub_type'))
            G.add_node(_e2, sub_type=all_entities[_e2].get('sub_type'))

            if G.has_edge(_e1, _e2):
                G[_e1][_e2]['types'].append(_rtype)
            else:
                G.add_edge(_e1, _e2, types=[_rtype])

        return G

    G_rel = build_rel_network(rel_type_filter.value)

    # Layout
    if len(G_rel.nodes()) > 0:
        pos_rel = nx.spring_layout(G_rel, k=3, iterations=50, seed=42)
    else:
        pos_rel = {}

    # Color map for relationship types
    _rel_color_map = {
        'Colleagues': '#2ECC71',     # Green
        'Operates': '#3498DB',       # Blue
        'Reports': '#9B59B6',        # Purple
        'Coordinates': '#F39C12',    # Orange
        'Suspicious': '#E74C3C',     # Red
        'Friends': '#1ABC9C',        # Teal
        'Unfriendly': '#C0392B',     # Dark red
    }

    # Node type color map
    _node_color_map = {
        'Person': '#4ECDC4',
        'Vessel': '#FF6B6B',
        'Organization': '#95E1D3',
        'Group': '#F38181'
    }

    # Create edge traces for each relationship type
    _rel_edge_traces = []
    for _rtype in rel_type_filter.value:
        _x_edges, _y_edges = [], []
        for _u, _v, _data in G_rel.edges(data=True):
            if _rtype in _data['types']:
                _x0, _y0 = pos_rel[_u]
                _x1, _y1 = pos_rel[_v]
                _x_edges.extend([_x0, _x1, None])
                _y_edges.extend([_y0, _y1, None])

        if _x_edges:
            _trace = go.Scatter(
                x=_x_edges, y=_y_edges,
                mode='lines',
                line=dict(width=2, color=_rel_color_map.get(_rtype, 'gray')),
                name=_rtype,
                hoverinfo='none'
            )
            _rel_edge_traces.append(_trace)

    # Node traces
    _rel_node_traces = []
    for _ntype in ['Person', 'Vessel', 'Organization', 'Group']:
        _nodes_of_type = [n for n in G_rel.nodes() 
                        if nodes_by_id.get(n, {}).get('sub_type') == _ntype]
        if not _nodes_of_type:
            continue

        _x_vals = [pos_rel[n][0] for n in _nodes_of_type]
        _y_vals = [pos_rel[n][1] for n in _nodes_of_type]

        # Hover text with relationship info
        _hover_texts = []
        for _n in _nodes_of_type:
            _neighbors = list(G_rel.neighbors(_n))
            _rel_summary = []
            for _nb in _neighbors:
                _types = G_rel[_n][_nb]['types']
                _rel_summary.append(f"  • {_nb}: {', '.join(_types)}")
            _hover_text = f"<b>{_n}</b> ({_ntype})<br>Relationships:<br>" + "<br>".join(_rel_summary[:10])
            if len(_rel_summary) > 10:
                _hover_text += f"<br>  ...and {len(_rel_summary)-10} more"
            _hover_texts.append(_hover_text)

        _trace = go.Scatter(
            x=_x_vals, y=_y_vals,
            mode='markers+text',
            marker=dict(
                size=20,
                color=_node_color_map[_ntype],
                line=dict(width=2, color='white')
            ),
            text=_nodes_of_type,
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            hovertext=_hover_texts,
            name=f'{_ntype}s',
            legendgroup='nodes'
        )
        _rel_node_traces.append(_trace)

    fig_rel_network = go.Figure(data=_rel_edge_traces + _rel_node_traces)

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
    ## **5. Individual Communication Profiles (Entity Deep Dive)**

    In this section I have created an interactive entity profile analysis. By selecting an entity from the dropdown, we can examine:
    - **Messages Sent** - Who does the entity send messages to and how many?
    - **Messages Received** - Who sends messages to the entity and how many?
    - **Formal Relationships** - What structural relationships does this entity have?

    This helps Clepper to dive deep into individual actors and understand their communication patterns and relationships in detail. For example:
    """)
    return


@app.cell
def _(all_entities, mo):
    # I have created the entity selector dropdown for the deep dive analysis
    entity_selector = mo.ui.dropdown(
        options=sorted(all_entities.keys()),
        value='Nadia Conti',
        label="Select Entity to Analyze:"
    )
    entity_selector
    return (entity_selector,)


@app.cell
def _(mo):
    mo.md(r"""
    In the following visualization, I display a **side-by-side bar chart** showing:
    - Left: Messages sent by the selected entity (sorted by frequency)
    - Right: Messages received by the selected entity (sorted by frequency)

    The color encoding distinguishes between **Persons (Teal)** and **Vessels (Red)**, helping to quickly identify the type of communication partners.
    """)
    return


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
    # I get the selected entity from the dropdown
    selected_entity = entity_selector.value

    # In here I gather all communications involving this entity
    _sent_to = {}
    _received_from = {}

    # Messages sent BY selected entity
    if selected_entity in comm_matrix:
        for _receiver, _comms in comm_matrix[selected_entity].items():
            _sent_to[_receiver] = len(_comms)

    # Messages received BY selected entity
    for _sender in comm_matrix:
        if selected_entity in comm_matrix[_sender]:
            _received_from[_sender] = len(comm_matrix[_sender][selected_entity])

    # I create the subplot figure with two bar charts side by side
    fig_entity_profile = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Messages Sent by {selected_entity}',
            f'Messages Received by {selected_entity}'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Sent messages bar chart (sorted by frequency, top 15)
    _sent_data = sorted(_sent_to.items(), key=lambda x: -x[1])[:15]
    if _sent_data:
        _recipients, _counts = zip(*_sent_data)
        # Color based on entity type: Teal for Person, Red for Vessel
        _colors_sent = ['#FF6B6B' if all_entities.get(_r, {}).get('sub_type') == 'Vessel' else '#4ECDC4' for _r in _recipients]
        fig_entity_profile.add_trace(
            go.Bar(x=list(_recipients), y=list(_counts), marker_color=_colors_sent, name='Sent', showlegend=False),
            row=1, col=1
        )

    # Received messages bar chart (sorted by frequency, top 15)
    _recv_data = sorted(_received_from.items(), key=lambda x: -x[1])[:15]
    if _recv_data:
        _senders, _counts_r = zip(*_recv_data)
        # Color based on entity type: Teal for Person, Red for Vessel
        _colors_recv = ['#FF6B6B' if all_entities.get(_s, {}).get('sub_type') == 'Vessel' else '#4ECDC4' for _s in _senders]
        fig_entity_profile.add_trace(
            go.Bar(x=list(_senders), y=list(_counts_r), marker_color=_colors_recv, name='Received', showlegend=False),
            row=1, col=2
        )

    # I update the layout for better readability
    fig_entity_profile.update_layout(
        title=dict(
            text=f'<b>Communication Profile: {selected_entity}</b><br><sup>Type: {all_entities.get(selected_entity, {}).get("sub_type", "Unknown")} | Teal = Person, Red = Vessel</sup>',
            x=0.5
        ),
        showlegend=False,
        height=450
    )

    fig_entity_profile.update_xaxes(tickangle=45)

    # In here I build the relationships summary for the selected entity
    _entity_rels = [_r for _r in relationship_data 
                  if _r['entity1'] == selected_entity or _r['entity2'] == selected_entity]

    _rel_summary_data = []
    for _r in _entity_rels:
        _other = _r['entity2'] if _r['entity1'] == selected_entity else _r['entity1']
        _direction = '↔' if _r['bidirectional'] else ('→' if _r['entity1'] == selected_entity else '←')
        _rel_summary_data.append({
            'Relationship': _r['type'],
            'Direction': _direction,
            'Other Entity': _other,
            'Other Type': all_entities.get(_other, {}).get('sub_type', 'Unknown')
        })

    # I create a DataFrame for the relationships table
    rel_df = pd.DataFrame(_rel_summary_data) if _rel_summary_data else pd.DataFrame(columns=['Relationship', 'Direction', 'Other Entity', 'Other Type'])

    # Display the entity profile figure
    fig_entity_profile
    return rel_df, selected_entity


@app.cell
def _(mo, selected_entity):
    # In here I display the formal relationships table for the selected entity
    mo.md(f"""
    ### Formal Relationships of {selected_entity}

    The table below shows all formal relationships that **{selected_entity}** has with other entities. The direction indicates:
    - **↔** for bidirectional relationships (Colleagues, Friends)
    - **→** for outgoing relationships (the entity is the source)
    - **←** for incoming relationships (the entity is the target)
    """)
    return


@app.cell
def _(mo, rel_df):
    # I display the relationships table using marimo's table UI
    if len(rel_df) > 0:
        mo.ui.table(rel_df)
    else:
        mo.md("*No formal relationships found for this entity.*")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## **6. Communication Timeline**

    In this section I visualize the **temporal distribution** of communications over the two-week observation period (October 1-14, 2040). This helps to understand:
    - **Daily communication volume** - Which days had more activity?
    - **Trends over time** - Are there patterns or anomalies?
    """)
    return


@app.cell
def _(comm_events, go, pd):
    # In here I parse the timestamps and create timeline data
    _timeline_data = []
    for _comm in comm_events:
        _ts = _comm.get('timestamp', '')
        if _ts:
            _timeline_data.append({
                'timestamp': pd.to_datetime(_ts),
                'comm_id': _comm['id']
            })

    # I create a DataFrame from the timeline data
    timeline_df = pd.DataFrame(_timeline_data)
    timeline_df['date'] = timeline_df['timestamp'].dt.date
    timeline_df['hour'] = timeline_df['timestamp'].dt.hour
    timeline_df['day_name'] = timeline_df['timestamp'].dt.day_name()

    # I calculate the daily message counts
    daily_counts = timeline_df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])

    # I create the daily communication volume bar chart
    fig_timeline = go.Figure()

    fig_timeline.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['count'],
        marker_color='#3498DB',
        name='Daily Messages',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Messages: %{y}<extra></extra>'
    ))

    fig_timeline.update_layout(
        title=dict(
            text='<b>Communication Volume Over Time</b><br><sup>Daily message counts across the two-week observation period (Oct 1-14, 2040)</sup>',
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='%b %d', tickangle=45),
        plot_bgcolor='#fafafa'
    )

    # Display the timeline figure
    fig_timeline
    return (timeline_df,)


@app.cell
def _(mo):
    mo.md(r"""
    In addition to the daily view, I have also created a **heatmap showing communication activity by hour and day of week**. This helps to identify patterns such as:
    - Peak communication hours
    - Differences between weekdays and weekends
    - Regular scheduling patterns that might indicate coordinated activities
    """)
    return


@app.cell
def _(go, timeline_df):
    # In here I create a heatmap of communication activity by hour and day of week
    # I aggregate the data by day of week and hour
    _hourly_daily = timeline_df.groupby(['day_name', 'hour']).size().reset_index(name='count')

    # I define the order of days for proper sorting
    _day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # I create a pivot table for the heatmap
    _pivot_data = _hourly_daily.pivot(index='day_name', columns='hour', values='count').fillna(0)

    # I reorder the rows by day of week
    _pivot_data = _pivot_data.reindex(_day_order)

    # I create the heatmap figure
    fig_hourly_heatmap = go.Figure(data=go.Heatmap(
        z=_pivot_data.values,
        x=[f'{h:02d}:00' for h in range(24)],
        y=_day_order,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Messages: %{z}<extra></extra>'
    ))

    fig_hourly_heatmap.update_layout(
        title=dict(
            text='<b>Communication Activity by Hour and Day of Week</b><br><sup>Darker cells indicate more communication activity</sup>',
            x=0.5
        ),
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10))
    )

    # Display the hourly heatmap
    fig_hourly_heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **7. Key Statistics**

    In this section I present a summary of the **key statistics** from the knowledge graph analysis to provide a quick overview of the network scale and help identify the most important actors.
    """)
    return


@app.cell
def _(all_entities, comm_events, mo, relationship_data):
    # In here I calculate the key statistics for the summary
    _total_entities = len(all_entities)
    _total_comms = len(comm_events)
    _total_rels = len(relationship_data)

    # I display the key statistics using marimo's stat component
    mo.hstack([
        mo.stat(value=_total_entities, label="Total Entities", bordered=True),
        mo.stat(value=_total_comms, label="Communications", bordered=True),
        mo.stat(value=_total_rels, label="Formal Relationships", bordered=True),
    ], justify='center', gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Most Active Communicators

    The table below shows the **top 15 most active communicators** based on total message count (sent + received).
    """)
    return


@app.cell
def _(all_entities, comm_matrix, mo, pd):
    # In here I calculate the most active communicators
    _sent_counts = {}
    _recv_counts = {}

    # I count messages sent by each entity
    for _sender in comm_matrix:
        for _receiver, _comms in comm_matrix[_sender].items():
            _sent_counts[_sender] = _sent_counts.get(_sender, 0) + len(_comms)
            _recv_counts[_receiver] = _recv_counts.get(_receiver, 0) + len(_comms)

    # I calculate total activity (sent + received)
    _total_activity = {_k: _sent_counts.get(_k, 0) + _recv_counts.get(_k, 0) 
                      for _k in set(_sent_counts) | set(_recv_counts)}

    # I get the top 15 most active communicators
    _top_active = sorted(_total_activity.items(), key=lambda x: -x[1])[:15]

    # I create a DataFrame for the top active communicators
    top_active_df = pd.DataFrame(_top_active, columns=['Entity', 'Total Messages'])
    top_active_df['Type'] = top_active_df['Entity'].apply(lambda x: all_entities.get(x, {}).get('sub_type', 'Unknown'))
    top_active_df['Sent'] = top_active_df['Entity'].apply(lambda x: _sent_counts.get(x, 0))
    top_active_df['Received'] = top_active_df['Entity'].apply(lambda x: _recv_counts.get(x, 0))

    # I reorder the columns for better readability
    top_active_df = top_active_df[['Entity', 'Type', 'Sent', 'Received', 'Total Messages']]

    # I display the table using marimo's table UI
    mo.ui.table(top_active_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Entity Type Distribution

    The chart below shows the **distribution of entity types** in the network, helping to understand the composition of the actors involved.
    """)
    return


@app.cell
def _(all_entities, go):
    # In here I calculate the entity type distribution
    _type_counts = {}
    for _eid, _entity in all_entities.items():
        _etype = _entity.get('sub_type', 'Unknown')
        _type_counts[_etype] = _type_counts.get(_etype, 0) + 1

    # I create a pie chart for entity type distribution
    fig_entity_distribution = go.Figure(data=[go.Pie(
        labels=list(_type_counts.keys()),
        values=list(_type_counts.values()),
        hole=0.4,
        marker_colors=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig_entity_distribution.update_layout(
        title=dict(
            text='<b>Entity Type Distribution</b><br><sup>Breakdown of actors in the knowledge graph</sup>',
            x=0.5
        ),
        height=400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )

    # Display the entity distribution chart
    fig_entity_distribution
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Relationship Type Distribution

    The chart below shows the **distribution of relationship types** in the network, helping to understand what kinds of formal connections exist between entities.
    """)
    return


@app.cell
def _(go, relationship_data):
    # In here I calculate the relationship type distribution
    _rel_type_counts = {}
    for _rel in relationship_data:
        _rtype = _rel['type']
        _rel_type_counts[_rtype] = _rel_type_counts.get(_rtype, 0) + 1

    # I sort by count for better visualization
    _sorted_rel_types = sorted(_rel_type_counts.items(), key=lambda x: -x[1])

    # I create a horizontal bar chart for relationship type distribution
    fig_rel_distribution = go.Figure(data=[go.Bar(
        x=[_v for _k, _v in _sorted_rel_types],
        y=[_k for _k, _v in _sorted_rel_types],
        orientation='h',
        marker_color=['#F39C12', '#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#C0392B', '#1ABC9C'][:len(_sorted_rel_types)],
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )])

    fig_rel_distribution.update_layout(
        title=dict(
            text='<b>Relationship Type Distribution</b><br><sup>Count of each relationship type in the knowledge graph</sup>',
            x=0.5
        ),
        xaxis_title='Count',
        yaxis_title='Relationship Type',
        height=400,
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='#fafafa'
    )

    # Display the relationship distribution chart
    fig_rel_distribution
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Findings for Question 2.1**

    Based on the visual analytics executed above, several key insights emerge about the interactions and relationships between vessels and people in Oceanus. The following findings are all derived directly from the data analysis.

    TODO
    """)
    return


if __name__ == "__main__":
    app.run()
