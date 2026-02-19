# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "community==1.0.0b1",
#     "marimo>=0.19.11",
#     "matplotlib==3.10.8",
#     "networkx==3.6.1",
#     "pandas==3.0.0",
#     "plotly==6.5.2",
#     "pyvis==0.3.2",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import json
    import networkx as nx
    from collections import defaultdict
    from itertools import combinations

    @mo.cache
    def load_mc3(path="MC3_graph.json"):
        with open(path, "r") as f:
            g = json.load(f)
        nodes_by_id = {n["id"]: n for n in g["nodes"]}
        edge_list = g["edges"]
        return nodes_by_id, edge_list

    @mo.cache
    def build_entity_interaction_graph(nodes_by_id, edge_list, include_relationships=True):
        ENTITY_SUBTYPES = {"Person", "Vessel", "Organization"}

        def is_entity(nid: str) -> bool:
            n = nodes_by_id.get(nid)
            return n is not None and n.get("type") == "Entity" and n.get("sub_type") in ENTITY_SUBTYPES

        def is_comm_event(nid: str) -> bool:
            n = nodes_by_id.get(nid)
            return n is not None and n.get("type") == "Event" and n.get("sub_type") == "Communication"

        # 1) sender/receiver projection via Communication events
        senders = defaultdict(set)   # event -> set(entity)
        receivers = defaultdict(set) # event -> set(entity)

        for e in edge_list:
            src, tgt = e["source"], e["target"]
            etype = e.get("type")

            if etype == "sent" and is_entity(src) and is_comm_event(tgt):
                senders[tgt].add(src)

            if etype == "received" and is_comm_event(src) and is_entity(tgt):
                receivers[src].add(tgt)

        entity_graph = nx.Graph()

        # add entity nodes
        for nid, n in nodes_by_id.items():
            if is_entity(nid):
                entity_graph.add_node(
                    nid,
                    label=n.get("label"),
                    sub_type=n.get("sub_type"),
                    inferred=n.get("is_inferred", False),
                )

        # add weighted entity-entity edges from comm events
        for ev in set(senders) | set(receivers):
            for s in senders.get(ev, ()):
                for r in receivers.get(ev, ()):
                    if s == r:
                        continue
                    if entity_graph.has_edge(s, r):
                        entity_graph[s][r]["weight"] += 1
                        entity_graph[s][r]["kinds"].add("communication")
                    else:
                        entity_graph.add_edge(s, r, weight=1, kinds={"communication"})

        # 2) optional: relationship projection via Relationship_* nodes
        if include_relationships:
            rel_to_entities = defaultdict(set)

            for e in edge_list:
                src, tgt = e["source"], e["target"]
                if is_entity(src) and isinstance(tgt, str) and tgt.startswith("Relationship_"):
                    rel_to_entities[tgt].add(src)
                if isinstance(src, str) and src.startswith("Relationship_") and is_entity(tgt):
                    rel_to_entities[src].add(tgt)

            for rel, ents in rel_to_entities.items():
                for a, b in combinations(sorted(ents), 2):
                    if a == b:
                        continue
                    if entity_graph.has_edge(a, b):
                        entity_graph[a][b]["weight"] += 1
                        entity_graph[a][b]["kinds"].add("relationship")
                    else:
                        entity_graph.add_edge(a, b, weight=1, kinds={"relationship"})

        return entity_graph

    nodes_by_id, edge_list = load_mc3()
    entity_G = build_entity_interaction_graph(nodes_by_id, edge_list, include_relationships=True)

    len(entity_G.nodes), len(entity_G.edges)
    return entity_G, mo, nx


@app.cell
def _(entity_G, mo):
    @mo.cache
    def person_vessel_subgraph(entity_G):
        keep_types = {"Person", "Vessel"}

        nodes_to_keep = [
            n for n, data in entity_G.nodes(data=True)
            if data.get("sub_type") in keep_types
        ]

        subG = entity_G.subgraph(nodes_to_keep).copy()
        return subG

    pv_G = person_vessel_subgraph(entity_G)

    return (pv_G,)


@app.cell
def _(mo, nx, pv_G):
    import matplotlib.pyplot as plt

    @mo.cache
    def plot_person_vessel_network(pv_G):

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(pv_G, k=0.5, seed=42)

        # color nodes by type
        colors = [
            "steelblue" if pv_G.nodes[n]["sub_type"] == "Person"
            else "orange"
            for n in pv_G.nodes
        ]

        # size nodes by degree
        sizes = [
            100 + 300 * pv_G.degree(n)
            for n in pv_G.nodes
        ]

        # edge width by weight
        widths = [
            pv_G[u][v]["weight"]
            for u, v in pv_G.edges
        ]

        nx.draw_networkx_nodes(
            pv_G,
            pos,
            node_color=colors,
            node_size=sizes,
            alpha=0.9,
        )

        nx.draw_networkx_edges(
            pv_G,
            pos,
            width=widths,
            alpha=0.5,
        )

        nx.draw_networkx_labels(
            pv_G,
            pos,
            font_size=8,
        )

        plt.title("Person–Vessel Interaction Network")
        plt.axis("off")
        plt.show()
    


    plot_person_vessel_network(pv_G)
    return (plt,)


@app.cell
def _(mo, nx, pv_G):
    import pandas as pd

    @mo.cache
    def interaction_metrics(pv_G):

        degree = dict(pv_G.degree())
        betweenness = nx.betweenness_centrality(pv_G)

        df = pd.DataFrame({
            "node": list(pv_G.nodes),
            "type": [pv_G.nodes[n]["sub_type"] for n in pv_G.nodes],
            "degree": [degree[n] for n in pv_G.nodes],
            "betweenness": [betweenness[n] for n in pv_G.nodes],
        })

        return df.sort_values("degree", ascending=False)

    metrics_df = interaction_metrics(pv_G)
    metrics_df.head(10)
    return (pd,)


@app.cell
def _(mo, nx, pv_G):
    @mo.cache
    def detect_communities(pv_G):
        communities = nx.community.greedy_modularity_communities(pv_G)

        return [
            [pv_G.nodes[n]["label"] for n in community]
            for community in communities
        ]

    detect_communities(pv_G)
    return


@app.cell
def _(mo, pv_G):
    node_selector = mo.ui.dropdown(
        options=sorted(pv_G.nodes),
        label="Select entity",
    )

    node_selector
    return (node_selector,)


@app.cell
def _(mo, node_selector, pd, pv_G):
    @mo.cache
    def show_neighbors(pv_G, node):
        neighbors = list(pv_G.neighbors(node))

        return pd.DataFrame({
            "connected_to": neighbors,
            "type": [pv_G.nodes[n]["sub_type"] for n in neighbors],
            "weight": [pv_G[node][n]["weight"] for n in neighbors],
        }).sort_values("weight", ascending=False)

    show_neighbors(pv_G, node_selector.value)
    return


@app.cell
def _(entity_G, mo, nx):
    from collections import Counter
    @mo.cache
    def detect_entity_communities(entity_G):
        communities = list(nx.community.greedy_modularity_communities(entity_G))
        return communities

    communities = detect_entity_communities(entity_G)

    len(communities)
    return Counter, communities


@app.cell
def _(Counter, communities, entity_G, mo, pd):
    @mo.cache
    def summarize_communities(entity_G, communities):

        rows = []

        for i, comm in enumerate(communities):

            subtypes = Counter(
                entity_G.nodes[n]["sub_type"]
                for n in comm
            )

            labels = [
                entity_G.nodes[n]["label"]
                for n in comm
            ]

            rows.append({
                "community_id": i,
                "size": len(comm),
                "people": subtypes.get("Person", 0),
                "vessels": subtypes.get("Vessel", 0),
                "organizations": subtypes.get("Organization", 0),
                "members": labels
            })

        return pd.DataFrame(rows).sort_values("size", ascending=False)

    community_df = summarize_communities(entity_G, communities)

    community_df
    return (community_df,)


@app.cell
def _(Counter, communities, entity_G, mo, pd):
    @mo.cache
    def infer_topics(entity_G, communities):

        topic_keywords = {
            "Environmentalism": [
                "Green", "Guardian", "Environmental"
            ],
            "Fishing / Leisure": [
                "Fishing", "Charter", "Leisure"
            ],
            "Sailor Shift": [
                "Shift", "Sailor"
            ],
            "Government": [
                "City Council", "Port Authority"
            ]
        }

        results = []

        for i, comm in enumerate(communities):

            labels = [
                entity_G.nodes[n]["label"]
                for n in comm
            ]

            topic_counts = Counter()

            for label in labels:
                for topic, keywords in topic_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in str(label).lower():
                            topic_counts[topic] += 1

            dominant_topic = None
            if topic_counts:
                dominant_topic = topic_counts.most_common(1)[0][0]

            results.append({
                "community_id": i,
                "dominant_topic": dominant_topic,
                "topic_counts": dict(topic_counts),
            })

        return pd.DataFrame(results)

    topic_df = infer_topics(entity_G, communities)

    topic_df
    return (topic_df,)


@app.cell
def _(community_df, topic_df):
    community_analysis = community_df.merge(topic_df, on="community_id")

    community_analysis
    return


@app.cell
def _(communities, entity_G, mo, nx, plt):
    @mo.cache
    def plot_communities(entity_G, communities):

        plt.figure(figsize=(12, 10))

        pos = nx.spring_layout(entity_G, seed=42)

        colors = plt.cm.tab10.colors

        for i, comm in enumerate(communities):

            nx.draw_networkx_nodes(
                entity_G,
                pos,
                nodelist=list(comm),
                node_color=[colors[i % len(colors)]],
                label=f"Community {i}",
                node_size=300,
            )

        nx.draw_networkx_edges(entity_G, pos, alpha=0.3)

        plt.title("Entity Communities and Associations")
        plt.legend()
        plt.axis("off")
        plt.show()

    plot_communities(entity_G, communities)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Group 1 — Environmentalism group

    Typical features:

    * Organization: Green Guardians

    * Connected people: activists, associates

    * Connected vessels: protest or activist vessels

    Topic: Environmental activism


    ### Group 2 — Sailor Shift group

    Typical features:

    * Organization: Sailor Shift

    * Crew members and operational vessels

    Topic: maritime operations


    ### Group 3 — Fishing / leisure vessels group

    Typical features:

    * Fishing vessels

    * Charter vessels

    * Recreational vessels

    * Individual operators

    Topic: fishing and leisure maritime activity


    ### Group 4 — Government / Port authority group

    Typical features:

    * City Council

    * Port Authority

    * Administrative entities

    Topic: governance and regulation
    """)
    return


if __name__ == "__main__":
    app.run()
