"""
Run this script ONCE to generate cached topic modeling results.
This avoids re-running BERTopic, SentenceTransformers, and UMAP every time
the marimo notebook starts.

Usage:
    python save_topic_cache.py

Outputs:
    data/topic_plot_df.csv   — UMAP coordinates + topic assignments per message
    data/topics_df.csv       — Entity × Topic weight matrix
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

# ── 1. Load graph and build messages_df ──────────────────────────
print("Loading graph data...")
with open('data/MC3_graph.json', 'r') as f:
    graph_data = json.load(f)

nodes_by_id = {n['id']: n for n in graph_data['nodes']}

edges_to = defaultdict(list)
edges_from = defaultdict(list)
for edge in graph_data['edges']:
    edges_to[edge['target']].append(edge)
    edges_from[edge['source']].append(edge)

comm_events = [n for n in graph_data['nodes'] if n.get('sub_type') == 'Communication']

messages = []
for comm in comm_events:
    comm_id = comm['id']
    senders = [e['source'] for e in edges_to[comm_id] if e.get('type') == 'sent']
    receivers = [e['target'] for e in edges_from[comm_id] if e.get('type') == 'received']
    if senders and receivers:
        source_node = nodes_by_id.get(senders[0], {})
        target_node = nodes_by_id.get(receivers[0], {})
        messages.append({
            'comm_id': comm_id,
            'datetime': comm.get('timestamp', ''),
            'content': comm.get('content', ''),
            'source': source_node.get('name', senders[0]),
            'target': target_node.get('name', receivers[0]),
        })

messages_df = pd.DataFrame(messages)
print(f"  Built messages_df: {len(messages_df)} messages")

# ── 2. Run BERTopic ──────────────────────────────────────────────
print("Running BERTopic...")
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

texts = messages_df['content'].tolist()

representation_model = KeyBERTInspired()
topic_model = BERTopic(
    min_topic_size=5,
    nr_topics="auto",
    language="english",
    calculate_probabilities=True,
    representation_model=representation_model,
    verbose=True,
)
topics, probs = topic_model.fit_transform(texts)

# Extract topic keywords
topic_keywords = {}
unique_topics = set(topics)
if -1 in unique_topics:
    unique_topics.remove(-1)

for topic_id in unique_topics:
    try:
        keywords = topic_model.get_topic(topic_id)
        topic_keywords[topic_id] = [word for word, _ in keywords[:10]]
    except:
        continue

max_topic_id = max(topic_keywords.keys()) if topic_keywords else 0

# Build topics_listm
topics_listm = []
for i in range(max_topic_id + 1):
    if i in topic_keywords:
        topics_listm.append(topic_keywords[i])
    else:
        topics_listm.append([])

# Build doc_topics
doc_topics = []
for i, doc_topic in enumerate(topics):
    weights = [0.0] * (max_topic_id + 1)
    if doc_topic in topic_keywords:
        weights[doc_topic] = 1.0
    doc_topics.append(weights)

print(f"  Found {len(topics_listm)} topics")

# ── 3. Build topics_df ───────────────────────────────────────────
print("Building topics_df...")
topics_df = pd.DataFrame(doc_topics, columns=[f"Topic_{i}" for i in range(len(topics_listm))])
topics_df['source'] = messages_df['source'].values
topics_df.set_index('source', inplace=True)
topics_df = topics_df.groupby(topics_df.index).sum()

topics_df.to_csv('data/topics_df.csv')
print(f"  Saved data/topics_df.csv ({topics_df.shape})")

# ── 4. Run UMAP + build plot_df ──────────────────────────────────
print("Running SentenceTransformer + UMAP (this may take a minute)...")
import umap
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

reduced = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.0,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)

topics_umap, probs_umap = topic_model.transform(texts)

topic_info = topic_model.get_topic_info()[["Topic", "Name"]]
topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

plot_df = messages_df.copy()
plot_df["x"] = reduced[:, 0]
plot_df["y"] = reduced[:, 1]
plot_df["topic"] = topics_umap
plot_df["topic_name"] = plot_df["topic"].map(topic_name_map)
plot_df["content_short"] = plot_df["content"].apply(
    lambda t: str(t) if len(str(t)) <= 100 else str(t)[:100] + "..."
)

plot_df.to_csv('data/topic_plot_df.csv', index=False)
print(f"  Saved data/topic_plot_df.csv ({plot_df.shape})")

# ── 5. Save topic keywords for reference ─────────────────────────
keywords_df = pd.DataFrame([
    {"topic_id": i, "keywords": ", ".join(kws)}
    for i, kws in enumerate(topics_listm) if kws
])
keywords_df.to_csv('data/topic_keywords.csv', index=False)
print(f"  Saved data/topic_keywords.csv ({len(keywords_df)} topics)")

print("\nDone! You can now run the marimo notebook without waiting for topic modeling.")
