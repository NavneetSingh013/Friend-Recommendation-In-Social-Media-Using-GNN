"""
Streamlit Demo App for Friend Recommendation using GNN
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.link_predictor import GraphSAGELinkPredictor, GATLinkPredictor
from src.models.seal import SEAL
from src.evaluation import get_top_k_recommendations
from src.evaluation.explainability import explain_recommendation
import networkx as nx
import plotly.graph_objects as go
import yaml

# Page config
st.set_page_config(page_title="Friend Recommendation GNN", layout="wide")

# Title
st.title("Friend Recommendation using Graph Neural Networks")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", ["facebook", "synthetic"])

# Load data
@st.cache_data
def load_data(dataset):
    """Load processed data."""
    if dataset == "facebook":
        try:
            data = torch.load("data/processed/facebook_combined.pt", weights_only=False)
            link_data = torch.load("data/processed/facebook_link_data.pt", weights_only=False)
            return data, link_data, "Facebook"
        except FileNotFoundError:
            st.error("Facebook dataset not found! Please run: python scripts/download_and_prepare.py --dataset facebook --download --preprocess")
            st.stop()
    else:  # synthetic
        try:
            data = torch.load("data/processed/synthetic.pt", weights_only=False)
            link_data = torch.load("data/processed/synthetic_link_data.pt", weights_only=False)
            return data, link_data, "Synthetic"
        except FileNotFoundError:
            st.error("Synthetic dataset not found! Please run: python scripts/download_and_prepare.py --dataset synthetic --preprocess")
            st.stop()

data, link_data, dataset_display_name = load_data(dataset_name)

# Model selection
model_name = st.sidebar.selectbox("Select Model", ["GraphSAGE", "GAT", "SEAL"])
# Use dataset-specific checkpoint path
default_checkpoint = f"data/checkpoints/{model_name.lower()}/{dataset_name}_best_model.pt"
# Fallback to old path if dataset-specific doesn't exist
fallback_checkpoint = f"data/checkpoints/{model_name.lower()}/best_model.pt"
checkpoint_path = st.sidebar.text_input("Checkpoint Path", default_checkpoint)

# Load model
@st.cache_resource
def load_model(_model_name, _checkpoint_path, _data, _dataset_name):
    """Load trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = _data.x.size(1)
    hidden_dim = 64
    embedding_dim = 64
    num_layers = 2
    dropout = 0.5
    
    if _model_name == "GraphSAGE":
        model = GraphSAGELinkPredictor(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, 
            aggregator='mean', predictor_method='mlp'
        ).to(device)
    elif _model_name == "GAT":
        model = GATLinkPredictor(
            input_dim, hidden_dim, embedding_dim, num_layers, 
            num_heads=4, dropout=dropout, predictor_method='mlp'
        ).to(device)
    else:  # SEAL
        num_hops = 2
        model = SEAL(
            input_dim, hidden_dim, num_layers, num_hops, dropout, pool='mean'
        ).to(device)
    
    # Load checkpoint if exists
    checkpoint_loaded = False
    if os.path.exists(_checkpoint_path):
        try:
            checkpoint = torch.load(_checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            st.sidebar.success(f"Loaded {_model_name} model from {_checkpoint_path}")
            checkpoint_loaded = True
        except RuntimeError as e:
            if "size mismatch" in str(e) or "shapes cannot be multiplied" in str(e):
                st.sidebar.error(f"Checkpoint dimension mismatch! The checkpoint was trained on a different dataset. Please train a model for the current dataset ({_dataset_name}).")
                st.sidebar.info(f"Train command: python scripts/train.py --model {_model_name.lower()} --dataset {_dataset_name} --config configs/{_model_name.lower()}_config.yaml")
            else:
                st.sidebar.warning(f"Error loading checkpoint: {e}")
    
    if not checkpoint_loaded:
        # Try fallback path (old naming convention)
        fallback_path = _checkpoint_path.replace(f"{_dataset_name}_", "")
        if os.path.exists(fallback_path) and fallback_path != _checkpoint_path:
            try:
                checkpoint = torch.load(fallback_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                st.sidebar.success(f"Loaded {_model_name} model from {fallback_path} (fallback)")
                checkpoint_loaded = True
            except RuntimeError:
                st.sidebar.warning(f"Fallback checkpoint also has dimension mismatch. Using untrained model.")
        
        if not checkpoint_loaded:
            st.sidebar.warning(f"Checkpoint not found or incompatible. Using untrained model.")
    
    return model, device

try:
    model, device = load_model(model_name, checkpoint_path, data, dataset_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main content
st.header("Friend Recommendation System")

# User selection
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("User ID", min_value=0, max_value=data.num_nodes - 1, value=0, step=1)
    k = st.slider("Number of Recommendations (K)", min_value=5, max_value=50, value=10, step=5)

with col2:
    st.metric("Total Users", data.num_nodes)
    st.metric("Total Edges", data.edge_index.size(1) // 2)
    st.metric("Feature Dimension", data.x.size(1))

# Get existing friends
existing_friends = set()
for i in range(data.edge_index.size(1)):
    src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
    if src == user_id:
        existing_friends.add(dst)
    if dst == user_id:
        existing_friends.add(src)

st.info(f"User {user_id} has {len(existing_friends)} existing friends.")

# Generate recommendations
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        # Get candidate nodes
        candidate_nodes = torch.tensor([i for i in range(data.num_nodes) 
                                       if i != user_id and i not in existing_friends])
        
        if len(candidate_nodes) == 0:
            st.warning("No candidate nodes available!")
        else:
            # Get top-K recommendations
            try:
                top_k_nodes, top_k_scores = get_top_k_recommendations(
                    model, data, user_id, candidate_nodes, k=k, device=device
                )
                
                # Display recommendations
                st.subheader(f"Top-{k} Friend Recommendations for User {user_id}")
                
                recommendations = []
                recommendation_details = {}
                for i, (node, score) in enumerate(zip(top_k_nodes, top_k_scores)):
                    node_id = node.item()
                    confidence = score.item()
                    
                    # Get explanation
                    explanation = explain_recommendation(data, user_id, node_id, confidence)
                    recommendations.append(explanation)
                    recommendation_details[node_id] = explanation
                
                # Create DataFrame
                df = pd.DataFrame(recommendations)
                df = df[['friend_id', 'confidence_score', 'mutual_friends', 
                        'num_shared_groups', 'profile_similarity', 'path_evidence']]
                df.columns = ['Friend ID', 'Confidence', 'Mutual Friends', 
                             'Shared Groups', 'Profile Similarity', 'Path Evidence']
                df.index = range(1, len(df) + 1)
                
                st.dataframe(df.style.format({
                    'Confidence': '{:.4f}',
                    'Profile Similarity': '{:.4f}'
                }))
                
                # Detailed explanations
                st.subheader("Detailed Explanations")
                for i, rec in enumerate(recommendations[:5]):  # Show top 5
                    with st.expander(f"Friend {rec['friend_id']} (Rank {i+1})"):
                        st.write(f"**Confidence Score**: {rec['confidence_score']:.4f}")
                        st.write(f"**Mutual Friends**: {rec['mutual_friends']}")
                        st.write(f"**Shared Groups**: {rec['num_shared_groups']}")
                        if rec['shared_groups']:
                            st.write(f"  - {', '.join(rec['shared_groups'])}")
                        st.write(f"**Profile Similarity**: {rec['profile_similarity']:.4f}")
                        st.write(f"**Path Evidence**: {rec['path_evidence']}")
                
                # Visualization
                st.subheader("Recommendation Visualization")
                
                # Create subgraph visualization limited for clarity
                G = nx.Graph()
                G.add_node(user_id, color='red', size=24, label='Selected User')
                
                limited_existing = list(existing_friends)[:10]
                for friend in limited_existing:
                    G.add_node(friend, color='blue', size=16, label='Existing Friend')
                    G.add_edge(user_id, friend)
                
                recommended_nodes = [node.item() for node in top_k_nodes]
                recommendation_scores = {
                    node.item(): score.item() for node, score in zip(top_k_nodes, top_k_scores)
                }
                for node_id in recommended_nodes:
                    G.add_node(node_id, color='green', size=18, label='Recommended Friend')
                    G.add_edge(user_id, node_id)
                
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Plotly edge trace
                edge_x, edge_y = [], []
                for src, dst in G.edges():
                    x0, y0 = pos[src]
                    x1, y1 = pos[dst]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                
                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1, color="#888", dash="dot"),
                    hoverinfo='skip',
                    mode='lines',
                    showlegend=False
                )
                
                traces = [edge_trace]
                node_groups = [
                    ("Selected User", [user_id], 'red', 24),
                    ("Existing Friend", limited_existing, 'blue', 16),
                    ("Recommended Friend", recommended_nodes, 'green', 18),
                ]
                
                for name, nodes_list, color, size in node_groups:
                    if not nodes_list:
                        continue
                    x_vals = [pos[n][0] for n in nodes_list]
                    y_vals = [pos[n][1] for n in nodes_list]
                    hover_text = []
                    for n in nodes_list:
                        if name == "Recommended Friend":
                            score = recommendation_scores.get(n)
                            profile_sim = recommendation_details.get(n, {}).get('profile_similarity')
                            hover_parts = [f"User {n}"]
                            hover_parts.append(f"Confidence: {score:.4f}" if score is not None else "Confidence: N/A")
                            hover_parts.append(f"Profile Similarity: {profile_sim:.4f}" if profile_sim is not None else "Profile Similarity: N/A")
                            hover_text.append("<br>".join(hover_parts))
                        elif n == user_id:
                            hover_text.append(f"User {n} (You)")
                        else:
                            hover_text.append(f"User {n}")
                    
                    traces.append(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='markers+text' if name == "Selected User" else 'markers',
                            text=[f'User {user_id}'] if name == "Selected User" else None,
                            textposition="top center",
                            marker=dict(size=size, color=color, opacity=0.85, line=dict(width=1, color='white')),
                            name=name,
                            hoverinfo='text',
                            hovertext=hover_text
                        )
                    )
                
                fig = go.Figure(data=traces)
                fig.update_layout(
                    title=f"Friend Network for User {user_id}",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=10, r=10, t=60, b=10),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    hovermode="closest",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.exception(e)

# Model information
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.write(f"**Model**: {model_name}")
st.sidebar.write(f"**Device**: {device}")
st.sidebar.write(f"**Parameters**: {sum(p.numel() for p in model.parameters()):,}")

# Dataset information
st.sidebar.markdown("---")
st.sidebar.header("Dataset Information")
st.sidebar.write(f"**Dataset**: {dataset_display_name}")
st.sidebar.write(f"**Nodes**: {data.num_nodes}")
st.sidebar.write(f"**Edges**: {data.edge_index.size(1) // 2}")
st.sidebar.write(f"**Features**: {data.x.size(1)}")
if hasattr(data, 'circles') and data.circles:
    st.sidebar.write(f"**Groups/Circles**: {len(data.circles)}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Select a model (GraphSAGE, GAT, or SEAL)
2. Enter a user ID
3. Set the number of recommendations (K)
4. Click "Get Recommendations"
5. View recommendations with explanations
""")

