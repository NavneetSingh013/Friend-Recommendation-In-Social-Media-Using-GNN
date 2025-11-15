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
import matplotlib.pyplot as plt
import yaml

# Page config
st.set_page_config(page_title="Friend Recommendation GNN", layout="wide")

# Title
st.title("Friend Recommendation using Graph Neural Networks")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Load data
@st.cache_data
def load_data():
    """Load processed data."""
    try:
        data = torch.load("data/processed/facebook_combined.pt", weights_only=False)
        link_data = torch.load("data/processed/facebook_link_data.pt", weights_only=False)
        return data, link_data
    except FileNotFoundError:
        # Try synthetic dataset
        try:
            data = torch.load("data/processed/synthetic.pt", weights_only=False)
            link_data = torch.load("data/processed/synthetic_link_data.pt", weights_only=False)
            return data, link_data
        except FileNotFoundError:
            st.error("No dataset found! Please run data preprocessing first.")
            st.stop()

data, link_data = load_data()

# Model selection
model_name = st.sidebar.selectbox("Select Model", ["GraphSAGE", "GAT", "SEAL"])
checkpoint_path = st.sidebar.text_input("Checkpoint Path", f"data/checkpoints/{model_name.lower()}/best_model.pt")

# Load model
@st.cache_resource
def load_model(_model_name, _checkpoint_path, _data):
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
    if os.path.exists(_checkpoint_path):
        checkpoint = torch.load(_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.sidebar.success(f"Loaded {_model_name} model from {_checkpoint_path}")
    else:
        st.sidebar.warning(f"Checkpoint not found: {_checkpoint_path}. Using untrained model.")
    
    return model, device

try:
    model, device = load_model(model_name, checkpoint_path, data)
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
                for i, (node, score) in enumerate(zip(top_k_nodes, top_k_scores)):
                    node_id = node.item()
                    confidence = score.item()
                    
                    # Get explanation
                    explanation = explain_recommendation(data, user_id, node_id, confidence)
                    recommendations.append(explanation)
                
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
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create subgraph visualization
                G = nx.Graph()
                G.add_node(user_id, color='red', size=500)
                
                # Add existing friends
                for friend in list(existing_friends)[:10]:  # Limit for visualization
                    G.add_node(friend, color='blue', size=300)
                    G.add_edge(user_id, friend)
                
                # Add recommended friends
                for node in top_k_nodes[:10]:
                    node_id = node.item()
                    G.add_node(node_id, color='green', size=400)
                    G.add_edge(user_id, node_id, style='dashed')
                
                # Draw graph
                pos = nx.spring_layout(G, k=1, iterations=50)
                colors = [G.nodes[node].get('color', 'gray') for node in G.nodes()]
                sizes = [G.nodes[node].get('size', 200) for node in G.nodes()]
                
                nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax, alpha=0.7)
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, style='dashed')
                nx.draw_networkx_labels(G, pos, {user_id: f'User {user_id}'}, ax=ax, font_size=8)
                
                ax.set_title(f"Friend Network for User {user_id}")
                ax.axis('off')
                
                st.pyplot(fig)
                
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
st.sidebar.write(f"**Nodes**: {data.num_nodes}")
st.sidebar.write(f"**Edges**: {data.edge_index.size(1) // 2}")
st.sidebar.write(f"**Features**: {data.x.size(1)}")

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

