# Friend Recommendation Demo App

Streamlit-based demo application for friend recommendation using Graph Neural Networks.

## Features

- **Interactive UI**: Select users and get friend recommendations
- **Multiple Models**: Support for GraphSAGE, GAT, and SEAL
- **Explainability**: Show mutual friends, shared groups, profile similarity, and path evidence
- **Visualization**: Network graph visualization of recommendations

## Running the Demo

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Preprocess data (see main README):
```bash
python scripts/download_and_prepare.py --dataset synthetic --preprocess
```

3. Train a model (optional, for better results):
```bash
python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
```

### Launch Demo

```bash
streamlit run demo/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select Model**: Choose from GraphSAGE, GAT, or SEAL
2. **Enter User ID**: Input the user ID for recommendations
3. **Set K**: Choose the number of recommendations (5-50)
4. **Get Recommendations**: Click the button to generate recommendations
5. **View Results**: See top-K recommendations with detailed explanations

## Explainability Features

For each recommended friend, the app shows:
- **Confidence Score**: Model prediction score
- **Mutual Friends**: Number of common friends
- **Shared Groups**: Common groups/interests
- **Profile Similarity**: Cosine similarity of node features
- **Path Evidence**: Shortest path length between users

## Notes

- The demo works with synthetic dataset out-of-the-box
- For Facebook dataset, download and preprocess first
- Trained models provide better recommendations
- Untrained models will still work but with random predictions

