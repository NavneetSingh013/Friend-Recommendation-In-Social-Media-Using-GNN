# Streamlit Installation and Usage Guide

## What the Error Means

The error `streamlit : The term 'streamlit' is not recognized` means that PowerShell cannot find the `streamlit` command in your system PATH. This is a common issue on Windows.

## Solutions

### Solution 1: Use Python Module Syntax (Recommended)

Instead of:
```powershell
streamlit run demo/streamlit_app.py
```

Use:
```powershell
python -m streamlit run demo/streamlit_app.py
```

This works because Python can find the streamlit module directly, regardless of PATH settings.

### Solution 2: Add Python Scripts to PATH

1. Find your Python Scripts directory:
   - Usually: `C:\Users\YourUsername\AppData\Roaming\Python\Python313\Scripts`
   - Or: `C:\Python313\Scripts` (if installed system-wide)

2. Add it to PATH:
   - Open System Properties → Environment Variables
   - Add the Scripts directory to your User PATH
   - Restart PowerShell

### Solution 3: Use Full Path

```powershell
C:\Users\YourUsername\AppData\Roaming\Python\Python313\Scripts\streamlit.exe run demo/streamlit_app.py
```

## Running the Demo App

Once Streamlit is accessible, run:

```powershell
python -m streamlit run demo/streamlit_app.py
```

The app will:
1. Start a local web server
2. Open in your browser at `http://localhost:8501`
3. Allow you to interact with the friend recommendation system

## Expected Output

You should see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Troubleshooting

### If the app doesn't open automatically:
- Copy the Local URL from the terminal
- Paste it into your browser

### If you see import errors:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the project directory

### If the model checkpoint is not found:
- The app will still work with an untrained model (random predictions)
- To use a trained model, make sure you've run training first:
  ```powershell
  python scripts/train.py --model graphsage --dataset synthetic --config configs/graphsage_config.yaml
  ```

## Features of the Demo App

- Select different models (GraphSAGE, GAT, SEAL)
- Choose a user ID
- Set number of recommendations (K)
- View top-K friend recommendations
- See detailed explanations (mutual friends, shared groups, etc.)
- Visualize the network graph

## Notes

- The app uses the synthetic dataset by default (if Facebook dataset is not available)
- Trained models provide better recommendations than untrained models
- The app will automatically load the best available dataset

