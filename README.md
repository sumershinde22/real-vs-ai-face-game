# Can you spot the AI face?

A ten-round game: you and a **MobileNet-FT** classifier both call each face, and the scores
are tracked side by side. Built for **CS6180 Deep Learning, HW1** (real vs. AI-generated face
detection).

**Live app:** <paste your Streamlit Community Cloud URL here>

## The model
| | |
|---|---|
| architecture | MobileNet-FT (MobileNetV3Small, last 2 blocks fine-tuned) |
| validation accuracy | 79.4% |
| test accuracy (300 held-out images) | 76.7% |
| ROC-AUC | 0.838 |
| input | 128x128 RGB |

The 80 game images come from the held-out course test set, so the model never trained
on any of them.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a **public GitHub repo**.
2. [share.streamlit.io](https://share.streamlit.io) -> **Create app** -> that repo, branch
   `main`, main file `streamlit_app.py`.
3. Deploy. The first build installs TensorFlow and takes a few minutes. If it fails on the
   Python version, pick Python 3.12 under *Advanced settings*.

## Files
- `streamlit_app.py` - the whole app
- `best_model.keras` - trained classifier
- `model_config.json` - the preprocessing the model expects (image size + input scale)
- `test_images/` - held-out faces, `real_*.png` / `ai_*.png` (the label is the filename)
- `requirements.txt` - runtime dependencies
