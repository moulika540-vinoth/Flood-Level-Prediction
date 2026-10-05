# FloodLens – Flood Level Prediction Using Deep Learning

Predicts flood water level (Level 0 – Level 4) from a flood/car image.

## Requirements

- Windows
- Anaconda
- Python 3.11
- TensorFlow 2.21
- Jupyter Notebook
- Python (floodlens) kernel

## IMPORTANT: Put your model here

Copy your existing **`floodlens_model.keras`** into the extracted **FloodLens** folder,
right next to `FloodLens.ipynb`. (Your dataset stays where it is.)

```
FloodLens/
├── FloodLens.ipynb
├── floodlens_model.keras   <-- put your model here
├── app.py
├── README.md
└── requirements.txt
```

## How to Run

1. Download the ZIP.
2. Extract the ZIP.
3. Open the extracted `FloodLens` folder.
4. Open Jupyter Notebook (Anaconda Prompt → `conda activate floodlens` → `jupyter notebook`).
5. Open `FloodLens.ipynb`.
6. Select the **Python (floodlens)** kernel (Kernel → Change Kernel).
7. Run the cells one by one (Shift + Enter).
8. In Section 5, change `image_path` to your image, or use the upload button / file picker in Section 6.
9. See the Predicted Flood Level and Confidence.

## Optional: Streamlit webpage

In Anaconda Prompt, inside the `FloodLens` folder:

```
conda activate floodlens
pip install streamlit
streamlit run app.py
```

A webpage opens in your browser. Upload an image to see the prediction.

## Model Results

Trained for 3 epochs. Final validation accuracy: **50.66%**, validation loss: 1.1612.
