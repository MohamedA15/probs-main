# probs-main

## Project Overview
This project contains trained chess models implemented in Python. The models are designed to handle both **6x6** and **8x8** chess boards. The main goal of the project is to train machine learning or AI-based models to evaluate chess positions and improve decision-making in gameplay.

---

## Models

### 1. Chess 6x6 Model
- Designed for a smaller 6x6 chessboard variant.
- Uses the same rules as standard chess but adapted to a smaller board.
- Trained using [describe dataset or method if applicable].
- Capable of predicting moves and evaluating board positions for this compact variant.
- Includes logging of training progress in `trained_models/mychess6x6/logs`.

### 2. Chess 8x8 Model
- Designed for standard chessboard (8x8).
- Trained to evaluate moves and positions in regular chess.
- Uses advanced training strategies to improve accuracy and decision-making.
- Training logs are saved in `trained_models/mychess8x8/logs/training_log.csv`.
- Model files are stored in `trained_models/mychess8x8`.


---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/MohamedA15/probs-main.git
Navigate to the relevant model folder:
cd probs-main/trained_models/mychess6x6
or
cd probs-main/trained_models/mychess8x8
Load the trained model in Python:
import joblib

model = joblib.load("path_to_model_file")
