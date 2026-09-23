![CI](https://github.com/UNarasimha/milk-api/actions/workflows/ci.yml/badge.svg)

# 🥛 Milk Composition Prediction from NIR Spectra

An end-to-end TensorFlow project that predicts Fat, Protein, and Lactose content in raw milk directly from Near-Infrared (NIR) spectroscopy data — enabling real-time, non-destructive milk quality analysis on the farm.

**Live API:** https://milk-api-docker.onrender.com
**Interactive Docs:** https://milk-api-docker.onrender.com/docs

## Problem Statement

Traditional milk composition testing requires chemical analysis in a laboratory. This is:

- Slow (hours to days)
- Destructive (sample is consumed)
- Expensive (lab equipment, technicians)
- Not scalable for continuous farm-level monitoring

Solution: A deep learning model that reads NIR light transmittance through raw milk and instantly predicts its chemical composition. Mimics a handheld on-farm sensor.

## Dataset

- Source: KU Leuven / Zenodo — "Near-infrared spectra dataset of milk composition in transmittance mode"
- Samples: 1,224 raw milk measurements
- Features: 256 spectral points (wavelength range 960-1690 nm)
- Targets: Fat (%), Protein (%), Lactose (%)
- Preprocessing: Absorbance computed from raw transmittance, then Standard Normal Variate (SNV) normalization per spectrum

## Model Architecture

Feed-forward neural network built with TensorFlow/Keras:

Input (256) -> Dense(128) + ReLU + Dropout(0.3) -> Dense(64) + ReLU + Dropout(0.2) -> Dense(32) + ReLU -> Dense(3)

Outputs: Fat, Protein, Lactose

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr = 1e-3)
- Regularization: Dropout + EarlyStopping
- Training: 200 epochs max, batch size 32

## Results

| Target | RMSE | MAE | R2 |
|--------|------|-----|-----|
| Fat | 0.313 | 0.249 | 0.857 |
| Protein | 0.298 | 0.237 | 0.271 |
| Lactose | 0.157 | 0.120 | 0.209 |

Interpretation: Fat is highly predictable from NIR data (strong absorption bands). Protein and Lactose show weaker predictability due to overlapping water absorption bands — consistent with published NIR spectroscopy literature.

## Training Notebook

The complete training pipeline is available at `notebooks/training.ipynb`. It includes:

- Data loading and exploration
- Absorbance calculation from raw transmittance
- SNV normalization
- Train/validation/test split with target scaling
- Model building with the Keras Sequential API
- Training with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
- Evaluation (RMSE, MAE, R2) and prediction plots

## Deployment

The trained model is served as a REST API using FastAPI, containerized with Docker, and deployed on Render.

**Live API:** https://milk-api-docker.onrender.com
**Interactive Docs:** https://milk-api-docker.onrender.com/docs

Endpoints:

- GET /          - API info
- GET /health    - Health check
- POST /predict  - Predict composition from 256 spectra
- GET /docs      - Interactive Swagger UI

Example Request (public API):

curl -X POST "https://milk-api-docker.onrender.com/predict" -H "Content-Type: application/json" -d "{\"spectra\": [0.01, 0.02, 0.55]}"

Example Response:

{"Fat": 3.547, "Prot": 3.407, "Lact": 4.724}

Note: The free Render instance spins down after 15 minutes of inactivity. First request after idle may take up to 50 seconds.

## Docker

The application is containerized with a multi-layer Docker image based on `python:3.13-slim`.

Build locally:

  docker build -t milk-api .

Run the container:

  docker run -p 8000:8000 milk-api

Test the containerized API at http://localhost:8000/docs

## CI/CD

This project uses GitHub Actions for continuous integration. Every push to `main`:

- Sets up Python 3.13
- Installs dependencies
- Runs pytest test suite (5 tests covering all endpoints)

See the badge at the top of this README for current build status.

## Tech Stack

- Python 3.13
- TensorFlow 2.21 - model training
- FastAPI - REST API serving
- Uvicorn - ASGI server
- scikit-learn - metrics, preprocessing
- NumPy / pandas - data handling
- Docker - containerization
- GitHub Actions - CI/CD
- Render - cloud deployment (Docker runtime)

## Project Structure

- main.py                      - FastAPI application
- test_api.py                  - Pytest test suite (5 tests)
- milk_model.keras             - Trained TensorFlow model
- target_scaler.pkl            - StandardScaler for targets
- requirements.txt             - Python dependencies
- Dockerfile                   - Docker build recipe
- .dockerignore                - Files excluded from Docker build
- .python-version              - Python 3.13.5 pin
- .gitignore                   - Git ignore rules
- LICENSE                      - MIT License
- README.md                    - This file
- .github/workflows/ci.yml     - GitHub Actions CI pipeline
- notebooks/training.ipynb     - Full training pipeline

## How to Run Locally

Option 1 - Python virtual environment:

1. Clone the repository:

   git clone https://github.com/UNarasimha/milk-api.git
   cd milk-api

2. Install dependencies:

   pip install -r requirements.txt

3. Start the API:

   python -m uvicorn main:app --reload --port 8000

4. Open the interactive docs at http://127.0.0.1:8000/docs

Option 2 - Docker:

1. Build the image:

   docker build -t milk-api .

2. Run the container:

   docker run -p 8000:8000 milk-api

3. Open the interactive docs at http://localhost:8000/docs

Option 3 - Run tests:

   pytest test_api.py -v

## Author

U Narasimha - https://github.com/UNarasimha

## License

This project is open-source under the MIT License. See LICENSE file.