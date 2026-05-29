# AI-MLOps Stock Prediction System

Production-grade end-to-end MLOps pipeline for NSE/BSE stock prediction.

**Stack:** OpenBB · LightGBM · XGBoost · PyTorch LSTM · Amazon Chronos-2 · AutoGluon · FinBERT · IBM Granite 4.1-3B · MLflow 3.12 · FastAPI · Evidently AI · DVC

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy environment file
```bash
cp .env.example .env
# Edit .env if you have IBM watsonx or NewsAPI credentials
```

### 3. Run full pipeline (manual mode — no Docker needed)
```bash
python main.py --mode ingest        # Download 5y OHLCV data for all stocks
python main.py --mode features      # Engineer 60+ features + sentiment labels
python main.py --mode train --trainer manual --skip-autogluon   # Train all models
python main.py --mode compare       # View side-by-side comparison table
python main.py --mode predict --symbol RELIANCE.NS              # Single prediction
```

### 4. Start FastAPI server
```bash
python main.py --mode serve
# → API running at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### 5. Full pipeline in one command
```bash
python main.py --mode full-pipeline --trainer manual
```

---

## CLI Reference

| Command | Description |
|---|---|
| `--mode ingest` | Download OHLCV data via OpenBB / yfinance |
| `--mode validate` | Pandera data quality checks |
| `--mode features` | Engineer 60+ technical indicators + FinBERT sentiment |
| `--mode train --trainer manual` | Train all models, save `.pkl`/`.pt` locally |
| `--mode train --trainer mlflow` | Train + log everything to MLflow |
| `--mode compare` | Side-by-side model comparison table |
| `--mode monitor` | Evidently AI drift report |
| `--mode predict --symbol X` | Single stock prediction (CLI output) |
| `--mode serve` | Start FastAPI service |
| `--mode full-pipeline` | Ingest → Features → Train → Compare |

**Common flags:**
```bash
--symbols RELIANCE.NS TCS.NS    # Override stock list
--symbol RELIANCE.NS            # Single stock
--trainer manual|mlflow         # Training mode
--skip-autogluon                # Skip AutoGluon (faster first run)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health + loaded models |
| GET | `/models` | List all trained model artifacts |
| POST | `/predict` | `{"symbol": "RELIANCE.NS"}` → prediction |
| POST | `/predict/batch` | Multiple symbols at once |
| GET | `/recommendation/{symbol}` | Signal + confidence + explanation |
| GET | `/sentiment/{symbol}` | Live FinBERT news sentiment |
| POST | `/ask` | `{"query": "Which stocks to buy today?"}` |
| GET | `/comparison` | All models performance JSON |
| GET | `/drift` | Latest Evidently AI drift status |
| POST | `/retrain` | Trigger background retraining |

---

## Project Structure

```
AI-MLOps-Solution/
├── config/config.yaml          ← All settings (stocks, thresholds, model params)
├── src/
│   ├── data/                   ← ingestion, validation, feature engineering
│   ├── slm/                    ← FinBERT sentiment, Granite explainer, news scraper
│   ├── models/                 ← LightGBM, XGBoost, LSTM, Chronos-2, AutoGluon, Ensemble
│   ├── training/               ← manual_trainer, mlflow_trainer
│   ├── evaluation/             ← evaluator, Evidently monitoring
│   └── api/app.py              ← FastAPI service
├── docker/                     ← Dockerfiles + docker-compose.yaml
├── models/manual/              ← Saved .pkl / .pt artifacts (DVC-tracked)
├── data/                       ← raw CSVs, processed Parquet, news cache
├── reports/                    ← comparison CSVs, drift HTML reports
├── notebooks/exploration.ipynb ← EDA + SHAP + Chronos-2 demo
├── main.py                     ← CLI entry point
└── requirements.txt
```

---

## Adding New Stocks

Edit `config/config.yaml` — no code changes needed:
```yaml
stocks:
  symbols:
    - RELIANCE.NS
    - BAJFINANCE.NS   # ← add here
```

---

## Model Versioning with DVC

```bash
# Initialise DVC (one-time)
dvc init
dvc remote add -d local_cache /tmp/dvc-cache

# After each training run — DVC auto-tracks .pkl files
git log --oneline                                          # view versions

# Roll back ALL models to a previous version
git checkout <commit-hash> && dvc checkout

# Roll back a SINGLE stock's model
git checkout <commit-hash> -- models/manual/HDFCBANK.NS_lgbm.pkl.dvc
dvc checkout models/manual/HDFCBANK.NS_lgbm.pkl
```

---

## MLflow Experiment Tracking

```bash
# Start MLflow server
docker-compose -f docker/docker-compose.yaml up mlflow -d
# → UI at http://localhost:5000

# Train with MLflow logging
python main.py --mode train --trainer mlflow
```

---

## Docker Deployment (requires Docker Desktop)

```bash
docker-compose -f docker/docker-compose.yaml up
```

| Service | Port | Description |
|---|---|---|
| mlflow | 5000 | Experiment tracking UI |
| api | 8000 | FastAPI + FinBERT |
| ollama | 11434 | IBM Granite 4.1-3B explanations |

---

## Enable IBM Granite Explanations (optional)

```bash
brew install ollama
ollama pull granite4.1:3b
# Restart the API — explanations will now be AI-generated
```

---

## Accuracy Expectations

| Model | Expected Range |
|---|---|
| LightGBM alone | 65–72% |
| XGBoost alone | 63–70% |
| PyTorch LSTM | 60–68% |
| Chronos-2 (zero-shot) | 58–66% |
| AutoGluon baseline | 66–74% |
| **Stacking Ensemble** | **70–82%** |
| **+ FinBERT sentiment** | **73–85%** |

---

*Stack: 2026 best-in-class | Python 3.12 | MLflow 3.12 | PyTorch 2.7+*
