# ZoneTrend

ZoneTrend is a lightweight research pipeline for detecting swing-support/resistance zones from OHLCV data (Yahoo Finance). This README explains how to set up the environment and run the main pipelines.

**Quick Start**
- **Create virtualenv:** Recommended Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# or use the included virtualenv:
# source projectenv/bin/activate
```

- **Install dependencies:**

```bash
pip install -r requirements.txt
# Install test tools if you plan to run tests:
pip install pytest
```

**Run Pipelines**
- **Data pipeline (fetch + preprocess)** — runs a small `dev` symbol set by default:

```bash
python run_data_pipeline.py
# full run (all symbols from config):
python run_data_pipeline.py --mode full
# run specific symbols:
python run_data_pipeline.py --symbols RELIANCE.NS TCS.NS
# skip download step (use existing raw data):
python run_data_pipeline.py --skip-fetch
```

- **Zone detection pipeline** — reads processed CSVs from `data/processed/` and writes zones to `data/zones/`:

```bash
python run_zone_pipeline.py
# full run:
python run_zone_pipeline.py --mode full
# run specific symbols:
python run_zone_pipeline.py --symbols RELIANCE.NS TCS.NS
# skip scoring (faster):
python run_zone_pipeline.py --skip-scoring
# override lookback:
python run_zone_pipeline.py --lookback 10
```

**Run Notebooks**
Open the notebooks for exploration and analysis (Jupyter):

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Run Tests**
Run unit tests with `pytest`:

```bash
pytest -q
```

**Configuration**
- Main configuration: `config/config.yaml` — controls symbol lists, data paths, logging, and preprocessing defaults.
- Zone parameters: `config/zone_config.yaml` — tune detection, merging and scoring behaviour.
- Model parameters: `config/model_config.yaml`.

Edit these YAML files to change behaviour; paths are relative to the project root.

**Data layout**
- `data/raw/`     — raw downloaded CSVs (OHLCV)
- `data/processed/` — cleaned & feature-engineered CSVs (pipeline output)
- `data/zones/`   — detected zone CSVs (pipeline output)
- `data/features/` and `data/labels/` — downstream feature/label outputs

If `data/processed/` is empty, run `python run_data_pipeline.py` first.

**Notes & Troubleshooting**
- Symbols use Yahoo Finance format (NSE stocks end with `.NS`).
- If you encounter missing data during the zone pipeline, ensure the corresponding processed CSV exists in `data/processed/`.
- Logs are written to the `logs/` directory (controlled in `config/config.yaml`).

**Contributing**
- Tests are under `tests/` — add tests for new functionality.
- Follow the existing project structure under `src/`.

**Contact / Next steps**
If you want, I can:
- Add a `Makefile` or `tox` config for convenience
- Add a minimal `setup.py` or `pyproject.toml` to support `pip install -e .`

---

Built for the local repository — adjust paths and Python version as needed.