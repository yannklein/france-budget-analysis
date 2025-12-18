# Budget Horizon

**Interactive French Government Budget Analysis with AI-Powered Predictions**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data visualization and analysis tool for exploring 20 years of French state budget data, featuring machine learning predictions, inflation adjustments, and multi-language support.

![Budget Horizon Screenshot](https://via.placeholder.com/800x400?text=Budget+Horizon+Dashboard)

## Features

### Data Visualization
- **Time Evolution Charts**: Track budget spending trends across missions over time
- **Mission Comparison**: Compare budget allocations with interactive bar and pie charts
- **Budget Breakdown**: Stacked area charts showing composition evolution
- **Revenue Analysis**: State revenue tracking and revenue-expense differentials

### AI-Powered Predictions
- **Ensemble ML Models**: Combines Linear Regression and Random Forest for robust predictions
- **2025-2030 Forecasts**: Project future budget spending with confidence intervals
- **Growth Constraints**: Intelligent bounds prevent unrealistic predictions

### Analysis Tools
- **Inflation Adjustment**: View amounts in constant euros with selectable base year
- **Government Period Overlays**: Visualize budget evolution across different administrations
- **Economic Event Markers**: Highlight key events (2008 crisis, COVID-19, etc.)
- **Trend Classification**: Automatic categorization of growth patterns

### Internationalization
- **8 EU Languages**: French, English, German, Spanish, Italian, Portuguese, Dutch, Polish
- **Dynamic Translation**: Machine translation fallback for missing translations

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python 3.11+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Data Source** | data.economie.gouv.fr API |
| **Translation** | deep-translator, googletrans |

## Project Structure

```
BudgetHorizon/
├── app.py              # Main Streamlit application
├── config.py           # Configuration constants and settings
├── data_fetcher.py     # API data retrieval and processing
├── predictor.py        # ML-based budget predictions
├── utils.py            # Utility functions and i18n
├── account_name.json   # French accounting code mappings
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit configuration
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/BudgetHorizon.git
   cd BudgetHorizon
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Using uv (faster alternative)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. **Select Language**: Choose your preferred language from the sidebar
2. **Configure Analysis**:
   - Enable/disable inflation adjustment
   - Toggle government period overlays
   - Include/exclude debt interest
3. **Set Parameters**:
   - Select account hierarchy level (1-3)
   - Choose year range (2015-2024)
4. **Load Data**: Click "Load Data" to fetch budget data
5. **Explore**: Navigate through the 6 analysis tabs

## Data Sources

| Source | Description |
|--------|-------------|
| [data.economie.gouv.fr](https://data.economie.gouv.fr/) | Official French government budget data |
| [INSEE](https://www.insee.fr/) | Consumer Price Index (CPI) for inflation |
| [Banque de France](https://www.banque-france.fr/) | Public debt statistics |

## Architecture

### Data Flow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ data.gouv.fr    │────▶│ DataFetcher  │────▶│   app.py    │
│ API             │     │              │     │ (Streamlit) │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │                    │
                               ▼                    ▼
                        ┌──────────────┐     ┌─────────────┐
                        │  predictor   │────▶│   Charts    │
                        │  (ML Models) │     │   & Export  │
                        └──────────────┘     └─────────────┘
```

### ML Prediction Pipeline

1. **Feature Engineering**: Year trends, economic cycles, growth rates, moving averages
2. **Model Training**: Separate models per budget mission
3. **Ensemble Prediction**: Weighted average based on R² scores
4. **Constraint Application**: Realistic growth bounds (50%-150% of previous values)

## Configuration

### Environment Variables (Optional)

```bash
# For enhanced translation features
GOOGLE_TRANSLATE_API_KEY=your_key_here
```

### Streamlit Configuration

Edit `.streamlit/config.toml` for:
- Server settings (port, headless mode)
- Theme customization
- CORS and security settings

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app.py` as the main file
5. Deploy

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## Development

### Code Quality Tools

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### Project Conventions

- **Type Hints**: All functions include type annotations
- **Docstrings**: Google-style docstrings for public APIs
- **Column Names**: ASCII-only (e.g., `Annee` instead of `Année`)
- **Configuration**: Centralized in `config.py`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- French government for open data initiatives
- Streamlit team for the framework
- Plotly for interactive visualizations
- scikit-learn community for ML tools

---

**Built for transparent government budget analysis**
