"""
Configuration constants for the Budget Horizon application.

This module centralizes all configuration values, API endpoints, and constants
used throughout the application to improve maintainability and clarity.
"""

from typing import Dict, List, TypedDict


# =============================================================================
# API Configuration
# =============================================================================

API_ENDPOINTS = {
    "balances": (
        "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
        "balances_des_comptes_etat/records"
        "?select=annee,compte,sum(balance_sortie*multiplicateur)%20as%20montant"
        "&group_by=annee,compte"
        "&order_by=annee%20DESC,compte"
        "&limit=100000"
    ),
}

API_TIMEOUT = 30  # seconds


# =============================================================================
# Data Processing Configuration
# =============================================================================

# Year range constraints
MIN_DATA_YEAR = 2015
MAX_DATA_YEAR = 2024
DEFAULT_PREDICTION_YEARS = list(range(2025, 2031))

# Value constraints for data sanitization
MAX_MONETARY_VALUE = 1e6  # In billions EUR - upper clip
MIN_MONETARY_VALUE = -1e6  # In billions EUR - lower clip
MIN_PREDICTION_VALUE = 0.1  # Minimum prediction value in billions

# Growth rate constraints for predictions
MAX_ANNUAL_GROWTH_RATE = 1.5  # 150% max annual growth
MIN_VALUE_RETENTION = 0.5  # Minimum 50% of last known value


# =============================================================================
# Debt Interest Configuration
# =============================================================================

DEBT_CONFIG = {
    "base_year": 2005,
    "base_value_billions": 40.0,  # Base debt interest in billions EUR
    "baseline_drift": 0.005,  # 0.5% annual baseline drift
    "crisis_shock_2008_2012": 0.02,  # 2% additional during 2008-2012
    "crisis_shock_2022_2024": 0.03,  # 3% additional during 2022-2024
}


# =============================================================================
# Inflation (CPI) Data
# =============================================================================

# Approximate yearly inflation rates for France (%)
INFLATION_RATES: Dict[int, float] = {
    2005: 1.9, 2006: 1.7, 2007: 1.5, 2008: 2.8, 2009: 0.1,
    2010: 1.5, 2011: 2.1, 2012: 2.0, 2013: 0.9, 2014: 0.5,
    2015: 0.1, 2016: 0.2, 2017: 1.0, 2018: 1.8, 2019: 1.1,
    2020: 0.5, 2021: 1.6, 2022: 5.2, 2023: 4.9, 2024: 2.5,
    2025: 2.0, 2026: 2.0, 2027: 2.0, 2028: 2.0, 2029: 2.0, 2030: 2.0,
}

DEFAULT_INFLATION_RATE = 2.0  # Default inflation rate for missing years


# =============================================================================
# Economic Cycle Configuration (for ML predictions)
# =============================================================================

class EconomicEvent(TypedDict):
    years: Dict[int, float]
    description: str


ECONOMIC_CYCLES: Dict[str, EconomicEvent] = {
    "crisis": {
        "years": {
            2008: -0.8, 2009: -1.0,  # Financial crisis
            2020: -0.9, 2021: -0.5,  # COVID crisis
            2011: -0.3, 2012: -0.4,  # European debt crisis
        },
        "description": "Economic crisis periods"
    },
    "boom": {
        "years": {
            2006: 0.5, 2007: 0.6,    # Pre-crisis boom
            2017: 0.3, 2018: 0.4, 2019: 0.3,  # Economic recovery
        },
        "description": "Economic growth periods"
    },
}


# =============================================================================
# Government Periods (Prime Ministers of France)
# =============================================================================

class GovernmentPeriod(TypedDict):
    label: str
    start: int
    end: int
    color: str


GOVERNMENT_PERIODS: List[GovernmentPeriod] = [
    {"label": "Gouv. de Villepin", "start": 2005, "end": 2007, "color": "#9b59b6"},
    {"label": "Gouv. Fillon", "start": 2007, "end": 2012, "color": "#2980b9"},
    {"label": "Gouv. Ayrault", "start": 2012, "end": 2014, "color": "#16a085"},
    {"label": "Gouv. Valls", "start": 2014, "end": 2016, "color": "#27ae60"},
    {"label": "Gouv. Cazeneuve", "start": 2016, "end": 2017, "color": "#2c3e50"},
    {"label": "Gouv. Philippe", "start": 2017, "end": 2020, "color": "#f39c12"},
    {"label": "Gouv. Castex", "start": 2020, "end": 2022, "color": "#d35400"},
    {"label": "Gouv. Borne", "start": 2022, "end": 2024, "color": "#c0392b"},
    {"label": "Gouv. Attal", "start": 2024, "end": 2030, "color": "#8e44ad"},
]


# =============================================================================
# Key Economic Events (for chart annotations)
# =============================================================================

class KeyEvent(TypedDict):
    year: int
    label: str


KEY_EVENTS: List[KeyEvent] = [
    {"year": 2008, "label": "Crise financiere mondiale"},
    {"year": 2009, "label": "Recession et relance"},
    {"year": 2011, "label": "Crise dette zone euro"},
    {"year": 2015, "label": "Securite interieure renforcee"},
    {"year": 2020, "label": "COVID-19: plans de soutien"},
    {"year": 2022, "label": "Crise energie/inflation"},
]


# =============================================================================
# UI Configuration
# =============================================================================

CHART_HEIGHT = {
    "default": 500,
    "large": 600,
    "small": 400,
}

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Budget Horizon",
    "page_icon": "chart_with_upwards_trend",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


# =============================================================================
# Machine Learning Configuration
# =============================================================================

ML_CONFIG = {
    "min_data_points": 3,  # Minimum data points for ML models
    "random_forest_estimators": 100,
    "random_seed": 42,
    "simple_linear_max_growth": 1.3,  # 30% max annual growth for simple model
    "simple_linear_max_decline": 0.7,  # 30% max annual decline
}


# =============================================================================
# Supported Languages
# =============================================================================

class LanguageOption(TypedDict):
    code: str
    name: str


EU_LANGUAGES: List[LanguageOption] = [
    {"code": "fr", "name": "Francais"},
    {"code": "en", "name": "English"},
    {"code": "de", "name": "Deutsch"},
    {"code": "es", "name": "Espanol"},
    {"code": "it", "name": "Italiano"},
    {"code": "pt", "name": "Portugues"},
    {"code": "nl", "name": "Nederlands"},
    {"code": "pl", "name": "Polski"},
]

DEFAULT_LANGUAGE = "fr"
