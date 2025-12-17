"""
Utility functions and internationalization for Budget Horizon.

This module provides:
- Currency and number formatting
- Budget analysis functions (growth rates, trends, categories)
- Data export functionality
- Multi-language internationalization support
"""

from __future__ import annotations

import locale
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import EU_LANGUAGES, DEFAULT_LANGUAGE


# =============================================================================
# Locale Configuration
# =============================================================================

def _setup_locale() -> None:
    """Configure French locale for number formatting."""
    for loc in ["fr_FR.UTF-8", "fr_FR", "C"]:
        try:
            locale.setlocale(locale.LC_ALL, loc)
            break
        except locale.Error:
            continue


_setup_locale()


# =============================================================================
# Currency and Number Formatting
# =============================================================================

def format_currency(amount: float, currency: str = "EUR") -> str:
    """
    Format currency amounts in French style.

    Args:
        amount: Amount in billions to format.
        currency: Currency symbol (default: EUR).

    Returns:
        Formatted string like "12,5 MdEUR".

    Example:
        >>> format_currency(12.5)
        '12,5 MdEUR'
    """
    if pd.isna(amount) or amount == 0:
        return f"0,0 Md{currency}"

    # Values are already in billions
    suffix = f" Md{currency}"

    # Handle very large values (thousands of billions = trillions)
    if abs(amount) >= 1000:
        amount = amount / 1000
        suffix = f" Md{currency}"

    try:
        formatted = locale.format_string("%.1f", amount, grouping=True)
        return formatted + suffix
    except (ValueError, locale.Error):
        # Fallback: manual French formatting
        return f"{amount:,.1f}".replace(",", " ").replace(".", ",") + suffix


# =============================================================================
# Growth and Analysis Functions
# =============================================================================

def calculate_growth_rate(
    start_value: float, end_value: float, years: int
) -> float:
    """
    Calculate compound annual growth rate (CAGR).

    Args:
        start_value: Initial value.
        end_value: Final value.
        years: Number of years between values.

    Returns:
        Annual growth rate as percentage (e.g., 5.2 for 5.2%).

    Example:
        >>> calculate_growth_rate(100, 150, 5)
        8.45
    """
    if start_value <= 0 or years <= 0:
        return 0.0

    if end_value <= 0:
        return -100.0

    try:
        cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0


def get_top_categories(
    df: pd.DataFrame, year: int | None = None, n: int = 5
) -> pd.DataFrame:
    """
    Get top spending categories for a given year.

    Args:
        df: Budget DataFrame with columns [Annee, Mission, Montant].
        year: Year to analyze (default: latest year in data).
        n: Number of top categories to return.

    Returns:
        DataFrame with top n categories sorted by Montant descending.
    """
    if df.empty:
        return pd.DataFrame()

    if year is None:
        year = df["Annee"].max()

    year_data = df[df["Annee"] == year].copy()

    if year_data.empty:
        return pd.DataFrame()

    top_categories = year_data.groupby("Mission")["Montant"].sum().reset_index()
    return top_categories.sort_values("Montant", ascending=False).head(n)


def calculate_percentage_breakdown(
    df: pd.DataFrame, year: int | None = None
) -> pd.DataFrame:
    """
    Calculate percentage breakdown of budget by mission.

    Args:
        df: Budget DataFrame.
        year: Year to analyze (default: latest year).

    Returns:
        DataFrame with columns [Mission, Montant, Pourcentage].
    """
    if df.empty:
        return pd.DataFrame()

    if year is None:
        year = df["Annee"].max()

    year_data = df[df["Annee"] == year].copy()

    if year_data.empty:
        return pd.DataFrame()

    total_budget = year_data["Montant"].sum()
    if total_budget == 0:
        return pd.DataFrame()

    breakdown = year_data.groupby("Mission")["Montant"].sum().reset_index()
    breakdown["Pourcentage"] = (breakdown["Montant"] / total_budget) * 100
    return breakdown.sort_values("Pourcentage", ascending=False)


def identify_growth_trends(
    df: pd.DataFrame, min_years: int = 3
) -> pd.DataFrame:
    """
    Identify spending trends by analyzing growth patterns.

    Classifies each mission's trend as:
    - Forte croissance (>5% annual)
    - Croissance moderee (2-5% annual)
    - Stable (-2% to 2% annual)
    - Declin modere (-5% to -2% annual)
    - Forte baisse (<-5% annual)

    Args:
        df: Budget DataFrame.
        min_years: Minimum years of data required for analysis.

    Returns:
        DataFrame with trend analysis per mission.
    """
    if df.empty:
        return pd.DataFrame()

    trends = []

    for mission in df["Mission"].unique():
        mission_data = df[df["Mission"] == mission].sort_values("Annee")

        if len(mission_data) < min_years:
            continue

        start_amount = mission_data["Montant"].iloc[0]
        end_amount = mission_data["Montant"].iloc[-1]
        start_year = mission_data["Annee"].iloc[0]
        end_year = mission_data["Annee"].iloc[-1]

        # Total growth
        total_growth = (
            ((end_amount - start_amount) / start_amount) * 100
            if start_amount > 0
            else 0
        )

        # Annual growth rate
        years_span = end_year - start_year
        annual_growth = calculate_growth_rate(start_amount, end_amount, years_span)

        # Volatility (coefficient of variation)
        mean_val = mission_data["Montant"].mean()
        volatility = (
            (mission_data["Montant"].std() / mean_val) * 100 if mean_val > 0 else 0
        )

        # Trend classification
        if annual_growth > 5:
            trend_class = "Forte croissance"
        elif annual_growth > 2:
            trend_class = "Croissance moderee"
        elif annual_growth > -2:
            trend_class = "Stable"
        elif annual_growth > -5:
            trend_class = "Declin modere"
        else:
            trend_class = "Forte baisse"

        trends.append({
            "Mission": mission,
            "Montant_Initial": start_amount,
            "Montant_Final": end_amount,
            "Croissance_Totale_Pct": round(total_growth, 1),
            "Croissance_Annuelle_Pct": annual_growth,
            "Volatilite_Pct": round(volatility, 1),
            "Classification": trend_class,
            "Annees_Analysees": years_span,
            "Points_Donnees": len(mission_data),
        })

    trends_df = pd.DataFrame(trends)
    if not trends_df.empty:
        trends_df = trends_df.sort_values("Croissance_Annuelle_Pct", ascending=False)

    return trends_df


# =============================================================================
# Data Validation
# =============================================================================

def validate_budget_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate budget data quality and provide diagnostics.

    Args:
        df: Budget DataFrame to validate.

    Returns:
        Dictionary with keys:
        - is_valid: Boolean indicating data validity
        - errors: List of critical errors
        - warnings: List of non-critical warnings
        - stats: Dictionary of data statistics
    """
    validation: dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if df.empty:
        validation["is_valid"] = False
        validation["errors"].append("DataFrame is empty")
        return validation

    # Check required columns
    required_columns = ["Annee", "Mission", "Montant"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        validation["is_valid"] = False
        validation["errors"].append(f"Missing required columns: {missing_columns}")
        return validation

    # Check data types
    if not pd.api.types.is_numeric_dtype(df["Annee"]):
        validation["errors"].append("Column 'Annee' must be numeric")

    if not pd.api.types.is_numeric_dtype(df["Montant"]):
        validation["errors"].append("Column 'Montant' must be numeric")

    # Check for negative amounts
    negative_count = (df["Montant"] < 0).sum()
    if negative_count > 0:
        validation["warnings"].append(f"{negative_count} negative budget amounts found")

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation["warnings"].append(f"Missing values: {null_counts.to_dict()}")

    # Check year range
    if "Annee" in df.columns:
        year_range = df["Annee"].max() - df["Annee"].min()
        if year_range < 1:
            validation["warnings"].append("Data covers less than 2 years")

    # Calculate statistics
    validation["stats"] = {
        "total_records": len(df),
        "unique_missions": df["Mission"].nunique() if "Mission" in df.columns else 0,
        "year_range": (
            (int(df["Annee"].min()), int(df["Annee"].max()))
            if "Annee" in df.columns
            else (0, 0)
        ),
        "total_budget_latest": (
            df[df["Annee"] == df["Annee"].max()]["Montant"].sum()
            if "Annee" in df.columns and "Montant" in df.columns
            else 0
        ),
        "avg_mission_budget": (
            df.groupby("Mission")["Montant"].mean().mean()
            if "Mission" in df.columns and "Montant" in df.columns
            else 0
        ),
    }

    validation["is_valid"] = len(validation["errors"]) == 0
    return validation


# =============================================================================
# Data Export
# =============================================================================

def export_data_with_metadata(
    df: pd.DataFrame, predictions_df: pd.DataFrame | None = None
) -> str:
    """
    Export data with metadata in CSV format.

    Args:
        df: Historical budget data.
        predictions_df: Optional predictions data.

    Returns:
        CSV string with metadata header comments.
    """
    lines = []

    # Metadata header
    lines.append("# Donnees Budgetaires de l'Etat Francais")
    lines.append(f"# Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    lines.append(f"# Periode: {df['Annee'].min()}-{df['Annee'].max()}")
    lines.append(f"# Missions analysees: {df['Mission'].nunique()}")
    lines.append("# Montants en milliards d'euros")
    lines.append("")

    # Validation info
    validation = validate_budget_data(df)
    status = "Valide" if validation["is_valid"] else "Erreurs detectees"
    lines.append("# Validation des donnees:")
    lines.append(f"# - Status: {status}")
    lines.append(f"# - Total enregistrements: {validation['stats']['total_records']}")
    lines.append(
        f"# - Budget total (derniere annee): "
        f"{format_currency(validation['stats']['total_budget_latest'])}"
    )
    lines.append("")

    # Column descriptions
    lines.append("# Colonnes:")
    lines.append("# - Annee: Annee budgetaire")
    lines.append("# - Mission: Mission budgetaire de l'Etat")
    lines.append("# - Montant: Montant alloue en milliards d'euros")
    if predictions_df is not None:
        lines.append("# - Montant_Predit: Montant predit (donnees futures)")
        lines.append("# - Confiance: Niveau de confiance de la prediction (0-1)")
    lines.append("")

    # Historical data
    lines.append("# === DONNEES HISTORIQUES ===")
    lines.append(df.to_csv(index=False, encoding="utf-8"))

    # Predictions
    if predictions_df is not None:
        lines.append("")
        lines.append("# === PREDICTIONS ===")
        lines.append(predictions_df.to_csv(index=False, encoding="utf-8"))

    return "\n".join(lines)


# =============================================================================
# Inflation Adjustment
# =============================================================================

def adjust_to_constant_euros(
    df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    base_year: int | None = None,
    amount_col: str = "Montant",
) -> pd.DataFrame:
    """
    Deflate nominal amounts to constant euros using a CPI index.

    Args:
        df: DataFrame with columns [Annee, amount_col].
        cpi_df: DataFrame with columns [Annee, CPI].
        base_year: Year to express amounts in (default: latest year in df).
        amount_col: Name of the amount column to adjust.

    Returns:
        New DataFrame with amounts adjusted to constant euros.
    """
    if df is None or df.empty or cpi_df is None or cpi_df.empty:
        return df

    if base_year is None:
        base_year = int(df["Annee"].max())

    # Verify base year exists in CPI data
    if base_year not in set(cpi_df["Annee"].astype(int)):
        return df

    cpi_df = cpi_df[["Annee", "CPI"]].copy()
    cpi_df["Annee"] = cpi_df["Annee"].astype(int)

    base_cpi = float(cpi_df.loc[cpi_df["Annee"] == base_year, "CPI"].iloc[0])
    if base_cpi == 0:
        return df

    # Normalize CPI so base_year = 100
    cpi_df["CPI_Norm"] = cpi_df["CPI"] / base_cpi * 100.0

    # Merge with main data
    merged = df.merge(
        cpi_df[["Annee", "CPI_Norm"]],
        on="Annee",
        how="left",
        validate="many_to_one",
    )

    # Handle missing CPI values
    merged["CPI_Norm"] = merged["CPI_Norm"].replace({0: np.nan})
    merged["CPI_Norm"] = merged["CPI_Norm"].ffill().bfill()

    # Adjust amounts
    if amount_col in merged.columns:
        merged[amount_col] = merged[amount_col] * (100.0 / merged["CPI_Norm"])

    return merged.drop(columns=["CPI_Norm"])


# =============================================================================
# Internationalization (i18n)
# =============================================================================

# Translation cache to avoid repeated API calls
_TRANSLATION_CACHE: dict[tuple[str, str], str] = {}

# Static translations dictionary
_TRANSLATIONS: dict[str, dict[str, str]] = {
    # Sidebar
    "sidebar.config": {
        "en": "Settings", "fr": "Configuration", "de": "Einstellungen",
        "es": "Ajustes", "it": "Impostazioni", "pt": "Definicoes",
        "nl": "Instellingen", "pl": "Ustawienia",
    },
    "sidebar.language": {
        "en": "Language", "fr": "Langue", "de": "Sprache",
        "es": "Idioma", "it": "Lingua", "pt": "Idioma",
        "nl": "Taal", "pl": "Jezyk",
    },
    "sidebar.data_source": {
        "en": "Data source", "fr": "Source de donnees", "de": "Datenquelle",
        "es": "Fuente de datos", "it": "Fonte dati", "pt": "Fonte de dados",
        "nl": "Gegevensbron", "pl": "Zrodlo danych",
    },
    "sidebar.year_range": {
        "en": "Year range", "fr": "Plage d'annees", "de": "Jahresbereich",
        "es": "Rango de anos", "it": "Intervallo di anni", "pt": "Intervalo de anos",
        "nl": "Jarenbereik", "pl": "Zakres lat",
    },
    "sidebar.load_data": {
        "en": "Load data", "fr": "Charger les donnees", "de": "Daten laden",
        "es": "Cargar datos", "it": "Carica dati", "pt": "Carregar dados",
        "nl": "Gegevens laden", "pl": "Wczytaj dane",
    },
    # Toggles
    "toggle.inflation": {
        "en": "Adjust for inflation (constant euros)",
        "fr": "Ajuster pour l'inflation (euros constants)",
        "de": "Fur Inflation anpassen (konstante Euro)",
        "es": "Ajustar por inflacion (euros constantes)",
        "it": "Correggi per l'inflazione (euro costanti)",
        "pt": "Ajustar pela inflacao (euros constantes)",
        "nl": "Aanpassen voor inflatie (constante euro's)",
        "pl": "Dostosuj do inflacji (stale euro)",
    },
    "toggle.gov_periods": {
        "en": "Show government periods",
        "fr": "Afficher periodes gouvernementales",
        "de": "Regierungsperioden anzeigen",
        "es": "Mostrar periodos de gobierno",
        "it": "Mostra periodi di governo",
        "pt": "Mostrar periodos de governo",
        "nl": "Regeringsperiodes tonen",
        "pl": "Pokaz okresy rzadow",
    },
    "toggle.events": {
        "en": "Show key events",
        "fr": "Afficher evenements majeurs",
        "de": "Schlusselereignisse anzeigen",
        "es": "Mostrar eventos clave",
        "it": "Mostra eventi chiave",
        "pt": "Mostrar eventos-chave",
        "nl": "Belangrijke gebeurtenissen tonen",
        "pl": "Pokaz kluczowe wydarzenia",
    },
    "toggle.debt": {
        "en": "Include debt interest",
        "fr": "Inclure charge de la dette (interets)",
        "de": "Schuldzinsen einbeziehen",
        "es": "Incluir intereses de la deuda",
        "it": "Includi interessi sul debito",
        "pt": "Incluir juros da divida",
        "nl": "Schuldrente opnemen",
        "pl": "Uwzglednij odsetki od dlugu",
    },
    # Tabs
    "tab.evolution": {
        "en": "Time Evolution", "fr": "Evolution Temporelle",
        "de": "Zeitliche Entwicklung", "es": "Evolucion temporal",
        "it": "Evoluzione temporale", "pt": "Evolucao temporal",
        "nl": "Tijdsevolutie", "pl": "Ewolucja w czasie",
    },
    "tab.compare": {
        "en": "Mission Comparison", "fr": "Comparaison Missions",
        "de": "Vergleich der Missionen", "es": "Comparacion de misiones",
        "it": "Confronto missioni", "pt": "Comparacao de missoes",
        "nl": "Missievergelijking", "pl": "Porownanie misji",
    },
    "tab.split": {
        "en": "Budget Breakdown", "fr": "Repartition Budgetaire",
        "de": "Budgetaufteilung", "es": "Desglose del presupuesto",
        "it": "Ripartizione del budget", "pt": "Desagregacao do orcamento",
        "nl": "Budgetverdeling", "pl": "Podzial budzetu",
    },
    "tab.pred": {
        "en": "Predictions", "fr": "Predictions", "de": "Prognosen",
        "es": "Predicciones", "it": "Previsioni", "pt": "Previsoes",
        "nl": "Voorspellingen", "pl": "Prognozy",
    },
    "tab.details": {
        "en": "Detailed Analysis", "fr": "Analyse Detaillee",
        "de": "Detaillierte Analyse", "es": "Analisis detallado",
        "it": "Analisi dettagliata", "pt": "Analise detalhada",
        "nl": "Gedetailleerde analyse", "pl": "Szczegolowa analiza",
    },
    "tab.revenue": {
        "en": "State Revenue", "fr": "Recettes de l'Etat",
        "de": "Staatseinnahmen", "es": "Ingresos del Estado",
        "it": "Entrate dello Stato", "pt": "Receitas do Estado",
        "nl": "Staatsinkomsten", "pl": "Dochody panstwa",
    },
    # Titles
    "title.app": {
        "fr": "Budget de l'Etat Francais - Analyse et Predictions",
        "en": "French State Budget - Analysis and Predictions",
        "de": "Franzosischer Staatshaushalt - Analyse und Vorhersagen",
        "es": "Presupuesto del Estado Frances - Analisis y Predicciones",
        "it": "Bilancio dello Stato francese - Analisi e Previsioni",
        "pt": "Orcamento do Estado Frances - Analise e Previsoes",
        "nl": "Franse Rijksbegroting - Analyse en Voorspellingen",
        "pl": "Budzet Panstwa Francuskiego - Analiza i Prognozy",
    },
    "desc.app": {
        "fr": "Cette application analyse l'evolution des depenses budgetaires de l'Etat francais sur 20 ans et propose des predictions basees sur l'intelligence artificielle.",
        "en": "This application analyzes the evolution of the French state budget over 20 years and provides AI-based predictions.",
        "de": "Diese Anwendung analysiert die Entwicklung des franzosischen Staatshaushalts uber 20 Jahre und bietet KI-basierte Vorhersagen.",
        "es": "Esta aplicacion analiza la evolucion del presupuesto estatal frances durante 20 anos y proporciona predicciones basadas en IA.",
        "it": "Questa applicazione analizza l'evoluzione del bilancio dello Stato francese negli ultimi 20 anni e fornisce previsioni basate sull'intelligenza artificiale.",
        "pt": "Esta aplicacao analisa a evolucao do orcamento do Estado frances ao longo de 20 anos e fornece previsoes baseadas em IA.",
        "nl": "Deze applicatie analyseert de evolutie van de Franse rijksbegroting over 20 jaar en geeft AI-gebaseerde voorspellingen.",
        "pl": "Ta aplikacja analizuje ewolucje budzetu panstwa francuskiego na przestrzeni 20 lat i dostarcza prognozy oparte na sztucznej inteligencji.",
    },
}


def get_eu_languages() -> list[dict[str, str]]:
    """
    Get list of supported EU languages.

    Returns:
        List of dictionaries with 'code' and 'name' keys.
    """
    return EU_LANGUAGES


def _mt_translate_text(text: str, target_lang: str) -> str | None:
    """
    Machine translate text using available translation libraries.

    Tries deep_translator first, then googletrans as fallback.

    Args:
        text: Text to translate.
        target_lang: Target language code (e.g., 'de', 'es').

    Returns:
        Translated text, or None on failure.
    """
    if not text or not target_lang:
        return None

    cache_key = (text, target_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # Try deep_translator
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        if isinstance(translated, str) and translated.strip():
            _TRANSLATION_CACHE[cache_key] = translated
            return translated
    except Exception:
        pass

    # Try googletrans as fallback
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest=target_lang)
        translated = getattr(result, "text", None)
        if isinstance(translated, str) and translated.strip():
            _TRANSLATION_CACHE[cache_key] = translated
            return translated
    except Exception:
        pass

    return None


def translate(key: str, lang: str, default: str | None = None) -> str:
    """
    Get translated text for a key and language.

    Args:
        key: Translation key (e.g., 'sidebar.config').
        lang: Target language code (e.g., 'fr', 'en').
        default: Fallback text if translation not found.

    Returns:
        Translated string.
    """
    lang = (lang or DEFAULT_LANGUAGE).lower()
    entry = _TRANSLATIONS.get(key, {})

    # Exact match
    if lang in entry:
        return entry[lang]

    # Use English as seed for machine translation
    seed = entry.get("en", default if default is not None else key)

    # Try machine translation
    mt_result = _mt_translate_text(seed, lang)
    if mt_result:
        entry[lang] = mt_result
        _TRANSLATIONS[key] = entry
        return mt_result

    # Fallback to English or default
    return entry.get("en", default if default is not None else key)


def _prefill_eu_translations() -> None:
    """Pre-fill missing EU translations with English values."""
    eu_codes = {opt["code"] for opt in EU_LANGUAGES}
    for key, mapping in _TRANSLATIONS.items():
        en_val = mapping.get("en")
        if not en_val:
            continue
        for code in eu_codes:
            if code not in mapping:
                mapping[code] = en_val


_prefill_eu_translations()


# =============================================================================
# Insight Generation
# =============================================================================

def generate_insights(
    df: pd.DataFrame, predictions_df: pd.DataFrame | None = None
) -> list[str]:
    """
    Generate key insights from budget data analysis (French).

    Args:
        df: Historical budget data.
        predictions_df: Optional predictions data.

    Returns:
        List of insight strings.
    """
    insights = []

    if df.empty:
        return ["Aucune donnee disponible pour l'analyse."]

    total_years = df["Annee"].nunique()
    total_missions = df["Mission"].nunique()

    insights.append(
        f"Analyse portant sur {total_years} annees et {total_missions} missions budgetaires"
    )

    # Budget evolution
    if total_years > 1:
        start_year = df["Annee"].min()
        end_year = df["Annee"].max()
        start_total = df[df["Annee"] == start_year]["Montant"].sum()
        end_total = df[df["Annee"] == end_year]["Montant"].sum()

        if start_total > 0:
            total_growth = ((end_total - start_total) / start_total) * 100
            annual_growth = total_growth / (end_year - start_year)

            if total_growth > 0:
                insights.append(
                    f"Le budget total a augmente de {total_growth:.1f}% entre {start_year} "
                    f"et {end_year} (croissance annuelle moyenne: {annual_growth:.1f}%)"
                )
            else:
                insights.append(
                    f"Le budget total a diminue de {abs(total_growth):.1f}% entre "
                    f"{start_year} et {end_year}"
                )

    # Top missions
    latest_year = df["Annee"].max()
    top_missions = get_top_categories(df, latest_year, 3)

    if not top_missions.empty:
        top = top_missions.iloc[0]
        insights.append(
            f"En {latest_year}, '{top['Mission']}' represente la plus grosse mission "
            f"avec {format_currency(top['Montant'])}"
        )

    # Growth trends
    trends = identify_growth_trends(df)
    if not trends.empty:
        fastest = trends.iloc[0]
        if fastest["Croissance_Annuelle_Pct"] > 5:
            insights.append(
                f"'{fastest['Mission']}' affiche la plus forte croissance avec "
                f"+{fastest['Croissance_Annuelle_Pct']:.1f}% par an"
            )

        declining = trends[trends["Croissance_Annuelle_Pct"] < -2]
        if not declining.empty:
            worst = declining.iloc[-1]
            insights.append(
                f"'{worst['Mission']}' montre la plus forte baisse avec "
                f"{worst['Croissance_Annuelle_Pct']:.1f}% par an"
            )

    # Predictions
    if predictions_df is not None and not predictions_df.empty:
        pred_2030 = predictions_df[predictions_df["Annee"] == 2030]["Montant_Predit"].sum()
        current_total = df[df["Annee"] == latest_year]["Montant"].sum()

        if current_total > 0:
            pred_growth = ((pred_2030 - current_total) / current_total) * 100

            if pred_growth > 0:
                insights.append(
                    f"Les predictions suggerent une croissance du budget de "
                    f"{pred_growth:.1f}% d'ici 2030"
                )
            else:
                insights.append(
                    f"Les predictions suggerent une baisse du budget de "
                    f"{abs(pred_growth):.1f}% d'ici 2030"
                )

    return insights


def generate_insights_i18n(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    lang: str,
) -> list[str]:
    """
    Generate localized insights using translation templates.

    Args:
        df: Historical budget data.
        predictions_df: Optional predictions data.
        lang: Target language code.

    Returns:
        List of translated insight strings.
    """
    messages: list[str] = []

    if df is None or df.empty:
        messages.append(
            translate("insights.no_data", lang, "Aucune donnee disponible pour l'analyse.")
        )
        return messages

    total_years = int(df["Annee"].nunique())
    total_missions = int(df["Mission"].nunique()) if "Mission" in df.columns else 0

    header_tpl = translate(
        "insights.header",
        lang,
        "Analyse portant sur {years} annees et {missions} missions budgetaires",
    )
    messages.append(header_tpl.format(years=total_years, missions=total_missions))

    if total_years > 1:
        start_year = int(df["Annee"].min())
        end_year = int(df["Annee"].max())
        start_total = float(df[df["Annee"] == start_year]["Montant"].sum())
        end_total = float(df[df["Annee"] == end_year]["Montant"].sum())

        if start_total > 0:
            total_growth = (end_total - start_total) / start_total * 100.0
            annual_growth = total_growth / (end_year - start_year)

            if total_growth >= 0:
                tpl = translate(
                    "insights.total_up",
                    lang,
                    "Le budget total a augmente de {pct}% entre {start} et {end} "
                    "(croissance annuelle moyenne: {avg}%)",
                )
                messages.append(
                    tpl.format(
                        pct=f"{total_growth:.1f}",
                        start=start_year,
                        end=end_year,
                        avg=f"{annual_growth:.1f}",
                    )
                )
            else:
                tpl = translate(
                    "insights.total_down",
                    lang,
                    "Le budget total a diminue de {pct}% entre {start} et {end}",
                )
                messages.append(
                    tpl.format(
                        pct=f"{abs(total_growth):.1f}",
                        start=start_year,
                        end=end_year,
                    )
                )

    # Top mission
    latest_year = int(df["Annee"].max())
    top = get_top_categories(df, latest_year, 1)
    if top is not None and not top.empty:
        row = top.iloc[0]
        tpl = translate(
            "insights.top_mission",
            lang,
            "En {year}, '{mission}' represente la plus grosse mission avec {amount}",
        )
        messages.append(
            tpl.format(
                year=latest_year,
                mission=row["Mission"],
                amount=format_currency(row["Montant"]),
            )
        )

    # Growth trends
    trends = identify_growth_trends(df)
    if trends is not None and not trends.empty:
        fastest = trends.iloc[0]
        if float(fastest["Croissance_Annuelle_Pct"]) > 5:
            tpl = translate(
                "insights.fastest",
                lang,
                "'{mission}' affiche la plus forte croissance avec +{pct}% par an",
            )
            messages.append(
                tpl.format(
                    mission=fastest["Mission"],
                    pct=f"{float(fastest['Croissance_Annuelle_Pct']):.1f}",
                )
            )

        declining = trends[trends["Croissance_Annuelle_Pct"] < -2]
        if not declining.empty:
            worst = declining.iloc[-1]
            tpl = translate(
                "insights.decline",
                lang,
                "'{mission}' montre la plus forte baisse avec {pct}% par an",
            )
            messages.append(
                tpl.format(
                    mission=worst["Mission"],
                    pct=f"{float(worst['Croissance_Annuelle_Pct']):.1f}",
                )
            )

    # Predictions
    if predictions_df is not None and not predictions_df.empty:
        pred_2030 = float(
            predictions_df[predictions_df["Annee"] == 2030]["Montant_Predit"].sum()
        )
        current_total = float(df[df["Annee"] == latest_year]["Montant"].sum())

        if current_total > 0:
            pred_growth = (pred_2030 - current_total) / current_total * 100.0

            if pred_growth >= 0:
                tpl = translate(
                    "insights.pred_up",
                    lang,
                    "Les predictions suggerent une croissance du budget de {pct}% d'ici 2030",
                )
                messages.append(tpl.format(pct=f"{pred_growth:.1f}"))
            else:
                tpl = translate(
                    "insights.pred_down",
                    lang,
                    "Les predictions suggerent une baisse du budget de {pct}% d'ici 2030",
                )
                messages.append(tpl.format(pct=f"{abs(pred_growth):.1f}"))

        # Find top predicted growth
        growth_data = []
        for mission in predictions_df["Mission"].unique():
            curr = df[(df["Mission"] == mission) & (df["Annee"] == latest_year)]["Montant"]
            pred = predictions_df[
                (predictions_df["Mission"] == mission) & (predictions_df["Annee"] == 2030)
            ]["Montant_Predit"]

            if not curr.empty and not pred.empty and float(curr.iloc[0]) > 0:
                growth = (float(pred.iloc[0]) - float(curr.iloc[0])) / float(curr.iloc[0]) * 100.0
                growth_data.append((mission, growth))

        if growth_data:
            growth_data.sort(key=lambda x: x[1], reverse=True)
            top_mission, top_growth = growth_data[0]
            if top_growth > 10:
                tpl = translate(
                    "insights.pred_top",
                    lang,
                    "'{mission}' devrait connaitre la plus forte croissance avec +{pct}% d'ici 2030",
                )
                messages.append(tpl.format(mission=top_mission, pct=f"{top_growth:.1f}"))

    return messages
