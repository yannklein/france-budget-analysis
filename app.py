"""
Budget Horizon - French State Budget Analysis Application.

A Streamlit application for analyzing French government budget data
with AI-powered predictions, inflation adjustments, and multi-language support.

French Accounting Structure:
- Accounts 1-5: Balance Sheet (Bilan)
- Account 6: Expenses (Depenses)
- Account 7: Revenue (Recettes)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import GOVERNMENT_PERIODS, KEY_EVENTS, PAGE_CONFIG
from data_fetcher import DataFetcher
from predictor import BudgetPredictor
from utils import (
    adjust_to_constant_euros,
    calculate_growth_rate,
    format_currency,
    generate_insights_i18n,
    get_eu_languages,
    get_top_categories,
    translate,
)

# =============================================================================
# Constants and Configuration
# =============================================================================

# Load account mappings
_account_file = Path(__file__).parent / "account_name.json"
with open(_account_file, encoding="utf-8") as f:
    COMPTES: dict[str, str] = json.load(f)

# Account type definitions (French accounting system)
ACCOUNT_TYPES = {
    "balance_sheet": {
        "prefixes": ["1", "2", "3", "4", "5"],
        "icon": "balance_scale",
        "label_fr": "Bilan (Comptes 1-5)",
        "label_en": "Balance Sheet (Accounts 1-5)",
        "description_fr": "Comptes de capitaux, immobilisations, stocks, tiers et finances",
        "description_en": "Capital, fixed assets, inventory, third parties, and financial accounts",
    },
    "expenses": {
        "prefixes": ["6"],
        "icon": "money_with_wings",
        "label_fr": "Depenses (Compte 6)",
        "label_en": "Expenses (Account 6)",
        "description_fr": "Charges et depenses de l'Etat",
        "description_en": "State expenses and charges",
    },
    "revenue": {
        "prefixes": ["7"],
        "icon": "chart_increasing",
        "label_fr": "Recettes (Compte 7)",
        "label_en": "Revenue (Account 7)",
        "description_fr": "Produits et recettes de l'Etat",
        "description_en": "State products and revenue",
    },
}

# Default values for query params
DEFAULTS = {
    "view": "expenses",
    "lang": "fr",
    "c1": "Tous",
    "c2": "",
    "c3": "",
    "y1": 2015,
    "y2": 2024,
    "lvl": 1,
    "inf": False,
    "gov": False,
    "evt": True,
    "debt": False,
}


def filter_comptes_by_type(account_type: str) -> dict[str, str]:
    """Filter COMPTES dictionary by account type."""
    prefixes = ACCOUNT_TYPES[account_type]["prefixes"]
    return {
        code: name
        for code, name in COMPTES.items()
        if any(code.startswith(p) for p in prefixes)
    }


def get_base_compte_for_type(account_type: str) -> str:
    """Get default base compte prefix for account type."""
    return ACCOUNT_TYPES[account_type]["prefixes"][0]


# =============================================================================
# Query Parameter Helpers
# =============================================================================

def get_query_param(key: str, default=None, param_type: type = str):
    """Get a query parameter with type conversion."""
    params = st.query_params
    if key not in params:
        return default if default is not None else DEFAULTS.get(key)

    value = params[key]

    if param_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif param_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default if default is not None else DEFAULTS.get(key)
    return value


def update_query_params(**kwargs) -> None:
    """Update multiple query parameters at once."""
    params = dict(st.query_params)
    for key, value in kwargs.items():
        if value is None or value == DEFAULTS.get(key):
            # Remove default values to keep URL clean
            params.pop(key, None)
        else:
            # Convert booleans to lowercase strings
            if isinstance(value, bool):
                params[key] = "true" if value else "false"
            else:
                params[key] = str(value)
    st.query_params.update(params)


def sync_param(key: str, value, default=None) -> None:
    """Sync a single parameter to query params if changed from default."""
    effective_default = default if default is not None else DEFAULTS.get(key)
    if value != effective_default:
        if isinstance(value, bool):
            st.query_params[key] = "true" if value else "false"
        else:
            st.query_params[key] = str(value)
    elif key in st.query_params:
        del st.query_params[key]


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title=PAGE_CONFIG["page_title"],
    page_icon=PAGE_CONFIG["page_icon"],
    layout=PAGE_CONFIG["layout"],
    initial_sidebar_state=PAGE_CONFIG["initial_sidebar_state"],
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "data_loaded": False,
        "budget_data": None,
        "predictions": None,
        "revenue_data": None,
        "balance_data": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# =============================================================================
# Read Initial Values from Query Params
# =============================================================================

# Get initial values from URL query params
init_view = get_query_param("view", DEFAULTS["view"])
init_lang = get_query_param("lang", DEFAULTS["lang"])
init_c1 = get_query_param("c1", DEFAULTS["c1"])
init_c2 = get_query_param("c2", DEFAULTS["c2"])
init_c3 = get_query_param("c3", DEFAULTS["c3"])
init_y1 = get_query_param("y1", DEFAULTS["y1"], int)
init_y2 = get_query_param("y2", DEFAULTS["y2"], int)
init_lvl = get_query_param("lvl", DEFAULTS["lvl"], int)
init_inf = get_query_param("inf", DEFAULTS["inf"], bool)
init_gov = get_query_param("gov", DEFAULTS["gov"], bool)
init_evt = get_query_param("evt", DEFAULTS["evt"], bool)
init_debt = get_query_param("debt", DEFAULTS["debt"], bool)

# Validate view
if init_view not in ACCOUNT_TYPES:
    init_view = DEFAULTS["view"]


# =============================================================================
# Sidebar Configuration
# =============================================================================

# Language selection
lang_options = get_eu_languages()
lang_codes = [opt["code"] for opt in lang_options]
lang_names = {opt["code"]: opt["name"] for opt in lang_options}

# Get initial language index
init_lang_idx = lang_codes.index(init_lang) if init_lang in lang_codes else 0

current_lang = st.sidebar.selectbox(
    translate("sidebar.language", "fr", "Langue"),
    options=lang_codes,
    index=init_lang_idx,
    format_func=lambda c: lang_names.get(c, c),
    key="lang_select",
)

# Sync language to URL
sync_param("lang", current_lang)

st.sidebar.header(translate("sidebar.config", current_lang, "Configuration"))

# =============================================================================
# Main View Selector
# =============================================================================

st.sidebar.subheader("Type de vue")

view_options = {
    "balance_sheet": "Bilan (1-5)" if current_lang == "fr" else "Balance Sheet (1-5)",
    "expenses": "Depenses (6)" if current_lang == "fr" else "Expenses (6)",
    "revenue": "Recettes (7)" if current_lang == "fr" else "Revenue (7)",
}

view_keys = list(view_options.keys())
init_view_idx = view_keys.index(init_view) if init_view in view_keys else 1

selected_view = st.sidebar.radio(
    "Selectionner le type de compte",
    options=view_keys,
    format_func=lambda x: view_options[x],
    index=init_view_idx,
    help="Les comptes 1-5 sont le bilan, 6 les depenses, 7 les recettes",
    key="view_select",
)

# Sync view to URL
sync_param("view", selected_view)

# =============================================================================
# Dynamic Account Filter (based on selected view)
# =============================================================================

# Filter COMPTES based on selected view
filtered_comptes = filter_comptes_by_type(selected_view)
view_prefixes = ACCOUNT_TYPES[selected_view]["prefixes"]

st.sidebar.markdown("---")
st.sidebar.subheader("Filtre des comptes")

# Level 1: Main account category within the view
compte_lvl_1_options = ["Tous"] + [
    code for code in filtered_comptes.keys() if len(code) == 1
]

# Validate init_c1 for current view
if init_c1 != "Tous" and init_c1 not in compte_lvl_1_options:
    init_c1 = "Tous"

init_c1_idx = compte_lvl_1_options.index(init_c1) if init_c1 in compte_lvl_1_options else 0

compte_lvl_1 = st.sidebar.selectbox(
    "Compte principal",
    compte_lvl_1_options,
    index=init_c1_idx,
    format_func=lambda key: (
        "Tous les comptes" if key == "Tous" else f"{key} - {filtered_comptes.get(key, 'Inconnu')}"
    ),
    key="c1_select",
)

# Sync c1 to URL
sync_param("c1", compte_lvl_1)

# Level 2: Sub-account (filtered based on level 1)
if compte_lvl_1 != "Tous":
    compte_lvl_2_options = [""] + [
        code
        for code in filtered_comptes.keys()
        if code.startswith(compte_lvl_1) and len(code) == 2
    ]
else:
    compte_lvl_2_options = [""]

# Validate init_c2
if init_c2 and init_c2 not in compte_lvl_2_options:
    init_c2 = ""

init_c2_idx = compte_lvl_2_options.index(init_c2) if init_c2 in compte_lvl_2_options else 0

compte_lvl_2 = st.sidebar.selectbox(
    "Sous-compte (niveau 2)",
    compte_lvl_2_options,
    index=init_c2_idx,
    format_func=lambda key: (
        "Tous" if key == "" else f"{key} - {filtered_comptes.get(key, 'Inconnu')}"
    ),
    key="c2_select",
)

# Sync c2 to URL
sync_param("c2", compte_lvl_2)

# Level 3: Detail account (filtered based on level 2)
if compte_lvl_2:
    compte_lvl_3_options = [""] + [
        code
        for code in filtered_comptes.keys()
        if code.startswith(compte_lvl_2) and len(code) == 3
    ]
else:
    compte_lvl_3_options = [""]

# Validate init_c3
if init_c3 and init_c3 not in compte_lvl_3_options:
    init_c3 = ""

init_c3_idx = compte_lvl_3_options.index(init_c3) if init_c3 in compte_lvl_3_options else 0

compte_lvl_3 = st.sidebar.selectbox(
    "Detail (niveau 3)",
    compte_lvl_3_options,
    index=init_c3_idx,
    format_func=lambda key: (
        "Tous" if key == "" else f"{key} - {filtered_comptes.get(key, 'Inconnu')}"
    ),
    key="c3_select",
)

# Sync c3 to URL
sync_param("c3", compte_lvl_3)


# Determine the effective base compte
def get_effective_base_compte() -> str:
    """Get the effective base compte based on user selection."""
    if compte_lvl_3:
        return compte_lvl_3
    elif compte_lvl_2:
        return compte_lvl_2
    elif compte_lvl_1 != "Tous":
        return compte_lvl_1
    else:
        return get_base_compte_for_type(selected_view)


# =============================================================================
# Analysis Options
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Options d'analyse")

adjust_inflation = st.sidebar.checkbox(
    translate(
        "toggle.inflation", current_lang, "Ajuster pour l'inflation (euros constants)"
    ),
    value=init_inf,
    key="inf_check",
)
sync_param("inf", adjust_inflation)

show_governments = st.sidebar.checkbox(
    translate("toggle.gov_periods", current_lang, "Afficher periodes gouvernementales"),
    value=init_gov,
    key="gov_check",
)
sync_param("gov", show_governments)

show_key_events = st.sidebar.checkbox(
    translate("toggle.events", current_lang, "Afficher evenements majeurs"),
    value=init_evt,
    key="evt_check",
)
sync_param("evt", show_key_events)

# Debt interest option only for expenses view
include_debt_interest = False
if selected_view == "expenses":
    include_debt_interest = st.sidebar.checkbox(
        translate("toggle.debt", current_lang, "Inclure charge de la dette"),
        value=init_debt,
        key="debt_check",
    )
    sync_param("debt", include_debt_interest)

# =============================================================================
# Year Range and Detail Level
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Periode et detail")

# Validate year range
init_y1 = max(2015, min(2024, init_y1))
init_y2 = max(2015, min(2024, init_y2))
if init_y1 > init_y2:
    init_y1, init_y2 = init_y2, init_y1

year_range = st.sidebar.slider(
    translate("sidebar.year_range", current_lang, "Plage d'annees"),
    min_value=2015,
    max_value=2024,
    value=(init_y1, init_y2),
    key="year_slider",
)

# Sync year range to URL
sync_param("y1", year_range[0])
sync_param("y2", year_range[1])

# Validate acc_level
init_lvl = max(1, min(3, init_lvl))

acc_level_range = st.sidebar.slider(
    "Niveau de detail des comptes",
    min_value=1,
    max_value=3,
    value=init_lvl,
    help="1 = agregats principaux, 3 = detail maximum",
    key="lvl_slider",
)

# Sync level to URL
sync_param("lvl", acc_level_range)


# =============================================================================
# Share URL Button
# =============================================================================

st.sidebar.markdown("---")
if st.sidebar.button("Copier le lien de partage", use_container_width=True):
    st.sidebar.success("Lien copie! Utilisez l'URL de votre navigateur.")
    st.sidebar.info("L'URL contient tous vos filtres actuels.")


# =============================================================================
# Data Loading
# =============================================================================

def load_data_for_view() -> None:
    """Load data based on selected view and filters."""
    base_compte = get_effective_base_compte()
    fetcher = DataFetcher()

    try:
        data = fetcher.fetch_budget_data(
            start_year=year_range[0],
            end_year=year_range[1],
            acc_level_range=acc_level_range,
            base_compte=base_compte,
        )

        if selected_view == "expenses":
            st.session_state.budget_data = data
            if year_range[1] >= 2024 and not data.empty:
                predictor = BudgetPredictor()
                st.session_state.predictions = predictor.predict_future_spending(data)
            else:
                st.session_state.predictions = None
        elif selected_view == "revenue":
            st.session_state.revenue_data = data
        else:
            st.session_state.balance_data = data

        st.session_state.data_loaded = True

    except Exception as e:
        st.sidebar.error(f"Erreur: {str(e)}")
        st.session_state.data_loaded = False


if st.sidebar.button(
    "Charger les donnees",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Chargement des donnees..."):
        load_data_for_view()
        st.sidebar.success("Donnees chargees!")


# =============================================================================
# Helper Functions for Charts
# =============================================================================

def add_period_overlays(
    fig: go.Figure, periods: list[dict], min_year: int, max_year: int
) -> None:
    """Add government period overlays to a figure."""
    if not show_governments:
        return
    for p in periods:
        x0 = max(min_year, p["start"])
        x1 = min(max_year, p["end"])
        if x1 <= min_year or x0 >= max_year:
            continue
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=p["color"],
            opacity=0.08,
            line_width=0,
            annotation_text=p["label"],
            annotation_position="top left",
        )


def add_event_markers(
    fig: go.Figure, events: list[dict], min_year: int, max_year: int
) -> None:
    """Add key event markers to a figure."""
    if not show_key_events:
        return
    for e in events:
        x = int(e["year"])
        if x < min_year or x > max_year:
            continue
        fig.add_vline(x=x, line_dash="dot", line_color="#7f8c8d")
        fig.add_annotation(
            x=x,
            y=1.02,
            xref="x",
            yref="paper",
            text=e["label"],
            showarrow=False,
            font=dict(size=10, color="#7f8c8d"),
            align="left",
        )


def apply_inflation_adjustment(
    df: pd.DataFrame, amount_col: str = "Montant"
) -> tuple[pd.DataFrame, str]:
    """Apply inflation adjustment if enabled."""
    if not adjust_inflation or df is None or df.empty:
        return df, "Montant (Milliards EUR)"

    base_year = int(df["Annee"].max())
    fetcher = DataFetcher()
    cpi_df = fetcher.get_cpi_series(int(df["Annee"].min()), 2030)
    adjusted_df = adjust_to_constant_euros(
        df, cpi_df, base_year=base_year, amount_col=amount_col
    )
    return adjusted_df, f"Montant (EUR constants {base_year})"


# =============================================================================
# Main Title and Description
# =============================================================================

view_info = ACCOUNT_TYPES[selected_view]
view_label = view_info["label_fr"] if current_lang == "fr" else view_info["label_en"]
view_desc = view_info["description_fr"] if current_lang == "fr" else view_info["description_en"]

st.title(f"Budget Horizon - {view_label}")
st.markdown(view_desc)


# =============================================================================
# Main Content Area
# =============================================================================

if not st.session_state.data_loaded:
    st.info("Utilisez la barre laterale pour charger les donnees.")
    st.markdown("### Structure des comptes disponibles")

    sample_accounts = list(filtered_comptes.items())[:10]
    sample_df = pd.DataFrame(sample_accounts, columns=["Code", "Description"])
    st.dataframe(sample_df, use_container_width=True)

else:
    # =============================================================================
    # BALANCE SHEET VIEW (Accounts 1-5)
    # =============================================================================
    if selected_view == "balance_sheet":
        df = st.session_state.balance_data
        if df is None or df.empty:
            st.warning("Aucune donnee de bilan disponible. Cliquez sur 'Charger les donnees'.")
        else:
            df, montant_label = apply_inflation_adjustment(df)

            col1, col2, col3 = st.columns(3)
            latest_year = int(df["Annee"].max())
            earliest_year = int(df["Annee"].min())
            latest_total = df[df["Annee"] == latest_year]["Montant"].sum()
            earliest_total = df[df["Annee"] == earliest_year]["Montant"].sum()
            total_growth = ((latest_total - earliest_total) / earliest_total) * 100 if earliest_total else 0

            with col1:
                st.metric(f"Total Bilan {latest_year}", format_currency(latest_total))
            with col2:
                st.metric("Evolution totale", f"{total_growth:.1f}%", f"depuis {earliest_year}")
            with col3:
                st.metric("Nombre de comptes", len(df["Mission"].unique()))

            tab1, tab2, tab3 = st.tabs(["Evolution", "Comparaison", "Repartition"])

            with tab1:
                st.subheader("Evolution des Comptes de Bilan")
                fig = px.line(
                    df, x="Annee", y="Montant", color="Mission",
                    title="Evolution des Comptes de Bilan",
                    labels={"Montant": montant_label, "Annee": "Annee"},
                )
                fig.update_layout(height=600, hovermode="x unified")
                fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2))
                add_period_overlays(fig, GOVERNMENT_PERIODS, earliest_year, latest_year)
                add_event_markers(fig, KEY_EVENTS, earliest_year, latest_year)
                st.plotly_chart(fig, use_container_width=True)

                total_df = df.groupby("Annee", as_index=False)["Montant"].sum()
                fig_total = px.line(total_df, x="Annee", y="Montant",
                    title="Evolution du Total Bilan", labels={"Montant": montant_label})
                fig_total.update_layout(height=400)
                st.plotly_chart(fig_total, use_container_width=True)

            with tab2:
                st.subheader("Comparaison des Comptes")
                comparison_year = st.selectbox("Annee", sorted(df["Annee"].unique(), reverse=True), key="balance_compare_year")
                year_data = df[df["Annee"] == comparison_year].sort_values("Montant", ascending=True)
                fig_bar = px.bar(year_data, x="Montant", y="Mission", orientation="h",
                    title=f"Repartition - {comparison_year}", labels={"Montant": montant_label})
                fig_bar.update_layout(height=600)
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab3:
                st.subheader("Repartition du Bilan")
                pivot_df = df.pivot(index="Annee", columns="Mission", values="Montant").fillna(0)
                fig_stack = go.Figure()
                for mission in pivot_df.columns:
                    fig_stack.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[mission],
                        stackgroup="one", name=mission, mode="lines"))
                fig_stack.update_layout(title="Evolution de la Repartition", height=600,
                    legend=dict(orientation="h", yanchor="top", y=-0.2))
                st.plotly_chart(fig_stack, use_container_width=True)

    # =============================================================================
    # EXPENSES VIEW (Account 6)
    # =============================================================================
    elif selected_view == "expenses":
        df = st.session_state.budget_data
        predictions_df = st.session_state.predictions

        if df is None or df.empty:
            st.warning("Aucune donnee de depenses disponible. Cliquez sur 'Charger les donnees'.")
        else:
            df, montant_label = apply_inflation_adjustment(df)

            if include_debt_interest:
                fetcher = DataFetcher()
                debt_series = fetcher.get_debt_interest_series(int(df["Annee"].min()), int(df["Annee"].max()))
                debt_series["Mission"] = "Charge de la dette"
                df = pd.concat([df, debt_series[["Annee", "Mission", "Montant"]]], ignore_index=True)

            col1, col2, col3, col4 = st.columns(4)
            latest_year = int(df["Annee"].max())
            earliest_year = int(df["Annee"].min())
            latest_total = df[df["Annee"] == latest_year]["Montant"].sum()
            earliest_total = df[df["Annee"] == earliest_year]["Montant"].sum()
            total_growth = ((latest_total - earliest_total) / earliest_total) * 100 if earliest_total else 0
            avg_growth = total_growth / (latest_year - earliest_year) if latest_year > earliest_year else 0

            with col1:
                st.metric(f"Depenses {latest_year}", format_currency(latest_total))
            with col2:
                st.metric("Croissance annuelle", f"{avg_growth:.1f}%")
            with col3:
                top_cat = get_top_categories(df, n=1)
                if not top_cat.empty:
                    st.metric("Plus grosse mission", top_cat.iloc[0]["Mission"][:20])
            with col4:
                if predictions_df is not None and not predictions_df.empty:
                    pred_2030 = predictions_df[predictions_df["Annee"] == 2030]["Montant_Predit"].sum()
                    st.metric("Prediction 2030", format_currency(pred_2030))
                else:
                    st.metric("Prediction 2030", "N/A")

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Evolution", "Comparaison", "Repartition", "Predictions", "Analyse"])

            with tab1:
                st.subheader("Evolution des Depenses par Mission")
                fig = px.line(df, x="Annee", y="Montant", color="Mission",
                    title="Evolution des Depenses Budgetaires", labels={"Montant": montant_label})
                fig.update_layout(height=600, hovermode="x unified")
                fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2))
                add_period_overlays(fig, GOVERNMENT_PERIODS, earliest_year, latest_year)
                add_event_markers(fig, KEY_EVENTS, earliest_year, latest_year)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Taux de Croissance par Mission")
                growth_data = []
                for mission in df["Mission"].unique():
                    m_data = df[df["Mission"] == mission].sort_values("Annee")
                    if len(m_data) > 1:
                        gr = calculate_growth_rate(m_data["Montant"].iloc[0], m_data["Montant"].iloc[-1], len(m_data) - 1)
                        growth_data.append({"Mission": mission, "Croissance (%)": gr})

                if growth_data:
                    growth_df = pd.DataFrame(growth_data).sort_values("Croissance (%)", ascending=False)
                    fig_growth = px.bar(growth_df, x="Mission", y="Croissance (%)",
                        color="Croissance (%)", color_continuous_scale="RdYlBu_r")
                    fig_growth.update_xaxes(tickangle=45)
                    fig_growth.update_layout(height=500)
                    st.plotly_chart(fig_growth, use_container_width=True)

            with tab2:
                st.subheader("Comparaison des Missions")
                comparison_year = st.selectbox("Annee de comparaison", sorted(df["Annee"].unique(), reverse=True), key="expense_compare_year")
                year_data = df[df["Annee"] == comparison_year].sort_values("Montant", ascending=True)
                fig_bar = px.bar(year_data, x="Montant", y="Mission", orientation="h",
                    title=f"Repartition des Depenses - {comparison_year}")
                fig_bar.update_layout(height=600)
                st.plotly_chart(fig_bar, use_container_width=True)

                year_data["Pourcentage"] = (year_data["Montant"] / year_data["Montant"].sum()) * 100
                fig_pie = px.pie(year_data, values="Montant", names="Mission")
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

            with tab3:
                st.subheader("Repartition Budgetaire")
                pivot_df = df.pivot(index="Annee", columns="Mission", values="Montant").fillna(0)
                fig_stack = go.Figure()
                for mission in pivot_df.columns:
                    fig_stack.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[mission],
                        stackgroup="one", name=mission, mode="lines"))
                fig_stack.update_layout(title="Evolution de la Repartition (Aires Empilees)", height=600,
                    legend=dict(orientation="h", yanchor="top", y=-0.2))
                add_period_overlays(fig_stack, GOVERNMENT_PERIODS, earliest_year, latest_year)
                st.plotly_chart(fig_stack, use_container_width=True)

                pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
                fig_pct = go.Figure()
                for mission in pivot_pct.columns:
                    fig_pct.add_trace(go.Scatter(x=pivot_pct.index, y=pivot_pct[mission],
                        stackgroup="one", name=mission))
                fig_pct.update_layout(title="Evolution en Pourcentage", height=600, yaxis_title="Pourcentage (%)",
                    legend=dict(orientation="h", yanchor="top", y=-0.2))
                st.plotly_chart(fig_pct, use_container_width=True)

            with tab4:
                st.subheader("Predictions Budgetaires 2025-2030")
                if predictions_df is not None and not predictions_df.empty:
                    hist_df = df.copy()
                    hist_df["Type"] = "Historique"
                    hist_df["Montant_Predit"] = hist_df["Montant"]
                    pred_df = predictions_df.copy()
                    pred_df["Type"] = "Prediction"
                    pred_df["Montant"] = pred_df["Montant_Predit"]
                    combined = pd.concat([hist_df, pred_df], ignore_index=True)

                    fig_pred = px.line(combined, x="Annee", y="Montant_Predit", color="Mission",
                        line_dash="Type", title="Historique et Predictions")
                    fig_pred.add_vline(x=2024.5, line_dash="dash", line_color="red")
                    fig_pred.update_layout(height=600, legend=dict(orientation="h", y=-0.3))
                    st.plotly_chart(fig_pred, use_container_width=True)

                    st.subheader("Resume des Predictions 2030")
                    pred_2030 = predictions_df[predictions_df["Annee"] == 2030]
                    summary = []
                    for mission in pred_2030["Mission"].unique():
                        pred_val = pred_2030[pred_2030["Mission"] == mission]["Montant_Predit"].iloc[0]
                        hist_2024 = df[(df["Mission"] == mission) & (df["Annee"] == latest_year)]
                        hist_val = hist_2024["Montant"].iloc[0] if not hist_2024.empty else 0
                        growth = ((pred_val - hist_val) / hist_val * 100) if hist_val > 0 else 0
                        summary.append({"Mission": mission, f"Budget {latest_year}": round(hist_val, 2),
                            "Prediction 2030": round(pred_val, 2), "Croissance (%)": round(growth, 1)})
                    st.dataframe(pd.DataFrame(summary).sort_values("Croissance (%)", ascending=False))
                else:
                    st.warning("Selectionnez 2024 comme annee de fin pour generer des predictions.")

            with tab5:
                st.subheader("Analyse Detaillee")
                insights = generate_insights_i18n(df, predictions_df, current_lang)
                for msg in insights:
                    st.markdown(f"- {msg}")

                st.markdown("---")
                st.subheader("Export des Donnees")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Telecharger Historique (CSV)", df.to_csv(index=False),
                        f"depenses_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
                with col2:
                    if predictions_df is not None:
                        st.download_button("Telecharger Predictions (CSV)", predictions_df.to_csv(index=False),
                            f"predictions_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    # =============================================================================
    # REVENUE VIEW (Account 7)
    # =============================================================================
    elif selected_view == "revenue":
        df = st.session_state.revenue_data

        if df is None or df.empty:
            st.warning("Aucune donnee de recettes disponible. Cliquez sur 'Charger les donnees'.")
        else:
            df, montant_label = apply_inflation_adjustment(df)

            col1, col2, col3 = st.columns(3)
            latest_year = int(df["Annee"].max())
            earliest_year = int(df["Annee"].min())
            latest_total = df[df["Annee"] == latest_year]["Montant"].sum()
            earliest_total = df[df["Annee"] == earliest_year]["Montant"].sum()
            total_growth = ((latest_total - earliest_total) / earliest_total) * 100 if earliest_total else 0

            with col1:
                st.metric(f"Recettes {latest_year}", format_currency(latest_total))
            with col2:
                st.metric("Evolution", f"{total_growth:.1f}%", f"depuis {earliest_year}")
            with col3:
                st.metric("Categories", len(df["Mission"].unique()))

            tab1, tab2, tab3 = st.tabs(["Evolution", "Comparaison", "Recettes vs Depenses"])

            with tab1:
                st.subheader("Evolution des Recettes")
                fig = px.line(df, x="Annee", y="Montant", color="Mission",
                    title="Evolution des Recettes de l'Etat", labels={"Montant": montant_label})
                fig.update_layout(height=600, hovermode="x unified")
                fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2))
                add_period_overlays(fig, GOVERNMENT_PERIODS, earliest_year, latest_year)
                add_event_markers(fig, KEY_EVENTS, earliest_year, latest_year)
                st.plotly_chart(fig, use_container_width=True)

                total_df = df.groupby("Annee", as_index=False)["Montant"].sum()
                fig_total = px.line(total_df, x="Annee", y="Montant", title="Recettes Totales")
                fig_total.update_layout(height=400)
                st.plotly_chart(fig_total, use_container_width=True)

            with tab2:
                st.subheader("Comparaison des Categories de Recettes")
                comparison_year = st.selectbox("Annee", sorted(df["Annee"].unique(), reverse=True), key="revenue_compare_year")
                year_data = df[df["Annee"] == comparison_year].sort_values("Montant", ascending=True)
                fig_bar = px.bar(year_data, x="Montant", y="Mission", orientation="h",
                    title=f"Repartition des Recettes - {comparison_year}")
                fig_bar.update_layout(height=600)
                st.plotly_chart(fig_bar, use_container_width=True)

                fig_pie = px.pie(year_data, values="Montant", names="Mission")
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

            with tab3:
                st.subheader("Recettes vs Depenses")
                expense_df = st.session_state.budget_data
                if expense_df is None or expense_df.empty:
                    st.info("Chargez d'abord les donnees de depenses pour voir la comparaison.")
                    if st.button("Charger les depenses"):
                        fetcher = DataFetcher()
                        expense_df = fetcher.fetch_budget_data(start_year=year_range[0], end_year=year_range[1],
                            base_compte="6", acc_level_range=1)
                        st.session_state.budget_data = expense_df
                        st.rerun()
                else:
                    revenue_agg = df.groupby("Annee", as_index=False)["Montant"].sum()
                    revenue_agg = revenue_agg.rename(columns={"Montant": "Recettes"})
                    expense_agg = expense_df.groupby("Annee", as_index=False)["Montant"].sum()
                    expense_agg = expense_agg.rename(columns={"Montant": "Depenses"})
                    combined = pd.merge(revenue_agg, expense_agg, on="Annee", how="inner")
                    combined["Solde"] = combined["Recettes"] - combined["Depenses"]

                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=combined["Annee"], y=combined["Recettes"],
                        name="Recettes", mode="lines+markers"))
                    fig_compare.add_trace(go.Scatter(x=combined["Annee"], y=combined["Depenses"],
                        name="Depenses", mode="lines+markers"))
                    fig_compare.update_layout(title="Recettes vs Depenses", height=500, yaxis_title=montant_label)
                    st.plotly_chart(fig_compare, use_container_width=True)

                    fig_solde = px.bar(combined, x="Annee", y="Solde",
                        title="Solde Budgetaire (Recettes - Depenses)", color="Solde", color_continuous_scale="RdYlGn")
                    fig_solde.update_layout(height=400)
                    st.plotly_chart(fig_solde, use_container_width=True)

                    latest_solde = combined[combined["Annee"] == combined["Annee"].max()]["Solde"].iloc[0]
                    avg_solde = combined["Solde"].mean()
                    col1, col2 = st.columns(2)
                    with col1:
                        status = "Excedent" if latest_solde > 0 else "Deficit"
                        st.metric(f"Solde {int(combined['Annee'].max())}", format_currency(abs(latest_solde)), status)
                    with col2:
                        st.metric("Solde moyen", format_currency(abs(avg_solde)))


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Budget Horizon | Donnees: data.economie.gouv.fr</p>
    <p>Derniere mise a jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
