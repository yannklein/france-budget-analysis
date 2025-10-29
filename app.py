import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import sys
import os
import json

# Import custom modules
from data_fetcher import DataFetcher
from predictor import BudgetPredictor
from utils import (
    format_currency,
    calculate_growth_rate,
    get_top_categories,
    adjust_to_constant_euros,
    get_eu_languages,
    translate,
    generate_insights_i18n,
)

with open("account_name.json") as f:
    COMPTES = json.load(f)

# Configure page
st.set_page_config(
    page_title="Budget Horizon",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description are set after language selection below

# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "budget_data" not in st.session_state:
    st.session_state.budget_data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# Sidebar configuration
lang_options = get_eu_languages()
lang_codes = [opt["code"] for opt in lang_options]
lang_names = {opt["code"]: opt["name"] for opt in lang_options}
current_lang = st.sidebar.selectbox(
    translate("sidebar.language", "fr", "Langue"),
    options=lang_codes,
    index=lang_codes.index("fr") if "fr" in lang_codes else 0,
    format_func=lambda c: lang_names.get(c, c),
)
st.sidebar.header(translate("sidebar.config", current_lang, "Configuration"))

# Update title/description based on selected language
st.title(
    "🏛️ "
    + translate(
        "title.app", current_lang, "Budget de l'État Français - Analyse et Prédictions"
    )
)
st.markdown(
    translate(
        "desc.app",
        current_lang,
        "Cette application analyse l'évolution des dépenses budgétaires de l'État français sur 20 ans et propose des prédictions basées sur l'intelligence artificielle.",
    )
)
# Inflation adjustment option
adjust_inflation = st.sidebar.checkbox(
    translate(
        "toggle.inflation", current_lang, "Ajuster pour l'inflation (euros constants)"
    ),
    value=False,
    help=translate(
        "toggle.inflation",
        current_lang,
        "Affiche les montants en euros constants en utilisant un indice des prix (CPI).",
    ),
)
show_governments = st.sidebar.checkbox(
    translate("toggle.gov_periods", current_lang, "Afficher périodes gouvernementales"),
    value=False,
    help=translate(
        "toggle.gov_periods",
        current_lang,
        "Superpose les périodes des gouvernements (Premiers ministres) sur les graphiques temporels.",
    ),
)
show_key_events = st.sidebar.checkbox(
    translate("toggle.events", current_lang, "Afficher événements majeurs"),
    value=True,
    help=translate(
        "toggle.events",
        current_lang,
        "Affiche des marqueurs pour des événements macro-économiques ou politiques impactant le budget.",
    ),
)
include_debt_interest = st.sidebar.checkbox(
    translate("toggle.debt", current_lang, "Inclure charge de la dette (intérêts)"),
    value=True,
    help=translate(
        "toggle.debt",
        current_lang,
        "Ajoute la mission 'Charge de la dette de l'État' aux montants.",
    ),
)

# Compte selectimon
compte_lvl_1 = st.sidebar.selectbox(
    translate("sidebar.compte_lvl_1", current_lang, "Compte de base"),
    list(COMPTES.keys()),
    format_func=lambda key: f"{key} - {COMPTES[key]}",
    help=translate(
        "sidebar.compte_lvl_1",
        current_lang,
        "Choisir le compte de base pour l'analyse budgétaire.",
    ),
)

compte_lvl_2 = st.sidebar.selectbox(
    translate("sidebar.compte_lvl_2", current_lang, "Compte de base niveau 2"),
    list(COMPTES.keys() if compte_lvl_1 == "" else [compte for compte in COMPTES.keys() if compte.startswith(compte_lvl_1)] ),
    format_func=lambda key: f"{key} - {COMPTES[key]}",
    help=translate(
        "sidebar.compte_lvl_2",
        current_lang,
        "Choisir le compte de base pour l'analyse budgétaire.",
    ),
)

# Year range selection
year_range = st.sidebar.slider(
    translate("sidebar.year_range", current_lang, "Plage d'années"),
    min_value=2015,
    max_value=2024,
    value=(2015, 2024),
    help=translate(
        "sidebar.year_range", current_lang, "Sélectionner la période d'analyse"
    ),
)

# account level
acc_level_range = st.sidebar.slider(
    translate("sidebar.acc_level_range", current_lang, "Detail des comptes"),
    min_value=1,
    max_value=3,
    value=(1),
    help=translate(
        "sidebar.acc_level_range", current_lang, "Sélectionner le détail des comptes"
    ),
)


# Load data button
if st.sidebar.button(
    "🔄 " + translate("sidebar.load_data", current_lang, "Charger les données"),
    type="primary",
):
    with st.spinner("Chargement des données budgétaires..."):
        try:
            fetcher = DataFetcher()
            st.session_state.budget_data = fetcher.fetch_budget_data(
                start_year=year_range[0], end_year=year_range[1], acc_level_range=acc_level_range, base_compte=(compte_lvl_2 if compte_lvl_2 else compte_lvl_1)
            )
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Données chargées avec succès!")

            if year_range[1] < 2024:
                st.sidebar.warning(
                    "⚠️ Selectionnez 2024 comme année de fin pour afficher des prédictions."
                )
            else:
                # Generate predictions
                with st.spinner("Génération des prédictions..."):
                    predictor = BudgetPredictor()
                    st.session_state.predictions = predictor.predict_future_spending(
                        st.session_state.budget_data
                    )
                    st.sidebar.success("✅ Prédictions générées!")

        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors du chargement: {str(e)}")
            st.session_state.data_loaded = False

# Main content
if not st.session_state.data_loaded:
    st.info("👈 Utilisez la barre latérale pour charger les données budgétaires.")

    # Show sample structure while waiting for data
    st.subheader("📊 Structure des données attendues")
    sample_data = pd.DataFrame(
        {
            "Année": [2020, 2021, 2022, 2023, 2024],
            "Mission": ["Défense", "Éducation", "Santé", "Infrastructure", "Recherche"],
            "Montant (Milliards €)": [47.2, 53.8, 89.4, 32.1, 28.5],
            "Pourcentage du PIB": [1.8, 2.1, 3.4, 1.2, 1.1],
        }
    )
    st.dataframe(sample_data, use_container_width=True)

else:
    df = st.session_state.budget_data
    predictions_df = st.session_state.predictions
    montant_label = "Montant (Milliards €)"
    # Define government (Prime Minister) periods (years inclusive, coarse)
    government_periods = [
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

    # Key events (single-year markers)
    key_events = [
        {"year": 2008, "label": "Crise financière mondiale"},
        {"year": 2009, "label": "Récession et relance"},
        {"year": 2011, "label": "Crise dette zone euro"},
        {"year": 2015, "label": "Sécurité intérieure renforcée"},
        {"year": 2020, "label": "COVID-19: plans de soutien"},
        {"year": 2022, "label": "Crise énergie/inflation"},
    ]

    def add_period_overlays(fig, periods, min_year, max_year):
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

    def add_event_markers(fig, events, min_year, max_year):
        if not show_key_events:
            return
        for e in events:
            x = int(e["year"])
            if x < min_year or x > max_year:
                continue
            fig.add_vline(
                x=x,
                line_dash="dot",
                line_color="#7f8c8d",
            )
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

    # Apply inflation adjustment if requested
    if adjust_inflation and df is not None and not df.empty:
        base_year_choice = st.sidebar.number_input(
            "Année de base (euros constants)",
            min_value=int(df["Année"].min()),
            max_value=int(
                max(
                    df["Année"].max(),
                    (
                        2030
                        if predictions_df is not None and not predictions_df.empty
                        else df["Année"].max()
                    ),
                )
            ),
            value=int(df["Année"].max()),
            step=1,
        )
        fetcher = DataFetcher()
        cpi_df = fetcher.get_cpi_series(
            int(df["Année"].min()),
            int(
                max(
                    df["Année"].max(),
                    (
                        2030
                        if predictions_df is not None and not predictions_df.empty
                        else df["Année"].max()
                    ),
                )
            ),
        )
        df = adjust_to_constant_euros(
            df, cpi_df, base_year=base_year_choice, amount_col="Montant"
        )
        if predictions_df is not None and not predictions_df.empty:
            predictions_df = adjust_to_constant_euros(
                predictions_df,
                cpi_df,
                base_year=base_year_choice,
                amount_col="Montant_Prédit",
            )
        montant_label = f"Montant (Milliards €, euros constants {base_year_choice})"

    # Optionally include debt interest as a mission
    if include_debt_interest and df is not None and not df.empty:
        fetcher = DataFetcher()
        debt_series = fetcher.get_debt_interest_series(
            int(df["Année"].min()), int(df["Année"].max())
        )
        if adjust_inflation and "cpi_df" in locals():
            debt_series = adjust_to_constant_euros(
                debt_series, cpi_df, base_year=base_year_choice, amount_col="Montant"
            )
        debt_series["Mission"] = "Charge de la dette de l'État"
        df = pd.concat(
            [df, debt_series[["Année", "Mission", "Montant"]]], ignore_index=True
        )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    latest_year = df["Année"].max()
    latest_total = df[df["Année"] == latest_year]["Montant"].sum()
    earliest_year = df["Année"].min()
    earliest_total = df[df["Année"] == earliest_year]["Montant"].sum()

    total_growth = ((latest_total - earliest_total) / earliest_total) * 100
    avg_annual_growth = total_growth / (latest_year - earliest_year)

    with col1:
        st.metric(
            f"Budget Total {latest_year}",
            format_currency(latest_total),
            f"{total_growth:.1f}% depuis {earliest_year}",
        )

    with col2:
        st.metric(
            "Croissance Annuelle Moyenne", f"{avg_annual_growth:.1f}%", "Sur la période"
        )

    with col3:
        top_categories = get_top_categories(df, n=1)
        if not top_categories.empty:
            top_category = top_categories.iloc[0]
            st.metric(
                "Mission la Plus Importante",
                top_category["Mission"],
                format_currency(top_category["Montant"]),
            )
        else:
            st.metric("Mission la Plus Importante", "N/A", "N/A")

    with col4:
        if predictions_df is not None and not predictions_df.empty:
            future_total = predictions_df[predictions_df["Année"] == 2030][
                "Montant_Prédit"
            ].sum()
            st.metric(
                "Prédiction 2030",
                format_currency(future_total),
                f"{((future_total - latest_total) / latest_total * 100):.1f}% vs 2024",
            )
        else:
            st.metric("Prédiction 2030", "N/A", "Calcul en cours")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 " + translate("tab.evolution", current_lang, "Évolution Temporelle"),
            "🏛️ " + translate("tab.compare", current_lang, "Comparaison Missions"),
            "📊 " + translate("tab.split", current_lang, "Répartition Budgétaire"),
            "🔮 " + translate("tab.pred", current_lang, "Prédictions"),
            "📋 " + translate("tab.details", current_lang, "Analyse Détaillée"),
            "💰 " + translate("tab.revenue", current_lang, "Recettes de l'État"),
        ]
    )

    with tab1:
        st.subheader(
            translate(
                "header.evolution_missions",
                current_lang,
                "Évolution des Dépenses par Mission (2005-2024)",
            )
        )
        st.markdown(
            translate(
                "analysis.evolution",
                current_lang,
                "Cette vue met en évidence l'évolution de chaque mission dans le temps.",
            )
        )

        # Interactive line chart
        fig_evolution = px.line(
            df,
            x="Année",
            y="Montant",
            color="Mission",
            title=translate(
            "title.mission_evolution",
            current_lang,
            "Évolution des Dépenses Budgétaires par Mission",
            ),
            labels={"Montant": montant_label, "Année": "Année", "Mission": "Mission"},
        )
        fig_evolution.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        fig_evolution.update_layout(height=600, hovermode="x unified")
        add_period_overlays(
            fig_evolution,
            government_periods,
            int(df["Année"].min()),
            int(df["Année"].max()),
        )
        add_event_markers(
            fig_evolution, key_events, int(df["Année"].min()), int(df["Année"].max())
        )
        st.plotly_chart(fig_evolution, use_container_width=True)

        # Growth rate analysis
        st.subheader(
            translate(
                "header.growth_rate",
                current_lang,
                "Taux de Croissance Annuel par Mission",
            )
        )
        growth_data = []
        for mission in df["Mission"].unique():
            mission_data = df[df["Mission"] == mission].sort_values("Année")
            if len(mission_data) > 1:
                growth_rate = calculate_growth_rate(
                    mission_data["Montant"].iloc[0],
                    mission_data["Montant"].iloc[-1],
                    len(mission_data) - 1,
                )
                growth_data.append(
                    {
                        "Mission": mission,
                        "Croissance Annuelle (%)": growth_rate,
                        "Variation Totale (%)": (
                            (
                                mission_data["Montant"].iloc[-1]
                                - mission_data["Montant"].iloc[0]
                            )
                            / mission_data["Montant"].iloc[0]
                        )
                        * 100,
                    }
                )

        growth_df = pd.DataFrame(growth_data)
        if not growth_df.empty:
            growth_df["Mission"] = growth_df["Mission"].str.split("(").str[0].str.strip()

            growth_df = growth_df.sort_values(
                "Croissance Annuelle (%)", ascending=False
            )

            fig_growth = px.bar(
                growth_df,
                x="Mission",
                y="Croissance Annuelle (%)",
                title=translate(
                    "title.growth_rate",
                    current_lang,
                    "Taux de Croissance Annuel Moyen par Mission",
                ),
                color="Croissance Annuelle (%)",
                color_continuous_scale="RdYlBu_r",
            )
            fig_growth.update_xaxes(tickangle=45)
            fig_growth.update_layout(height=500)
            st.plotly_chart(fig_growth, use_container_width=True)

        # Total budget evolution (all missions)
        st.subheader(
            translate(
                "title.total_budget",
                current_lang,
                "Évolution du Budget Total de l'État",
            )
        )
        total_df = df.groupby("Année", as_index=False)["Montant"].sum()
        fig_total = px.line(
            total_df,
            x="Année",
            y="Montant",
            title="Évolution du Budget Total",
            labels={"Montant": montant_label, "Année": "Année"},
        )
        fig_total.update_layout(height=400, hovermode="x unified")
        add_period_overlays(
            fig_total,
            government_periods,
            int(df["Année"].min()),
            int(df["Année"].max()),
        )
        add_event_markers(
            fig_total, key_events, int(df["Année"].min()), int(df["Année"].max())
        )
        st.plotly_chart(fig_total, use_container_width=True)

    with tab2:
        st.subheader(
            translate(
                "header.compare", current_lang, "Comparaison des Missions Budgétaires"
            )
        )
        st.markdown(
            translate(
                "analysis.compare",
                current_lang,
                "Comparez les missions pour une année donnée afin d'identifier les postes les plus importants et leurs poids relatifs.",
            )
        )

        # Select year for comparison
        comparison_year = st.selectbox(
            "Année de comparaison", sorted(df["Année"].unique(), reverse=True)
        )

        year_data = df[df["Année"] == comparison_year].copy()
        year_data = year_data.sort_values("Montant", ascending=True)

        # Horizontal bar chart
        fig_comparison = px.bar(
            year_data,
            x="Montant",
            y="Mission",
            orientation="h",
            title=translate(
                "title.comparison", current_lang, "Répartition des Dépenses par Mission"
            )
            + f" - {comparison_year}",
            labels={"Montant": montant_label, "Mission": "Mission"},
            text="Montant",
        )
        fig_comparison.update_traces(
            texttemplate="%{text:.1f}B€", textposition="outside"
        )
        fig_comparison.update_layout(height=600)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Percentage breakdown
        year_data["Pourcentage"] = (
            year_data["Montant"] / year_data["Montant"].sum()
        ) * 100
        st.subheader(f"Répartition en Pourcentage - {comparison_year}")

        year_data["Mission"] = year_data["Mission"].str.split("(").str[0].str.strip()
        
        # Pie chart
        fig_pie = px.pie(
            year_data,
            values="Montant",
            names="Mission",
            title="Répartition du Budget par Mission",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            height=500,
            legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,  # Adjusted to move the legend further down
            xanchor="center",
            x=0.5
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Dataframe below the pie chart
        st.dataframe(
            year_data[["Mission", "Montant", "Pourcentage"]].round(2),
            use_container_width=True,
        )

    with tab3:
        st.subheader(
            translate(
                "header.split", current_lang, "Analyse de la Répartition Budgétaire"
            )
        )
        st.markdown(
            translate(
                "analysis.split",
                current_lang,
                "Les aires empilées montrent l'évolution de la composition du budget au fil des années.",
            )
        )

        # Stacked area chart
        pivot_df = df.pivot(index="Année", columns="Mission", values="Montant").fillna(
            0
        )

        fig_stacked = go.Figure()
        for mission in pivot_df.columns:
            fig_stacked.add_trace(
                go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[mission],
                    stackgroup="one",
                    name=mission,
                    mode="lines",
                    line=dict(width=0.5),
                    fillcolor=px.colors.qualitative.Set3[
                        hash(mission) % len(px.colors.qualitative.Set3)
                    ],
                )
            )

        fig_stacked.update_layout(
            title=translate(
                "title.stacked_area",
                current_lang,
                "Évolution de la Répartition Budgétaire (Aires Empilées)",
            ),
            xaxis_title="Année",
            yaxis_title=montant_label,
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,  # Position the legend below the graph
                xanchor="center",
                x=0.5,
            ),
        )
        add_period_overlays(
            fig_stacked,
            government_periods,
            int(df["Année"].min()),
            int(df["Année"].max()),
        )
        add_event_markers(
            fig_stacked, key_events, int(df["Année"].min()), int(df["Année"].max())
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Percentage evolution
        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

        fig_pct = go.Figure()
        for mission in pivot_pct.columns:
            fig_pct.add_trace(
                go.Scatter(
                    x=pivot_pct.index,
                    y=pivot_pct[mission],
                    stackgroup="one",
                    name=mission,
                    mode="lines",
                    line=dict(width=0.5),
                )
            )

        fig_pct.update_layout(
            title=translate(
                "title.percentage_evolution",
                current_lang,
                "Évolution de la Répartition Budgétaire (Pourcentages)",
            ),
            xaxis_title="Année",
            yaxis_title="Pourcentage (%)",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,  # Position the legend below the graph
                xanchor="center",
                x=0.5,
            ),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    with tab4:
        st.subheader(
            "🔮 "
            + translate(
                "header.pred", current_lang, "Prédictions Budgétaires 2026-2030"
            )
        )
        st.markdown(
            translate(
                "analysis.pred",
                current_lang,
                "Les prédictions prolongent les tendances récentes avec des contraintes.",
            )
        )

        if predictions_df is not None and not predictions_df.empty:
            # Combine historical and predicted data
            historical_df = df.copy()
            historical_df["Type"] = "Historique"
            historical_df["Montant_Prédit"] = historical_df["Montant"]

            pred_df = predictions_df.copy()
            pred_df["Type"] = "Prédit"
            pred_df["Montant"] = pred_df["Montant_Prédit"]

            combined_df = pd.concat([historical_df, pred_df], ignore_index=True)

            # Interactive prediction chart
            fig_pred = px.line(
                combined_df,
                x="Année",
                y="Montant_Prédit",
                color="Mission",
                line_dash="Type",
                title=translate(
                    "title.history_predictions",
                    current_lang,
                    "Évolution Historique et Prédictions Budgétaires",
                ),
                labels={
                    "Montant_Prédit": montant_label,
                    "Année": "Année",
                    "Mission": "Mission",
                    "Type": "Type de Données",
                },
            )

            # Add vertical line to separate historical and predicted data
            fig_pred.add_vline(
                x=2024.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Limite Historique/Prédiction",
            )

            fig_pred.update_layout(
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.4,  # Position the legend below the graph
                    xanchor="center",
                    x=0.5,
                ),
            )
            add_period_overlays(
                fig_pred,
                government_periods,
                int(combined_df["Année"].min()),
                int(combined_df["Année"].max()),
            )
            add_event_markers(
                fig_pred,
                key_events,
                int(combined_df["Année"].min()),
                int(combined_df["Année"].max()),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Prediction summary
            st.subheader(
                translate(
                    "header.pred_summary", current_lang, "Résumé des Prédictions 2030"
                )
            )

            pred_2030 = predictions_df[predictions_df["Année"] == 2030]
            hist_2024 = df[df["Année"] == 2024]

            summary_data = []
            for mission in pred_2030["Mission"].unique():
                pred_value = pred_2030[pred_2030["Mission"] == mission][
                    "Montant_Prédit"
                ].iloc[0]
                hist_value = (
                    hist_2024[hist_2024["Mission"] == mission]["Montant"].iloc[0]
                    if mission in hist_2024["Mission"].values
                    else 0
                )

                if hist_value > 0:
                    growth = ((pred_value - hist_value) / hist_value) * 100
                else:
                    growth = 0

                summary_data.append(
                    {
                        "Mission": mission,
                        "Budget 2024 (Milliards €)": round(hist_value, 2),
                        "Prédiction 2030 (Milliards €)": round(pred_value, 2),
                        "Croissance Prédite (%)": round(growth, 1),
                        "Variation (Milliards €)": round(pred_value - hist_value, 2),
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(
                "Croissance Prédite (%)", ascending=False
            )

            st.dataframe(summary_df, use_container_width=True)

        else:
            st.warning("⚠️ Prédictions en cours de génération. Veuillez patienter...")

    with tab5:
        st.subheader(
            "📋 "
            + translate(
                "header.details_export", current_lang, "Analyse Détaillée et Export"
            )
        )
        st.markdown(
            translate(
                "analysis.details",
                current_lang,
                "Utilisez le résumé et les tableaux de croissance pour approfondir les déterminants.",
            )
        )
        # Localized insights
        insights_msgs = generate_insights_i18n(df, predictions_df, current_lang)
        for msg in insights_msgs:
            st.markdown("- " + msg)
        # Data sources section
        st.markdown("---")
        st.subheader(
            translate(
                "sources.title", current_lang, "📚 Sources de Données et Références"
            )
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                translate(
                    "sources.budget",
                    current_lang,
                    "**Données Budgétaires** : Missions du budget de l'État français depuis data.gouv.fr",
                )
            )
            st.markdown(
                translate(
                    "sources.cpi",
                    current_lang,
                    "**Données d'Inflation** : Indice des Prix à la Consommation (IPC) de l'INSEE",
                )
            )
        with col2:
            st.markdown(
                translate(
                    "sources.debt",
                    current_lang,
                    "**Intérêts de la Dette** : Paiements d'intérêts de la dette publique modélisés",
                )
            )
            st.markdown(
                translate(
                    "sources.methodology",
                    current_lang,
                    "**Méthodologie** : Traitement des données, ajustement inflationniste et prédictions",
                )
            )

        st.markdown(
            translate(
                "sources.disclaimer",
                current_lang,
                "**Avertissement** : Cette analyse est à des fins d'information. Consultez les sources gouvernementales officielles pour les chiffres budgétaires officiels.",
            )
        )

        # Links to official sources
        st.markdown("**🔗 Official Sources:**")
        st.markdown(
            "- [data.gouv.fr - Budget de l'État](https://www.data.gouv.fr/fr/datasets/budget-de-letat/)"
        )
        st.markdown(
            "- [INSEE - Indices des Prix](https://www.insee.fr/fr/statistiques/serie/000436391)"
        )
        st.markdown(
            "- [Ministère de l'Économie - Finances Publiques](https://www.economie.gouv.fr/finances-publiques)"
        )
        st.markdown(
            "- [Banque de France - Dette Publique](https://www.banque-france.fr/statistiques/dette-publique)"
        )

        # Top growing categories
        st.subheader(
            "🚀 "
            + translate(
                "header.top_growth", current_lang, "Missions en Forte Croissance"
            )
        )

        growth_analysis = []
        for mission in df["Mission"].unique():
            mission_data = df[df["Mission"] == mission].sort_values("Année")
            if len(mission_data) >= 2:
                start_value = mission_data["Montant"].iloc[0]
                end_value = mission_data["Montant"].iloc[-1]
                total_growth = ((end_value - start_value) / start_value) * 100

                growth_analysis.append(
                    {
                        "Mission": mission,
                        "Montant Initial (Milliards €)": round(start_value, 2),
                        "Montant Final (Milliards €)": round(end_value, 2),
                        "Croissance Totale (%)": round(total_growth, 1),
                        "Tendance": "📈" if total_growth > 0 else "📉",
                    }
                )

        growth_analysis_df = pd.DataFrame(growth_analysis)
        growth_analysis_df = growth_analysis_df.sort_values(
            "Croissance Totale (%)", ascending=False
        )

        st.dataframe(growth_analysis_df, use_container_width=True)

        # Statistical summary
        st.subheader(
            "📊 "
            + translate("header.stats_summary", current_lang, "Résumé Statistique")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Nombre de Missions Analysées", len(df["Mission"].unique()))
            st.metric("Période d'Analyse", f"{df['Année'].min()} - {df['Année'].max()}")
            st.metric("Points de Données", len(df))

        with col2:
            if not growth_analysis_df.empty:
                best_performer = growth_analysis_df.iloc[0]
                st.metric(
                    "Meilleure Performance",
                    best_performer["Mission"],
                    f"+{best_performer['Croissance Totale (%)']}%",
                )

                worst_performer = growth_analysis_df.iloc[-1]
                st.metric(
                    "Performance la Plus Faible",
                    worst_performer["Mission"],
                    f"{worst_performer['Croissance Totale (%)']}%",
                )

        # Data export
        st.subheader(
            "💾 " + translate("export.header", current_lang, "Export des Données")
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export historical data
            csv_historical = df.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="📥 "
                + translate(
                    "button.download_historical",
                    current_lang,
                    "Télécharger Données Historiques (CSV)",
                ),
                data=csv_historical,
                file_name=f"budget_france_historique_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        with col2:
            # Export predictions
            if predictions_df is not None and not predictions_df.empty:
                csv_predictions = predictions_df.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    label="📥 "
                    + translate(
                        "button.download_predictions",
                        current_lang,
                        "Télécharger Prédictions (CSV)",
                    ),
                    data=csv_predictions,
                    file_name=f"budget_france_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

        with col3:
            # Export analysis
            if not growth_analysis_df.empty:
                csv_analysis = growth_analysis_df.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    label="📥 "
                    + translate(
                        "button.download_analysis",
                        current_lang,
                        "Télécharger Analyse (CSV)",
                    ),
                    data=csv_analysis,
                    file_name=f"budget_france_analyse_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    # Add a new page for revenue evolution
    with tab6:
        st.title(
            "📈 "
            + translate(
                "title.revenue_evolution",
                current_lang,
                "Évolution des Recettes de l'État Français",
            )
        )
        st.markdown(
            translate(
                "desc.revenue_evolution",
                current_lang,
                "Cette page présente l'évolution des recettes de l'État français au cours des 20 dernières années.",
            )
        )

        # Fetch revenue data
        with st.spinner(
            translate(
                "loading.revenue_data",
                current_lang,
                "Chargement des données de recettes...",
            )
        ):
            try:
                fetcher = DataFetcher()
                revenue_data = fetcher.fetch_revenue_20y(start_year=year_range[0], end_year=year_range[1])
            except Exception as e:
                st.error(
                    f"❌ "
                    + translate(
                        "error.revenue_data",
                        current_lang,
                        "Erreur lors du chargement des données de recettes",
                    )
                    + f": {str(e)}"
                )
                revenue_data = None

        if revenue_data is not None and not revenue_data.empty:
            # Apply inflation adjustment if requested
            if adjust_inflation:
                revenue_data = adjust_to_constant_euros(
                    revenue_data, cpi_df, base_year=base_year_choice, amount_col="Montant"
                )

            # Line chart for revenue evolution
            st.subheader(
                translate(
                    "header.revenue_evolution",
                    current_lang,
                    "Évolution des Recettes Totales",
                )
            )
            fig_revenue = px.line(
                revenue_data,
                x="Année",
                y="Montant",
                color="Postes",
                title=translate(
                    "title.revenue_evolution_chart",
                    current_lang,
                    "Évolution des Recettes de l'État",
                ),
                labels={"Montant": montant_label, "Année": "Année", "Postes": "Catégorie de recettes"},
            )
            fig_revenue.update_layout(
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.3,  # Position the legend below the graph
                    xanchor="center",
                    x=0.5,
                ),
                legend_title="Catégories de recettes",
            )
            add_period_overlays(
                fig_revenue,
                government_periods,
                int(revenue_data["Année"].min()),
                int(revenue_data["Année"].max()),
            )
            add_event_markers(
                fig_revenue,
                key_events,
                int(revenue_data["Année"].min()),
                int(revenue_data["Année"].max()),
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

            # Key metrics by revenue category (Postes)
            latest_year = revenue_data["Année"].max()
            earliest_year = revenue_data["Année"].min()

            latest_revenue_by_poste = (
                revenue_data[revenue_data["Année"] == latest_year]
                .groupby("Postes")["Montant"]
                .sum()
            )
            earliest_revenue_by_poste = (
                revenue_data[revenue_data["Année"] == earliest_year]
                .groupby("Postes")["Montant"]
                .sum()
            )

            # Combine for growth calculations
            metrics_df = pd.DataFrame({
                "Latest": latest_revenue_by_poste,
                "Earliest": earliest_revenue_by_poste
            }).fillna(0)

            metrics_df["TotalGrowthPct"] = (
                (metrics_df["Latest"] - metrics_df["Earliest"]) / metrics_df["Earliest"].replace(0, np.nan)
            ) * 100
            metrics_df["AvgAnnualGrowthPct"] = metrics_df["TotalGrowthPct"] / (latest_year - earliest_year)

            # Display metrics in 3 rows
            st.subheader(f"Évolution par catégorie ({earliest_year} → {latest_year})")
            rows = [metrics_df.iloc[i:i+4] for i in range(0, len(metrics_df), 4)]

            for row_group in rows:
                cols = st.columns(len(row_group))
                for (poste, row), col in zip(row_group.iterrows(), cols):
                    with col:
                        st.metric(
                            f"{poste} {latest_year}",
                            format_currency(row['Latest']),
                            f"{row['TotalGrowthPct']:.1f}% depuis {earliest_year} | {row['AvgAnnualGrowthPct']:.1f}%/an"
                        )

            # Add a total evolution metric
            total_latest = metrics_df['Latest'].sum()
            total_earliest = metrics_df['Earliest'].sum()
            total_growth_pct = ((total_latest - total_earliest) / total_earliest) * 100
            avg_annual_growth_pct = total_growth_pct / (latest_year - earliest_year)

            st.subheader("Évolution Totale")
            st.metric(
                f"Total {latest_year}",
                format_currency(total_latest),
                f"{total_growth_pct:.1f}% depuis {earliest_year} | {avg_annual_growth_pct:.1f}%/an"
            )

            # Calculate recettes - dépenses
            if st.session_state.data_loaded:
                combined_data = revenue_data.groupby("Année", as_index=False)["Montant"].sum()
                combined_data = combined_data.rename(columns={"Montant": "Recettes"})
                expense_data = df.groupby("Année", as_index=False)["Montant"].sum()
                # print(expense_data)
                expense_data = expense_data.rename(columns={"Montant": "Dépenses"})
                combined_data = pd.merge(combined_data, expense_data, on="Année", how="inner")
                combined_data["Recettes - Dépenses"] = combined_data["Recettes"] - combined_data["Dépenses"]

                # Plot total recettes over time
                st.subheader(
                    translate(
                        "header.total_revenue",
                        current_lang,
                        "Évolution des Recettes Totales",
                    )
                )
                fig_total_revenue = px.line(
                    combined_data,
                    x="Année",
                    y="Recettes",
                    title=translate(
                        "title.total_revenue_chart",
                        current_lang,
                        "Évolution des Recettes Totales",
                    ),
                    labels={"Recettes": "Montant (Milliards €)", "Année": "Année"},
                )
                fig_total_revenue.update_layout(height=600, hovermode="x unified")
                add_period_overlays(
                    fig_total_revenue,
                    government_periods,
                    int(combined_data["Année"].min()),
                    int(combined_data["Année"].max()),
                )
                add_event_markers(
                    fig_total_revenue,
                    key_events,
                    int(combined_data["Année"].min()),
                    int(combined_data["Année"].max()),
                )
                st.plotly_chart(fig_total_revenue, use_container_width=True)

                # Plot recettes - dépenses evolution
                st.subheader(
                    translate(
                        "header.revenue_expense_diff",
                        current_lang,
                        "Évolution de Recettes - Dépenses",
                    )
                )
                fig_diff = px.line(
                    combined_data,
                    x="Année",
                    y="Recettes - Dépenses",
                    title=translate(
                        "title.revenue_expense_diff_chart",
                        current_lang,
                        "Évolution de Recettes - Dépenses",
                    ),
                    labels={"Recettes - Dépenses": "Montant (Milliards €)", "Année": "Année"},
                )
                fig_diff.update_layout(height=600, hovermode="x unified")
                add_period_overlays(
                    fig_diff,
                    government_periods,
                    int(combined_data["Année"].min()),
                    int(combined_data["Année"].max()),
                )
                add_event_markers(
                    fig_diff,
                    key_events,
                    int(combined_data["Année"].min()),
                    int(combined_data["Année"].max()),
                )
                st.plotly_chart(fig_diff, use_container_width=True)
            else:
                st.warning(
                    "⚠️ "
                    + translate(
                        "warning.no_expense_data",
                        current_lang,
                        "Aucune donnée de dépenses disponible pour calculer Recettes - Dépenses.",
                    )
                )
        else:
            st.warning(
                "⚠️ "
                + translate(
                    "warning.no_revenue_data",
                    current_lang,
                    "Aucune donnée de recettes disponible.",
                )
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🏛️ Données officielles du gouvernement français | Sources: data.gouv.fr, INSEE, budget.gouv.fr</p>
    <p>Dernière mise à jour: """
    + datetime.now().strftime("%d/%m/%Y %H:%M")
    + """</p>
</div>
""",
    unsafe_allow_html=True,
)
