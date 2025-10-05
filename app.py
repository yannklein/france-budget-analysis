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

# Import custom modules
from data_fetcher import DataFetcher
from predictor import BudgetPredictor
from utils import format_currency, calculate_growth_rate, get_top_categories

# Configure page
st.set_page_config(
    page_title="Budget de l'État Français - Analyse et Prédictions",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏛️ Évolution du Budget de l'État Français (2005-2025)")
st.markdown("""
Cette application analyse l'évolution des dépenses budgétaires de l'État français sur 20 ans 
et propose des prédictions basées sur l'intelligence artificielle.
""")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'budget_data' not in st.session_state:
    st.session_state.budget_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Source de données",
    ["data.gouv.fr", "INSEE", "data.economie.gouv.fr"],
    help="Choisir la source de données gouvernementales"
)

# Year range selection
year_range = st.sidebar.slider(
    "Plage d'années",
    min_value=2005,
    max_value=2025,
    value=(2005, 2025),
    help="Sélectionner la période d'analyse"
)

# Load data button
if st.sidebar.button("🔄 Charger les données", type="primary"):
    with st.spinner("Chargement des données budgétaires..."):
        try:
            fetcher = DataFetcher()
            st.session_state.budget_data = fetcher.fetch_budget_data(
                source=data_source,
                start_year=year_range[0],
                end_year=year_range[1]
            )
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Données chargées avec succès!")
            
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
    sample_data = pd.DataFrame({
        'Année': [2020, 2021, 2022, 2023, 2024],
        'Mission': ['Défense', 'Éducation', 'Santé', 'Infrastructure', 'Recherche'],
        'Montant (Milliards €)': [47.2, 53.8, 89.4, 32.1, 28.5],
        'Pourcentage du PIB': [1.8, 2.1, 3.4, 1.2, 1.1]
    })
    st.dataframe(sample_data, use_container_width=True)
    
else:
    df = st.session_state.budget_data
    predictions_df = st.session_state.predictions
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_year = df['Année'].max()
    latest_total = df[df['Année'] == latest_year]['Montant'].sum()
    earliest_year = df['Année'].min()
    earliest_total = df[df['Année'] == earliest_year]['Montant'].sum()
    
    total_growth = ((latest_total - earliest_total) / earliest_total) * 100
    avg_annual_growth = total_growth / (latest_year - earliest_year)
    
    with col1:
        st.metric(
            "Budget Total 2025",
            format_currency(latest_total),
            f"{total_growth:.1f}% depuis {earliest_year}"
        )
    
    with col2:
        st.metric(
            "Croissance Annuelle Moyenne",
            f"{avg_annual_growth:.1f}%",
            "Sur la période"
        )
    
    with col3:
        top_categories = get_top_categories(df, n=1)
        if not top_categories.empty:
            top_category = top_categories.iloc[0]
            st.metric(
                "Mission la Plus Importante",
                top_category['Mission'],
                format_currency(top_category['Montant'])
            )
        else:
            st.metric("Mission la Plus Importante", "N/A", "N/A")
    
    with col4:
        if predictions_df is not None and not predictions_df.empty:
            future_total = predictions_df[predictions_df['Année'] == 2030]['Montant_Prédit'].sum()
            st.metric(
                "Prédiction 2030",
                format_currency(future_total),
                f"{((future_total - latest_total) / latest_total * 100):.1f}% vs 2025"
            )
        else:
            st.metric("Prédiction 2030", "N/A", "Calcul en cours")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Évolution Temporelle", 
        "🏛️ Comparaison Missions", 
        "📊 Répartition Budgétaire",
        "🔮 Prédictions",
        "📋 Analyse Détaillée"
    ])
    
    with tab1:
        st.subheader("Évolution des Dépenses par Mission (2005-2025)")
        
        # Interactive line chart
        fig_evolution = px.line(
            df, 
            x='Année', 
            y='Montant', 
            color='Mission',
            title="Évolution des Dépenses Budgétaires par Mission",
            labels={
                'Montant': 'Montant (Milliards €)',
                'Année': 'Année',
                'Mission': 'Mission'
            }
        )
        fig_evolution.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("Taux de Croissance Annuel par Mission")
        growth_data = []
        for mission in df['Mission'].unique():
            mission_data = df[df['Mission'] == mission].sort_values('Année')
            if len(mission_data) > 1:
                growth_rate = calculate_growth_rate(
                    mission_data['Montant'].iloc[0],
                    mission_data['Montant'].iloc[-1],
                    len(mission_data) - 1
                )
                growth_data.append({
                    'Mission': mission,
                    'Croissance Annuelle (%)': growth_rate,
                    'Variation Totale (%)': ((mission_data['Montant'].iloc[-1] - mission_data['Montant'].iloc[0]) / mission_data['Montant'].iloc[0]) * 100
                })
        
        growth_df = pd.DataFrame(growth_data)
        if not growth_df.empty:
            growth_df = growth_df.sort_values('Croissance Annuelle (%)', ascending=False)
            
            fig_growth = px.bar(
                growth_df,
                x='Mission',
                y='Croissance Annuelle (%)',
                title="Taux de Croissance Annuel Moyen par Mission",
                color='Croissance Annuelle (%)',
                color_continuous_scale='RdYlBu_r'
            )
            fig_growth.update_xaxis(tickangle=45)
            fig_growth.update_layout(height=500)
            st.plotly_chart(fig_growth, use_container_width=True)
    
    with tab2:
        st.subheader("Comparaison des Missions Budgétaires")
        
        # Select year for comparison
        comparison_year = st.selectbox(
            "Année de comparaison",
            sorted(df['Année'].unique(), reverse=True)
        )
        
        year_data = df[df['Année'] == comparison_year].copy()
        year_data = year_data.sort_values('Montant', ascending=True)
        
        # Horizontal bar chart
        fig_comparison = px.bar(
            year_data,
            x='Montant',
            y='Mission',
            orientation='h',
            title=f"Répartition des Dépenses par Mission - {comparison_year}",
            labels={'Montant': 'Montant (Milliards €)', 'Mission': 'Mission'},
            text='Montant'
        )
        fig_comparison.update_traces(texttemplate='%{text:.1f}B€', textposition='outside')
        fig_comparison.update_layout(height=600)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Percentage breakdown
        year_data['Pourcentage'] = (year_data['Montant'] / year_data['Montant'].sum()) * 100
        st.subheader(f"Répartition en Pourcentage - {comparison_year}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_pie = px.pie(
                year_data,
                values='Montant',
                names='Mission',
                title="Répartition du Budget par Mission"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.dataframe(
                year_data[['Mission', 'Montant', 'Pourcentage']].round(2),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Analyse de la Répartition Budgétaire")
        
        # Stacked area chart
        pivot_df = df.pivot(index='Année', columns='Mission', values='Montant').fillna(0)
        
        fig_stacked = go.Figure()
        for mission in pivot_df.columns:
            fig_stacked.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[mission],
                stackgroup='one',
                name=mission,
                mode='lines',
                line=dict(width=0.5),
                fillcolor=px.colors.qualitative.Set3[hash(mission) % len(px.colors.qualitative.Set3)]
            ))
        
        fig_stacked.update_layout(
            title="Évolution de la Répartition Budgétaire (Aires Empilées)",
            xaxis_title="Année",
            yaxis_title="Montant (Milliards €)",
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        # Percentage evolution
        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        fig_pct = go.Figure()
        for mission in pivot_pct.columns:
            fig_pct.add_trace(go.Scatter(
                x=pivot_pct.index,
                y=pivot_pct[mission],
                stackgroup='one',
                name=mission,
                mode='lines',
                line=dict(width=0.5)
            ))
        
        fig_pct.update_layout(
            title="Évolution de la Répartition Budgétaire (Pourcentages)",
            xaxis_title="Année",
            yaxis_title="Pourcentage (%)",
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_pct, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Prédictions Budgétaires 2026-2030")
        
        if predictions_df is not None and not predictions_df.empty:
            # Combine historical and predicted data
            historical_df = df.copy()
            historical_df['Type'] = 'Historique'
            historical_df['Montant_Prédit'] = historical_df['Montant']
            
            pred_df = predictions_df.copy()
            pred_df['Type'] = 'Prédit'
            pred_df['Montant'] = pred_df['Montant_Prédit']
            
            combined_df = pd.concat([historical_df, pred_df], ignore_index=True)
            
            # Interactive prediction chart
            fig_pred = px.line(
                combined_df,
                x='Année',
                y='Montant_Prédit',
                color='Mission',
                line_dash='Type',
                title="Évolution Historique et Prédictions Budgétaires",
                labels={
                    'Montant_Prédit': 'Montant (Milliards €)',
                    'Année': 'Année',
                    'Mission': 'Mission',
                    'Type': 'Type de Données'
                }
            )
            
            # Add vertical line to separate historical and predicted data
            fig_pred.add_vline(
                x=2025.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Limite Historique/Prédiction"
            )
            
            fig_pred.update_layout(height=600, hovermode='x unified')
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction summary
            st.subheader("Résumé des Prédictions 2030")
            
            pred_2030 = predictions_df[predictions_df['Année'] == 2030]
            hist_2025 = df[df['Année'] == 2025]
            
            summary_data = []
            for mission in pred_2030['Mission'].unique():
                pred_value = pred_2030[pred_2030['Mission'] == mission]['Montant_Prédit'].iloc[0]
                hist_value = hist_2025[hist_2025['Mission'] == mission]['Montant'].iloc[0] if mission in hist_2025['Mission'].values else 0
                
                if hist_value > 0:
                    growth = ((pred_value - hist_value) / hist_value) * 100
                else:
                    growth = 0
                
                summary_data.append({
                    'Mission': mission,
                    'Budget 2025 (Milliards €)': round(hist_value, 2),
                    'Prédiction 2030 (Milliards €)': round(pred_value, 2),
                    'Croissance Prédite (%)': round(growth, 1),
                    'Variation (Milliards €)': round(pred_value - hist_value, 2)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Croissance Prédite (%)', ascending=False)
            
            st.dataframe(summary_df, use_container_width=True)
            
        else:
            st.warning("⚠️ Prédictions en cours de génération. Veuillez patienter...")
    
    with tab5:
        st.subheader("📋 Analyse Détaillée et Export")
        
        # Top growing categories
        st.subheader("🚀 Missions en Forte Croissance")
        
        growth_analysis = []
        for mission in df['Mission'].unique():
            mission_data = df[df['Mission'] == mission].sort_values('Année')
            if len(mission_data) >= 2:
                start_value = mission_data['Montant'].iloc[0]
                end_value = mission_data['Montant'].iloc[-1]
                total_growth = ((end_value - start_value) / start_value) * 100
                
                growth_analysis.append({
                    'Mission': mission,
                    'Montant Initial (Milliards €)': round(start_value, 2),
                    'Montant Final (Milliards €)': round(end_value, 2),
                    'Croissance Totale (%)': round(total_growth, 1),
                    'Tendance': '📈' if total_growth > 0 else '📉'
                })
        
        growth_analysis_df = pd.DataFrame(growth_analysis)
        growth_analysis_df = growth_analysis_df.sort_values('Croissance Totale (%)', ascending=False)
        
        st.dataframe(growth_analysis_df, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Résumé Statistique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre de Missions Analysées", len(df['Mission'].unique()))
            st.metric("Période d'Analyse", f"{df['Année'].min()} - {df['Année'].max()}")
            st.metric("Points de Données", len(df))
        
        with col2:
            if not growth_analysis_df.empty:
                best_performer = growth_analysis_df.iloc[0]
                st.metric(
                    "Meilleure Performance",
                    best_performer['Mission'],
                    f"+{best_performer['Croissance Totale (%)']}%"
                )
                
                worst_performer = growth_analysis_df.iloc[-1]
                st.metric(
                    "Performance la Plus Faible",
                    worst_performer['Mission'],
                    f"{worst_performer['Croissance Totale (%)']}%"
                )
        
        # Data export
        st.subheader("💾 Export des Données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export historical data
            csv_historical = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Télécharger Données Historiques (CSV)",
                data=csv_historical,
                file_name=f"budget_france_historique_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export predictions
            if predictions_df is not None and not predictions_df.empty:
                csv_predictions = predictions_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 Télécharger Prédictions (CSV)",
                    data=csv_predictions,
                    file_name=f"budget_france_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export analysis
            if not growth_analysis_df.empty:
                csv_analysis = growth_analysis_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 Télécharger Analyse (CSV)",
                    data=csv_analysis,
                    file_name=f"budget_france_analyse_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🏛️ Données officielles du gouvernement français | Sources: data.gouv.fr, INSEE, budget.gouv.fr</p>
    <p>Dernière mise à jour: """ + datetime.now().strftime('%d/%m/%Y %H:%M') + """</p>
</div>
""", unsafe_allow_html=True)
