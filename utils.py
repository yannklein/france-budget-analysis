import pandas as pd
import numpy as np
from typing import List, Dict, Any
# For optional machine translation
from typing import Tuple
import locale

# Set French locale for number formatting (fallback to default if not available)
try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR')
    except:
        locale.setlocale(locale.LC_ALL, 'C')

def format_currency(amount: float, currency: str = "€") -> str:
    """
    Format currency amounts in French style.
    
    Args:
        amount: Amount to format
        currency: Currency symbol (default: €)
        
    Returns:
        Formatted currency string
    """
    if pd.isna(amount) or amount == 0:
        return f"0,0 Md{currency}"
    
    # Convert to billions if necessary
    if abs(amount) >= 1000:
        amount = amount / 1000
        suffix = f" Md{currency}"
    elif abs(amount) >= 1:
        suffix = f" Md{currency}"
    else:
        amount = amount * 1000
        suffix = f" M{currency}"
    
    try:
        # Use French number formatting
        formatted = locale.format_string("%.1f", amount, grouping=True)
        return formatted + suffix
    except:
        # Fallback formatting
        return f"{amount:,.1f}".replace(",", " ").replace(".", ",") + suffix

def calculate_growth_rate(start_value: float, end_value: float, years: int) -> float:
    """
    Calculate compound annual growth rate (CAGR).
    
    Args:
        start_value: Initial value
        end_value: Final value
        years: Number of years
        
    Returns:
        Annual growth rate as percentage
    """
    if start_value <= 0 or years <= 0:
        return 0.0
    
    if end_value <= 0:
        return -100.0
    
    try:
        cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    except:
        return 0.0

def get_top_categories(df: pd.DataFrame, year: int = None, n: int = 5) -> pd.DataFrame:
    """
    Get top spending categories for a given year.
    
    Args:
        df: Budget DataFrame
        year: Year to analyze (default: latest year)
        n: Number of top categories to return
        
    Returns:
        DataFrame with top categories
    """
    if df.empty:
        return pd.DataFrame()
    
    if year is None:
        year = df['Année'].max()
    
    year_data = df[df['Année'] == year].copy()
    
    if year_data.empty:
        return pd.DataFrame()
    
    # Group by mission and sum amounts
    top_categories = year_data.groupby('Mission')['Montant'].sum().reset_index()
    top_categories = top_categories.sort_values('Montant', ascending=False).head(n)
    
    return top_categories

def calculate_percentage_breakdown(df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """
    Calculate percentage breakdown of budget by mission.
    
    Args:
        df: Budget DataFrame
        year: Year to analyze (default: latest year)
        
    Returns:
        DataFrame with percentage breakdown
    """
    if df.empty:
        return pd.DataFrame()
    
    if year is None:
        year = df['Année'].max()
    
    year_data = df[df['Année'] == year].copy()
    
    if year_data.empty:
        return pd.DataFrame()
    
    # Calculate total and percentages
    total_budget = year_data['Montant'].sum()
    
    if total_budget == 0:
        return pd.DataFrame()
    
    breakdown = year_data.groupby('Mission')['Montant'].sum().reset_index()
    breakdown['Pourcentage'] = (breakdown['Montant'] / total_budget) * 100
    breakdown = breakdown.sort_values('Pourcentage', ascending=False)
    
    return breakdown

def identify_growth_trends(df: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """
    Identify spending trends by analyzing growth patterns.
    
    Args:
        df: Budget DataFrame
        min_years: Minimum years of data required
        
    Returns:
        DataFrame with trend analysis
    """
    if df.empty:
        return pd.DataFrame()
    
    trends = []
    
    for mission in df['Mission'].unique():
        mission_data = df[df['Mission'] == mission].sort_values('Année')
        
        if len(mission_data) < min_years:
            continue
        
        # Calculate various trend metrics
        start_amount = mission_data['Montant'].iloc[0]
        end_amount = mission_data['Montant'].iloc[-1]
        start_year = mission_data['Année'].iloc[0]
        end_year = mission_data['Année'].iloc[-1]
        
        # Total growth
        total_growth = ((end_amount - start_amount) / start_amount) * 100 if start_amount > 0 else 0
        
        # Annual growth rate
        years_span = end_year - start_year
        annual_growth = calculate_growth_rate(start_amount, end_amount, years_span)
        
        # Volatility (coefficient of variation)
        volatility = (mission_data['Montant'].std() / mission_data['Montant'].mean()) * 100 if mission_data['Montant'].mean() > 0 else 0
        
        # Trend classification
        if annual_growth > 5:
            trend_class = "Forte croissance"
        elif annual_growth > 2:
            trend_class = "Croissance modérée"
        elif annual_growth > -2:
            trend_class = "Stable"
        elif annual_growth > -5:
            trend_class = "Déclin modéré"
        else:
            trend_class = "Forte baisse"
        
        trends.append({
            'Mission': mission,
            'Montant_Initial': start_amount,
            'Montant_Final': end_amount,
            'Croissance_Totale_Pct': round(total_growth, 1),
            'Croissance_Annuelle_Pct': annual_growth,
            'Volatilité_Pct': round(volatility, 1),
            'Classification': trend_class,
            'Années_Analysées': years_span,
            'Points_Données': len(mission_data)
        })
    
    trends_df = pd.DataFrame(trends)
    
    if not trends_df.empty:
        trends_df = trends_df.sort_values('Croissance_Annuelle_Pct', ascending=False)
    
    return trends_df

def validate_budget_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate budget data quality and provide diagnostics.
    
    Args:
        df: Budget DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if df.empty:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame is empty")
        return validation
    
    # Check required columns
    required_columns = ['Année', 'Mission', 'Montant']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation['is_valid'] = False
        validation['errors'].append(f"Missing required columns: {missing_columns}")
        return validation
    
    # Check data types and values
    if not pd.api.types.is_numeric_dtype(df['Année']):
        validation['errors'].append("Column 'Année' must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['Montant']):
        validation['errors'].append("Column 'Montant' must be numeric")
    
    # Check for negative amounts
    negative_amounts = df['Montant'] < 0
    if negative_amounts.any():
        validation['warnings'].append(f"{negative_amounts.sum()} negative budget amounts found")
    
    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation['warnings'].append(f"Missing values found: {null_counts.to_dict()}")
    
    # Check year range
    if 'Année' in df.columns:
        year_range = df['Année'].max() - df['Année'].min()
        if year_range < 1:
            validation['warnings'].append("Data covers less than 2 years")
    
    # Calculate basic statistics
    validation['stats'] = {
        'total_records': len(df),
        'unique_missions': df['Mission'].nunique() if 'Mission' in df.columns else 0,
        'year_range': (int(df['Année'].min()), int(df['Année'].max())) if 'Année' in df.columns else (0, 0),
        'total_budget_latest': df[df['Année'] == df['Année'].max()]['Montant'].sum() if 'Année' in df.columns and 'Montant' in df.columns else 0,
        'avg_mission_budget': df.groupby('Mission')['Montant'].mean().mean() if 'Mission' in df.columns and 'Montant' in df.columns else 0
    }
    
    # Set validation status
    validation['is_valid'] = len(validation['errors']) == 0
    
    return validation

def export_data_with_metadata(df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> str:
    """
    Export data with metadata in CSV format.
    
    Args:
        df: Historical budget data
        predictions_df: Predictions data (optional)
        
    Returns:
        CSV string with metadata
    """
    from datetime import datetime
    
    # Create export data
    export_lines = []
    
    # Add metadata header
    export_lines.append("# Données Budgétaires de l'État Français")
    export_lines.append(f"# Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    export_lines.append(f"# Période: {df['Année'].min()}-{df['Année'].max()}")
    export_lines.append(f"# Missions analysées: {df['Mission'].nunique()}")
    export_lines.append("# Montants en milliards d'euros")
    export_lines.append("")
    
    # Add validation info
    validation = validate_budget_data(df)
    export_lines.append("# Validation des données:")
    export_lines.append(f"# - Status: {'✓ Valide' if validation['is_valid'] else '✗ Erreurs détectées'}")
    export_lines.append(f"# - Total enregistrements: {validation['stats']['total_records']}")
    export_lines.append(f"# - Budget total (dernière année): {format_currency(validation['stats']['total_budget_latest'])}")
    export_lines.append("")
    
    # Add column descriptions
    export_lines.append("# Colonnes:")
    export_lines.append("# - Année: Année budgétaire")
    export_lines.append("# - Mission: Mission budgétaire de l'État")
    export_lines.append("# - Montant: Montant alloué en milliards d'euros")
    if predictions_df is not None:
        export_lines.append("# - Montant_Prédit: Montant prédit (données futures uniquement)")
        export_lines.append("# - Confiance: Niveau de confiance de la prédiction (0-1)")
    export_lines.append("")
    
    # Convert to CSV
    historical_csv = df.to_csv(index=False, encoding='utf-8')
    
    if predictions_df is not None:
        predictions_csv = predictions_df.to_csv(index=False, encoding='utf-8')
        
        # Combine historical and predictions
        export_lines.append("# === DONNÉES HISTORIQUES ===")
        export_lines.append(historical_csv)
        export_lines.append("")
        export_lines.append("# === PRÉDICTIONS ===")
        export_lines.append(predictions_csv)
    else:
        export_lines.append("# === DONNÉES HISTORIQUES ===")
        export_lines.append(historical_csv)
    
    return "\n".join(export_lines)

def generate_insights(df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> List[str]:
    """
    Generate key insights from budget data analysis.
    
    Args:
        df: Historical budget data
        predictions_df: Predictions data (optional)
        
    Returns:
        List of insight strings
    """
    insights = []
    
    if df.empty:
        return ["Aucune donnée disponible pour l'analyse."]
    
    # Basic insights
    total_years = df['Année'].nunique()
    total_missions = df['Mission'].nunique()
    
    insights.append(f"📊 Analyse portant sur {total_years} années et {total_missions} missions budgétaires")
    
    # Budget evolution insight
    if total_years > 1:
        start_year = df['Année'].min()
        end_year = df['Année'].max()
        start_total = df[df['Année'] == start_year]['Montant'].sum()
        end_total = df[df['Année'] == end_year]['Montant'].sum()
        
        total_growth = ((end_total - start_total) / start_total) * 100
        annual_growth = total_growth / (end_year - start_year)
        
        if total_growth > 0:
            insights.append(f"📈 Le budget total a augmenté de {total_growth:.1f}% entre {start_year} et {end_year} (croissance annuelle moyenne: {annual_growth:.1f}%)")
        else:
            insights.append(f"📉 Le budget total a diminué de {abs(total_growth):.1f}% entre {start_year} et {end_year}")
    
    # Top missions insight
    latest_year = df['Année'].max()
    top_missions = get_top_categories(df, latest_year, 3)
    
    if not top_missions.empty:
        top_mission = top_missions.iloc[0]
        insights.append(f"🏛️ En {latest_year}, '{top_mission['Mission']}' représente la plus grosse mission avec {format_currency(top_mission['Montant'])}")
    
    # Growth trends insight
    trends = identify_growth_trends(df)
    if not trends.empty:
        fastest_growing = trends.iloc[0]
        if fastest_growing['Croissance_Annuelle_Pct'] > 5:
            insights.append(f"🚀 '{fastest_growing['Mission']}' affiche la plus forte croissance avec +{fastest_growing['Croissance_Annuelle_Pct']:.1f}% par an")
        
        # Declining missions
        declining = trends[trends['Croissance_Annuelle_Pct'] < -2]
        if not declining.empty:
            worst_decline = declining.iloc[-1]
            insights.append(f"📉 '{worst_decline['Mission']}' montre la plus forte baisse avec {worst_decline['Croissance_Annuelle_Pct']:.1f}% par an")
    
    # Predictions insights
    if predictions_df is not None and not predictions_df.empty:
        pred_2030_total = predictions_df[predictions_df['Année'] == 2030]['Montant_Prédit'].sum()
        current_total = df[df['Année'] == df['Année'].max()]['Montant'].sum()
        
        pred_growth = ((pred_2030_total - current_total) / current_total) * 100
        
        if pred_growth > 0:
            insights.append(f"🔮 Les prédictions suggèrent une croissance du budget de {pred_growth:.1f}% d'ici 2030")
        else:
            insights.append(f"🔮 Les prédictions suggèrent une baisse du budget de {abs(pred_growth):.1f}% d'ici 2030")
        
        # Fastest predicted growth
        pred_by_mission = []
        for mission in predictions_df['Mission'].unique():
            current_value = df[(df['Mission'] == mission) & (df['Année'] == df['Année'].max())]['Montant']
            pred_value = predictions_df[(predictions_df['Mission'] == mission) & (predictions_df['Année'] == 2030)]['Montant_Prédit']
            
            if not current_value.empty and not pred_value.empty:
                mission_growth = ((pred_value.iloc[0] - current_value.iloc[0]) / current_value.iloc[0]) * 100
                pred_by_mission.append({'Mission': mission, 'Growth': mission_growth})
        
        if pred_by_mission:
            pred_by_mission.sort(key=lambda x: x['Growth'], reverse=True)
            top_pred = pred_by_mission[0]
            if top_pred['Growth'] > 10:
                insights.append(f"🎯 '{top_pred['Mission']}' devrait connaître la plus forte croissance avec +{top_pred['Growth']:.1f}% prédits d'ici 2030")
    
    return insights

def adjust_to_constant_euros(df: pd.DataFrame, cpi_df: pd.DataFrame, base_year: int = None, amount_col: str = 'Montant') -> pd.DataFrame:
    """
    Deflate nominal amounts to constant euros using a CPI index.
    - df: must contain columns ['Année', amount_col]
    - cpi_df: columns ['Année','CPI'] where CPI is an index (any base)
    - base_year: year to express amounts in constant euros; if None, uses latest year in df
    - amount_col: name of the amount column to adjust (defaults to 'Montant')

    Returns a new DataFrame with the same schema, with amounts adjusted in-place.
    """
    if df is None or df.empty or cpi_df is None or cpi_df.empty:
        return df

    if base_year is None:
        base_year = int(df['Année'].max())

    if base_year not in set(cpi_df['Année'].astype(int)):
        return df

    cpi_df = cpi_df[['Année', 'CPI']].copy()
    cpi_df['Année'] = cpi_df['Année'].astype(int)

    base_cpi = float(cpi_df.loc[cpi_df['Année'] == base_year, 'CPI'].iloc[0])
    if base_cpi == 0:
        return df

    # Normalize CPI so base_year CPI = 100
    cpi_df['CPI_Norm'] = cpi_df['CPI'] / base_cpi * 100.0

    merged = df.merge(cpi_df[['Année', 'CPI_Norm']], on='Année', how='left', validate='many_to_one')
    merged['CPI_Norm'] = merged['CPI_Norm'].replace({0: np.nan})
    merged['CPI_Norm'] = merged['CPI_Norm'].fillna(method='ffill').fillna(method='bfill')

    if amount_col in merged.columns:
        merged[amount_col] = merged[amount_col] * (100.0 / merged['CPI_Norm'])

    return merged.drop(columns=['CPI_Norm'])

# === Internationalization (i18n) helpers ===

# Minimal list of EU languages (ISO 639-1 codes)
EU_LANGUAGES: List[Dict[str, str]] = [
    {"code": "bg", "name": "Български"},
    {"code": "hr", "name": "Hrvatski"},
    {"code": "cs", "name": "Čeština"},
    {"code": "da", "name": "Dansk"},
    {"code": "nl", "name": "Nederlands"},
    {"code": "en", "name": "English"},
    {"code": "et", "name": "Eesti"},
    {"code": "fi", "name": "Suomi"},
    {"code": "fr", "name": "Français"},
    {"code": "de", "name": "Deutsch"},
    {"code": "el", "name": "Ελληνικά"},
    {"code": "hu", "name": "Magyar"},
    {"code": "ga", "name": "Gaeilge"},
    {"code": "it", "name": "Italiano"},
    {"code": "lv", "name": "Latviešu"},
    {"code": "lt", "name": "Lietuvių"},
    {"code": "mt", "name": "Malti"},
    {"code": "pl", "name": "Polski"},
    {"code": "pt", "name": "Português"},
    {"code": "ro", "name": "Română"},
    {"code": "sk", "name": "Slovenčina"},
    {"code": "sl", "name": "Slovenščina"},
    {"code": "es", "name": "Español"},
    {"code": "sv", "name": "Svenska"},
]

_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Sidebar & common
    "sidebar.config": {
        "en": "Settings",
        "fr": "Configuration",
    },
    "sidebar.language": {
        "en": "Language",
        "fr": "Langue",
    },
    "sidebar.data_source": {
        "en": "Data source",
        "fr": "Source de données",
    },
    "sidebar.year_range": {
        "en": "Year range",
        "fr": "Plage d'années",
    },
    "sidebar.load_data": {
        "en": "Load data",
        "fr": "Charger les données",
    },
    "toggle.inflation": {
        "en": "Adjust for inflation (constant euros)",
        "fr": "Ajuster pour l'inflation (euros constants)",
    },
    "toggle.gov_periods": {
        "en": "Show government periods",
        "fr": "Afficher périodes gouvernementales",
    },
    "toggle.events": {
        "en": "Show key events",
        "fr": "Afficher événements majeurs",
    },
    "toggle.debt": {
        "en": "Include debt interest",
        "fr": "Inclure charge de la dette (intérêts)",
    },
    # Tabs
    "tab.evolution": {"en": "Time Evolution", "fr": "Évolution Temporelle"},
    "tab.compare": {"en": "Mission Comparison", "fr": "Comparaison Missions"},
    "tab.split": {"en": "Budget Breakdown", "fr": "Répartition Budgétaire"},
    "tab.pred": {"en": "Predictions", "fr": "Prédictions"},
    "tab.details": {"en": "Detailed Analysis", "fr": "Analyse Détaillée"},
    # Titles and headers
    "title.app": {
        "en": "French State Budget - Analysis and Predictions",
        "fr": "Budget de l'État Français - Analyse et Prédictions",
    },
    "title.total_budget": {
        "en": "State Total Budget Evolution",
        "fr": "Évolution du Budget Total de l'État",
    },
    "desc.app": {
        "en": "This application analyzes the evolution of French State expenditures over 20 years and provides AI-based predictions.",
        "fr": "Cette application analyse l'évolution des dépenses budgétaires de l'État français sur 20 ans et propose des prédictions basées sur l'intelligence artificielle.",
    },
    "header.evolution_missions": {
        "en": "Evolution of Expenditures by Mission (2005-2025)",
        "fr": "Évolution des Dépenses par Mission (2005-2025)",
    },
    "title.mission_evolution": {
        "en": "Evolution of Budget Expenditures by Mission",
        "fr": "Évolution des Dépenses Budgétaires par Mission",
    },
    "header.growth_rate": {
        "en": "Annual Growth Rate by Mission",
        "fr": "Taux de Croissance Annuel par Mission",
    },
    "title.growth_rate": {
        "en": "Average Annual Growth Rate by Mission",
        "fr": "Taux de Croissance Annuel Moyen par Mission",
    },
    "header.compare": {"en": "Mission Comparison", "fr": "Comparaison des Missions Budgétaires"},
    "title.comparison": {
        "en": "Spending by Mission",
        "fr": "Répartition des Dépenses par Mission",
    },
    "header.split": {"en": "Budget Breakdown Analysis", "fr": "Analyse de la Répartition Budgétaire"},
    "title.stacked_area": {
        "en": "Budget Share Evolution (Stacked Areas)",
        "fr": "Évolution de la Répartition Budgétaire (Aires Empilées)",
    },
    "title.percentage_evolution": {
        "en": "Budget Share Evolution (Percentages)",
        "fr": "Évolution de la Répartition Budgétaire (Pourcentages)",
    },
    "header.pred": {"en": "Budget Predictions 2026-2030", "fr": "Prédictions Budgétaires 2026-2030"},
    "title.history_predictions": {
        "en": "Historical Evolution and Budget Predictions",
        "fr": "Évolution Historique et Prédictions Budgétaires",
    },
    "header.pred_summary": {"en": "2030 Predictions Summary", "fr": "Résumé des Prédictions 2030"},
    "header.details_export": {"en": "Detailed Analysis and Export", "fr": "Analyse Détaillée et Export"},
    "header.top_growth": {"en": "Fastest Growing Missions", "fr": "Missions en Forte Croissance"},
    "header.stats_summary": {"en": "Statistical Summary", "fr": "Résumé Statistique"},
    # Metrics
    "metric.num_missions": {"en": "Number of Missions Analyzed", "fr": "Nombre de Missions Analysées"},
    "metric.analysis_period": {"en": "Analysis Period", "fr": "Période d'Analyse"},
    "metric.data_points": {"en": "Data Points", "fr": "Points de Données"},
    "metric.best_perf": {"en": "Best Performance", "fr": "Meilleure Performance"},
    "metric.worst_perf": {"en": "Lowest Performance", "fr": "Performance la Plus Faible"},
    # Export
    "export.header": {"en": "Export Data", "fr": "Export des Données"},
    "button.download_historical": {"en": "Download Historical (CSV)", "fr": "Télécharger Données Historiques (CSV)"},
    "button.download_predictions": {"en": "Download Predictions (CSV)", "fr": "Télécharger Prédictions (CSV)"},
    "button.download_analysis": {"en": "Download Analysis (CSV)", "fr": "Télécharger Analyse (CSV)"},
    # Insights (templates)
    "insights.no_data": {"en": "No data available for analysis.", "fr": "Aucune donnée disponible pour l'analyse."},
    "insights.header": {"en": "Analysis over {years} years and {missions} missions", "fr": "Analyse portant sur {years} années et {missions} missions budgétaires"},
    "insights.total_up": {"en": "Total budget increased by {pct}% between {start} and {end} (avg yearly: {avg}%)", "fr": "Le budget total a augmenté de {pct}% entre {start} et {end} (croissance annuelle moyenne: {avg}%)"},
    "insights.total_down": {"en": "Total budget decreased by {pct}% between {start} and {end}", "fr": "Le budget total a diminué de {pct}% entre {start} et {end}"},
    "insights.top_mission": {"en": "In {year}, '{mission}' is the largest mission with {amount}", "fr": "En {year}, '{mission}' représente la plus grosse mission avec {amount}"},
    "insights.fastest": {"en": "'{mission}' shows the fastest growth with +{pct}% per year", "fr": "'{mission}' affiche la plus forte croissance avec +{pct}% par an"},
    "insights.decline": {"en": "'{mission}' shows the steepest decline with {pct}% per year", "fr": "'{mission}' montre la plus forte baisse avec {pct}% par an"},
    "insights.pred_up": {"en": "Predictions suggest a budget increase of {pct}% by 2030", "fr": "Les prédictions suggèrent une croissance du budget de {pct}% d'ici 2030"},
    "insights.pred_down": {"en": "Predictions suggest a budget decrease of {pct}% by 2030", "fr": "Les prédictions suggèrent une baisse du budget de {pct}% d'ici 2030"},
    "insights.pred_top": {"en": "'{mission}' is expected to grow the most with +{pct}% by 2030", "fr": "'{mission}' devrait connaître la plus forte croissance avec +{pct}% d'ici 2030"},
    # Tab analyses
    "analysis.evolution": {
        "en": "This view highlights how each mission evolves over time. Look for persistent trends, inflection points, and the impact of major events or government changes.",
        "fr": "Cette vue met en évidence l'évolution de chaque mission dans le temps. Recherchez les tendances persistantes, les points d'inflexion et l'impact des événements majeurs ou des changements de gouvernement.",
    },
    "analysis.compare": {
        "en": "Compare missions for a specific year to identify the largest spending areas and relative weights. The percentage panel helps contextualize the distribution.",
        "fr": "Comparez les missions pour une année donnée afin d'identifier les postes les plus importants et leurs poids relatifs. Le panneau des pourcentages aide à contextualiser la répartition.",
    },
    "analysis.split": {
        "en": "The stacked shares show how the composition of the budget changes across years. Watch for structural shifts where some missions gain or lose share.",
        "fr": "Les aires empilées montrent l'évolution de la composition du budget au fil des années. Observez les changements structurels où certaines missions gagnent ou perdent en part.",
    },
    "analysis.pred": {
        "en": "Predictions extend recent patterns with constraints. Treat them as scenarios: validate against policy plans and macroeconomic assumptions before drawing conclusions.",
        "fr": "Les prédictions prolongent les tendances récentes avec des contraintes. Considérez-les comme des scénarios : confrontez-les aux plans de politique publique et aux hypothèses macroéconomiques avant d'en tirer des conclusions.",
    },
    "analysis.details": {
        "en": "Use the summary and growth tables to drill down into drivers. Export data to replicate the analysis or integrate it with external sources.",
        "fr": "Utilisez le résumé et les tableaux de croissance pour approfondir les déterminants. Exportez les données pour reproduire l'analyse ou les intégrer à des sources externes.",
    },
    # Data sources
    "sources.title": {
        "en": "📚 Data Sources and References",
        "fr": "📚 Sources de Données et Références",
    },
    "sources.budget": {
        "en": "**Budget Data**: French State Budget missions from data.gouv.fr (Ministry of Economy and Finance)",
        "fr": "**Données Budgétaires** : Missions du budget de l'État français depuis data.gouv.fr (Ministère de l'Économie et des Finances)",
    },
    "sources.cpi": {
        "en": "**Inflation Data**: Consumer Price Index (CPI) from INSEE (National Institute of Statistics)",
        "fr": "**Données d'Inflation** : Indice des Prix à la Consommation (IPC) de l'INSEE (Institut National de la Statistique)",
    },
    "sources.debt": {
        "en": "**Debt Interest**: State debt interest payments modeled from public debt statistics",
        "fr": "**Intérêts de la Dette** : Paiements d'intérêts de la dette publique modélisés à partir des statistiques de dette publique",
    },
    "sources.methodology": {
        "en": "**Methodology**: Data processing, inflation adjustment, and predictions follow standard economic analysis practices",
        "fr": "**Méthodologie** : Traitement des données, ajustement inflationniste et prédictions suivent les pratiques standard d'analyse économique",
    },
    "sources.disclaimer": {
        "en": "**Disclaimer**: This analysis is for informational purposes. Consult official government sources for official budget figures.",
        "fr": "**Avertissement** : Cette analyse est à des fins d'information. Consultez les sources gouvernementales officielles pour les chiffres budgétaires officiels.",
    },
}

def get_eu_languages() -> List[Dict[str, str]]:
    return EU_LANGUAGES

_TRANSLATION_CACHE: Dict[Tuple[str, str], str] = {}

def _mt_translate_text(text: str, target_lang: str) -> str | None:
    """
    Best-effort machine translation using optional libraries. Returns None on failure.
    Caches results to avoid repeated lookups.
    """
    if not text or not target_lang:
        return None
    key = (text, target_lang)
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    # Try deep_translator first
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        if isinstance(translated, str) and translated.strip():
            _TRANSLATION_CACHE[key] = translated
            return translated
    except Exception:
        pass

    # Try googletrans as a fallback
    try:
        from googletrans import Translator  # type: ignore
        translator = Translator()
        res = translator.translate(text, dest=target_lang)
        translated = getattr(res, 'text', None)
        if isinstance(translated, str) and translated.strip():
            _TRANSLATION_CACHE[key] = translated
            return translated
    except Exception:
        pass

    return None

def translate(key: str, lang: str, default: str | None = None) -> str:
    lang = (lang or "en").lower()
    entry = _TRANSLATIONS.get(key, {})
    # Exact hit
    if lang in entry:
        return entry[lang]
    # Prefer English seed
    seed = entry.get("en", default if default is not None else key)
    # Attempt machine translation for EU languages
    mt = _mt_translate_text(seed, lang)
    if mt:
        # Cache into entry for future calls in this session
        entry[lang] = mt
        _TRANSLATIONS[key] = entry
        return mt
    # Fallback to English or default
    if "en" in entry:
        return entry["en"]
    return default if default is not None else key

# Prefill missing EU translations by copying English as a machine-translation fallback
def _prefill_eu_translations() -> None:
    eu_codes = {opt["code"] for opt in EU_LANGUAGES}
    for key, mapping in _TRANSLATIONS.items():
        en_val = mapping.get("en")
        if not en_val:
            continue
        for code in eu_codes:
            if code not in mapping:
                mapping[code] = en_val

_prefill_eu_translations()

def generate_insights_i18n(df: pd.DataFrame, predictions_df: pd.DataFrame | None, lang: str) -> List[str]:
    """
    Generate localized insights using translation templates.
    """
    messages: List[str] = []
    if df is None or df.empty:
        messages.append(translate("insights.no_data", lang, "Aucune donnée disponible pour l'analyse."))
        return messages

    total_years = int(df['Année'].nunique())
    total_missions = int(df['Mission'].nunique()) if 'Mission' in df.columns else 0
    header_tpl = translate("insights.header", lang, "Analyse portant sur {years} années et {missions} missions budgétaires")
    messages.append(header_tpl.format(years=total_years, missions=total_missions))

    if total_years > 1:
        start_year = int(df['Année'].min())
        end_year = int(df['Année'].max())
        start_total = float(df[df['Année'] == start_year]['Montant'].sum())
        end_total = float(df[df['Année'] == end_year]['Montant'].sum())
        if start_total > 0:
            total_growth = (end_total - start_total) / start_total * 100.0
            annual_growth = total_growth / (end_year - start_year)
            if total_growth >= 0:
                tpl = translate("insights.total_up", lang, "Le budget total a augmenté de {pct}% entre {start} et {end} (croissance annuelle moyenne: {avg}%)")
                messages.append(tpl.format(pct=f"{total_growth:.1f}", start=start_year, end=end_year, avg=f"{annual_growth:.1f}"))
            else:
                tpl = translate("insights.total_down", lang, "Le budget total a diminué de {pct}% entre {start} et {end}")
                messages.append(tpl.format(pct=f"{abs(total_growth):.1f}", start=start_year, end=end_year))

    latest_year = int(df['Année'].max())
    top = get_top_categories(df, latest_year, 1)
    if top is not None and not top.empty:
        row = top.iloc[0]
        tpl = translate("insights.top_mission", lang, "En {year}, '{mission}' représente la plus grosse mission avec {amount}")
        messages.append(tpl.format(year=latest_year, mission=row['Mission'], amount=format_currency(row['Montant'])))

    trends = identify_growth_trends(df)
    if trends is not None and not trends.empty:
        fastest = trends.iloc[0]
        if float(fastest['Croissance_Annuelle_Pct']) > 5:
            tpl = translate("insights.fastest", lang, "'{mission}' affiche la plus forte croissance avec +{pct}% par an")
            messages.append(tpl.format(mission=fastest['Mission'], pct=f"{float(fastest['Croissance_Annuelle_Pct']):.1f}"))
        declining = trends[trends['Croissance_Annuelle_Pct'] < -2]
        if not declining.empty:
            worst = declining.iloc[-1]
            tpl = translate("insights.decline", lang, "'{mission}' montre la plus forte baisse avec {pct}% par an")
            messages.append(tpl.format(mission=worst['Mission'], pct=f"{float(worst['Croissance_Annuelle_Pct']):.1f}"))

    if predictions_df is not None and not predictions_df.empty:
        pred_2030_total = float(predictions_df[predictions_df['Année'] == 2030]['Montant_Prédit'].sum())
        current_total = float(df[df['Année'] == latest_year]['Montant'].sum())
        if current_total > 0:
            pred_growth = (pred_2030_total - current_total) / current_total * 100.0
            if pred_growth >= 0:
                tpl = translate("insights.pred_up", lang, "Les prédictions suggèrent une croissance du budget de {pct}% d'ici 2030")
                messages.append(tpl.format(pct=f"{pred_growth:.1f}"))
            else:
                tpl = translate("insights.pred_down", lang, "Les prédictions suggèrent une baisse du budget de {pct}% d'ici 2030")
                messages.append(tpl.format(pct=f"{abs(pred_growth):.1f}"))

        by_mission = []
        for mission in predictions_df['Mission'].unique():
            curr = df[(df['Mission'] == mission) & (df['Année'] == latest_year)]['Montant']
            pred = predictions_df[(predictions_df['Mission'] == mission) & (predictions_df['Année'] == 2030)]['Montant_Prédit']
            if not curr.empty and not pred.empty and float(curr.iloc[0]) > 0:
                growth = (float(pred.iloc[0]) - float(curr.iloc[0])) / float(curr.iloc[0]) * 100.0
                by_mission.append((mission, growth))
        if by_mission:
            by_mission.sort(key=lambda x: x[1], reverse=True)
            m, g = by_mission[0]
            if g > 10:
                tpl = translate("insights.pred_top", lang, "'{mission}' devrait connaître la plus forte croissance avec +{pct}% d'ici 2030")
                messages.append(tpl.format(mission=m, pct=f"{g:.1f}"))

    return messages
