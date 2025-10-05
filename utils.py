import pandas as pd
import numpy as np
from typing import List, Dict, Any
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
