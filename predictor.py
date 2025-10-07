import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BudgetPredictor:
    """Advanced budget prediction using multiple machine learning models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['Année', 'Mission_Encoded', 'Tendance', 'Cycle_Économique']
        
    def predict_future_spending(self, df: pd.DataFrame, predict_years: list = None) -> pd.DataFrame:
        """
        Predict future budget spending using machine learning models.
        
        Args:
            df: Historical budget data with columns [Année, Mission, Montant]
            predict_years: Years to predict (default: 2025-2030)
            
        Returns:
            DataFrame with predictions
        """
        
        if predict_years is None:
            predict_years = list(range(2025, 2031))
        
        # Prepare features
        df_features = self._prepare_features(df)
        
        # Train models for each mission
        predictions = []
        
        for mission in df['Mission'].unique():
            mission_data = df_features[df_features['Mission'] == mission].copy()
            
            if len(mission_data) < 3:  # Need minimum data points
                # Use simple linear trend for missions with limited data
                pred = self._simple_linear_prediction(df, mission, predict_years)
                predictions.extend(pred)
            else:
                # Use advanced ML models
                pred = self._ml_prediction(mission_data, mission, predict_years)
                predictions.extend(pred)
        
        result_df = pd.DataFrame(predictions)
        return result_df.sort_values(['Année', 'Mission']).reset_index(drop=True)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        
        df_features = df.copy()
        
        # Encode mission names
        missions = df['Mission'].unique()
        mission_encoding = {mission: i for i, mission in enumerate(missions)}
        df_features['Mission_Encoded'] = df_features['Mission'].map(mission_encoding)
        
        # Add trend feature (normalized year)
        min_year = df_features['Année'].min()
        df_features['Tendance'] = df_features['Année'] - min_year
        
        # Add economic cycle feature (simplified)
        df_features['Cycle_Économique'] = df_features['Année'].apply(self._get_economic_cycle)
        
        # Add growth rate feature
        df_features['Taux_Croissance'] = 0.0
        for mission in df['Mission'].unique():
            mission_mask = df_features['Mission'] == mission
            mission_data = df_features[mission_mask].sort_values('Année')
            
            if len(mission_data) > 1:
                growth_rates = mission_data['Montant'].pct_change().fillna(0)
                df_features.loc[mission_mask, 'Taux_Croissance'] = growth_rates.values
        
        # Add moving average
        df_features['Moyenne_Mobile'] = 0.0
        for mission in df['Mission'].unique():
            mission_mask = df_features['Mission'] == mission
            mission_data = df_features[mission_mask].sort_values('Année')
            
            if len(mission_data) >= 3:
                ma = mission_data['Montant'].rolling(window=3, center=True).mean().fillna(mission_data['Montant'])
                df_features.loc[mission_mask, 'Moyenne_Mobile'] = ma.values
            else:
                df_features.loc[mission_mask, 'Moyenne_Mobile'] = mission_data['Montant'].values
        
        return df_features
    
    def _get_economic_cycle(self, year: int) -> float:
        """Get economic cycle indicator for a given year."""
        
        # Simplified economic cycle based on known recessions/booms
        crisis_years = {
            2008: -0.8, 2009: -1.0,  # Financial crisis
            2020: -0.9, 2021: -0.5,  # COVID crisis
            2011: -0.3, 2012: -0.4,  # European debt crisis
        }
        
        boom_years = {
            2006: 0.5, 2007: 0.6,    # Pre-crisis boom
            2017: 0.3, 2018: 0.4, 2019: 0.3,  # Economic recovery
        }
        
        if year in crisis_years:
            return crisis_years[year]
        elif year in boom_years:
            return boom_years[year]
        else:
            return 0.0  # Neutral
    
    def _ml_prediction(self, mission_data: pd.DataFrame, mission: str, predict_years: list) -> list:
        """Use machine learning models for prediction."""
        
        # Prepare training data
        X = mission_data[['Année', 'Mission_Encoded', 'Tendance', 'Cycle_Économique', 'Taux_Croissance']].fillna(0)
        y = mission_data['Montant']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train ensemble of models
        models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        model_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                
                # Calculate model score (R²)
                y_pred_train = model.predict(X_scaled)
                score = r2_score(y, y_pred_train)
                model_scores[name] = max(0, score)  # Ensure non-negative weights
                
                # Make predictions
                predictions = []
                for year in predict_years:
                    # Create features for prediction year
                    X_pred = np.array([[
                        year,
                        mission_data['Mission_Encoded'].iloc[0],
                        year - mission_data['Année'].min(),
                        self._get_economic_cycle(year),
                        mission_data['Taux_Croissance'].mean()
                    ]])
                    
                    X_pred_scaled = scaler.transform(X_pred)
                    pred = model.predict(X_pred_scaled)[0]
                    predictions.append(pred)
                
                model_predictions[name] = predictions
                
            except Exception as e:
                # Fallback to simple trend if model fails
                model_predictions[name] = self._simple_trend_prediction(mission_data, predict_years)
                model_scores[name] = 0.1
        
        # Ensemble prediction using weighted average
        total_weight = sum(model_scores.values()) or 1
        weights = {name: score/total_weight for name, score in model_scores.items()}
        
        ensemble_predictions = []
        for i in range(len(predict_years)):
            weighted_pred = sum(
                weights[name] * model_predictions[name][i] 
                for name in model_predictions
            )
            
            # Apply constraints (no negative spending, reasonable growth)
            last_value = mission_data['Montant'].iloc[-1]
            max_growth = 1.5  # Maximum 50% growth per year
            min_value = last_value * 0.5  # Minimum 50% of last value
            max_value = last_value * (max_growth ** (predict_years[i] - mission_data['Année'].max()))
            
            constrained_pred = max(min_value, min(max_value, weighted_pred))
            ensemble_predictions.append(constrained_pred)
        
        # Return predictions
        results = []
        for i, year in enumerate(predict_years):
            results.append({
                'Année': year,
                'Mission': mission,
                'Montant_Prédit': ensemble_predictions[i],
                'Confiance': min(1.0, sum(model_scores.values()) / len(model_scores))
            })
        
        return results
    
    def _simple_linear_prediction(self, df: pd.DataFrame, mission: str, predict_years: list) -> list:
        """Simple linear regression prediction for missions with limited data."""
        
        mission_data = df[df['Mission'] == mission].sort_values('Année')
        
        if len(mission_data) < 2:
            # If only one data point, assume constant spending
            last_value = mission_data['Montant'].iloc[-1] if not mission_data.empty else 1.0
            return [
                {
                    'Année': year,
                    'Mission': mission,
                    'Montant_Prédit': last_value,
                    'Confiance': 0.3
                }
                for year in predict_years
            ]
        
        # Fit simple linear regression
        X = mission_data['Année'].values.reshape(-1, 1)
        y = mission_data['Montant'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate confidence based on R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        confidence = max(0.2, min(0.8, r2))
        
        # Make predictions
        results = []
        for year in predict_years:
            pred = model.predict([[year]])[0]
            
            # Apply growth constraints
            last_value = mission_data['Montant'].iloc[-1]
            years_ahead = year - mission_data['Année'].max()
            max_pred = last_value * (1.3 ** years_ahead)  # Max 30% annual growth
            min_pred = last_value * (0.7 ** years_ahead)  # Max 30% annual decline
            
            constrained_pred = max(min_pred, min(max_pred, pred))
            
            results.append({
                'Année': year,
                'Mission': mission,
                'Montant_Prédit': max(0.1, constrained_pred),  # Minimum 0.1 billion
                'Confiance': confidence
            })
        
        return results
    
    def _simple_trend_prediction(self, mission_data: pd.DataFrame, predict_years: list) -> list:
        """Simple trend-based prediction as fallback."""
        
        if len(mission_data) < 2:
            return [mission_data['Montant'].iloc[-1]] * len(predict_years)
        
        # Calculate simple trend
        x = mission_data['Année'].values
        y = mission_data['Montant'].values
        
        # Linear fit
        z = np.polyfit(x, y, 1)
        
        predictions = []
        for year in predict_years:
            pred = z[0] * year + z[1]
            predictions.append(max(0.1, pred))  # Minimum 0.1 billion
        
        return predictions
    
    def get_prediction_summary(self, df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
        """Generate summary statistics for predictions."""
        
        if predictions_df is None or predictions_df.empty:
            return {}
        
        # Get latest historical year
        latest_year = df['Année'].max()
        latest_total = df[df['Année'] == latest_year]['Montant'].sum()
        
        # Get prediction for 2030
        pred_2030 = predictions_df[predictions_df['Année'] == 2030]['Montant_Prédit'].sum()
        
        # Calculate overall growth
        years_ahead = 2030 - latest_year
        total_growth = ((pred_2030 - latest_total) / latest_total) * 100
        annual_growth = total_growth / years_ahead
        
        # Find fastest growing missions
        growth_by_mission = []
        for mission in predictions_df['Mission'].unique():
            hist_value = df[(df['Mission'] == mission) & (df['Année'] == latest_year)]['Montant']
            pred_value = predictions_df[(predictions_df['Mission'] == mission) & (predictions_df['Année'] == 2030)]['Montant_Prédit']
            
            if not hist_value.empty and not pred_value.empty:
                hist_val = hist_value.iloc[0]
                pred_val = pred_value.iloc[0]
                mission_growth = ((pred_val - hist_val) / hist_val) * 100
                
                growth_by_mission.append({
                    'Mission': mission,
                    'Croissance_Prédite': mission_growth
                })
        
        growth_df = pd.DataFrame(growth_by_mission)
        top_growing = growth_df.nlargest(3, 'Croissance_Prédite') if not growth_df.empty else pd.DataFrame()
        
        return {
            'budget_actuel': latest_total,
            'budget_predit_2030': pred_2030,
            'croissance_totale_pct': total_growth,
            'croissance_annuelle_pct': annual_growth,
            'missions_croissance_forte': top_growing.to_dict('records') if not top_growing.empty else [],
            'nombre_missions_analysees': len(predictions_df['Mission'].unique()),
            'confiance_moyenne': predictions_df['Confiance'].mean() if 'Confiance' in predictions_df.columns else 0.5
        }
