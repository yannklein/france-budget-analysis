"""
Machine learning prediction module for budget forecasting.

This module provides advanced ML-based predictions for future government
budget spending using ensemble methods combining Linear Regression and
Random Forest models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_PREDICTION_YEARS,
    ECONOMIC_CYCLES,
    MAX_ANNUAL_GROWTH_RATE,
    MAX_MONETARY_VALUE,
    MIN_MONETARY_VALUE,
    MIN_PREDICTION_VALUE,
    MIN_VALUE_RETENTION,
    ML_CONFIG,
)

warnings.filterwarnings("ignore")


class BudgetPredictor:
    """
    Advanced budget prediction using ensemble machine learning models.

    This class trains separate models for each budget mission and combines
    predictions using a weighted ensemble approach based on model performance.

    Attributes:
        models: Dictionary storing trained models per mission.
        scalers: Dictionary storing feature scalers per mission.
        feature_columns: List of feature names used for prediction.

    Example:
        >>> predictor = BudgetPredictor()
        >>> predictions = predictor.predict_future_spending(historical_df)
        >>> print(predictions.head())
    """

    def __init__(self) -> None:
        """Initialize the BudgetPredictor with empty model storage."""
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_columns = [
            "Annee",
            "Mission_Encoded",
            "Tendance",
            "Cycle_Economique",
        ]

    def predict_future_spending(
        self,
        df: pd.DataFrame,
        predict_years: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Predict future budget spending using machine learning models.

        Args:
            df: Historical budget data with columns [Annee, Mission, Montant].
            predict_years: Years to predict (default: 2025-2030).

        Returns:
            DataFrame with columns [Annee, Mission, Montant_Predit, Confiance].
        """
        if predict_years is None:
            predict_years = DEFAULT_PREDICTION_YEARS

        # Prepare features for all missions
        df_features = self._prepare_features(df)

        predictions: list[dict[str, Any]] = []

        for mission in df["Mission"].unique():
            mission_data = df_features[df_features["Mission"] == mission].copy()

            if len(mission_data) < ML_CONFIG["min_data_points"]:
                # Use simple linear trend for missions with limited data
                pred = self._simple_linear_prediction(df, mission, predict_years)
            else:
                # Use advanced ML models
                pred = self._ml_prediction(mission_data, mission, predict_years)

            predictions.extend(pred)

        result_df = pd.DataFrame(predictions)
        return result_df.sort_values(["Annee", "Mission"]).reset_index(drop=True)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning models.

        Creates derived features including mission encoding, trend indicators,
        economic cycle values, growth rates, and moving averages.

        Args:
            df: Raw budget DataFrame with [Annee, Mission, Montant].

        Returns:
            DataFrame with additional feature columns.
        """
        df_features = df.copy()

        # Encode mission names as integers
        missions = df["Mission"].unique()
        mission_encoding = {mission: i for i, mission in enumerate(missions)}
        df_features["Mission_Encoded"] = df_features["Mission"].map(mission_encoding)

        # Add trend feature (years since start)
        min_year = df_features["Annee"].min()
        df_features["Tendance"] = df_features["Annee"] - min_year

        # Add economic cycle indicator
        df_features["Cycle_Economique"] = df_features["Annee"].apply(
            self._get_economic_cycle
        )

        # Calculate growth rate per mission
        df_features["Taux_Croissance"] = 0.0
        for mission in df["Mission"].unique():
            mission_mask = df_features["Mission"] == mission
            mission_data = df_features[mission_mask].sort_values("Annee")

            if len(mission_data) > 1:
                growth_rates = mission_data["Montant"].pct_change().fillna(0)
                df_features.loc[mission_mask, "Taux_Croissance"] = growth_rates.values

        # Calculate moving average per mission
        df_features["Moyenne_Mobile"] = 0.0
        for mission in df["Mission"].unique():
            mission_mask = df_features["Mission"] == mission
            mission_data = df_features[mission_mask].sort_values("Annee")

            if len(mission_data) >= 3:
                ma = (
                    mission_data["Montant"]
                    .rolling(window=3, center=True)
                    .mean()
                    .fillna(mission_data["Montant"])
                )
                df_features.loc[mission_mask, "Moyenne_Mobile"] = ma.values
            else:
                df_features.loc[mission_mask, "Moyenne_Mobile"] = mission_data[
                    "Montant"
                ].values

        return df_features

    def _get_economic_cycle(self, year: int) -> float:
        """
        Get economic cycle indicator for a given year.

        Returns a value between -1 (crisis) and +1 (boom) based on
        historical economic events.

        Args:
            year: Year to evaluate.

        Returns:
            Economic cycle indicator (-1.0 to 1.0).
        """
        # Check crisis years
        crisis_years = ECONOMIC_CYCLES["crisis"]["years"]
        if year in crisis_years:
            return crisis_years[year]

        # Check boom years
        boom_years = ECONOMIC_CYCLES["boom"]["years"]
        if year in boom_years:
            return boom_years[year]

        return 0.0  # Neutral period

    def _ml_prediction(
        self,
        mission_data: pd.DataFrame,
        mission: str,
        predict_years: list[int],
    ) -> list[dict[str, Any]]:
        """
        Generate predictions using ensemble ML models.

        Combines Linear Regression and Random Forest predictions using
        weighted averaging based on training R² scores.

        Args:
            mission_data: Feature-enhanced data for a single mission.
            mission: Mission name.
            predict_years: Years to predict.

        Returns:
            List of prediction dictionaries with Annee, Mission, Montant_Predit, Confiance.
        """
        # Prepare training data
        feature_cols = [
            "Annee",
            "Mission_Encoded",
            "Tendance",
            "Cycle_Economique",
            "Taux_Croissance",
        ]
        X = mission_data[feature_cols].copy()
        y = mission_data["Montant"].copy()

        # Sanitize inputs
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = pd.to_numeric(y, errors="coerce")
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Clip extreme values
        X = X.clip(MIN_MONETARY_VALUE, MAX_MONETARY_VALUE)
        y = np.clip(y, MIN_MONETARY_VALUE, MAX_MONETARY_VALUE)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float64))

        # Train ensemble of models
        models = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(
                n_estimators=ML_CONFIG["random_forest_estimators"],
                random_state=ML_CONFIG["random_seed"],
            ),
        }

        model_predictions: dict[str, list[float]] = {}
        model_scores: dict[str, float] = {}

        for name, model in models.items():
            try:
                # Check for constant target
                if np.std(y) == 0:
                    raise ValueError("Constant target - using fallback")

                model.fit(X_scaled, y)

                y_pred_train = model.predict(X_scaled)
                score = r2_score(y, y_pred_train)
                model_scores[name] = max(0, np.nan_to_num(score))

                # Generate predictions
                predictions = []
                for year in predict_years:
                    X_pred = np.array(
                        [
                            [
                                year,
                                mission_data["Mission_Encoded"].iloc[0],
                                year - mission_data["Annee"].min(),
                                self._get_economic_cycle(year),
                                mission_data["Taux_Croissance"].mean(),
                            ]
                        ],
                        dtype=float,
                    )

                    X_pred_scaled = scaler.transform(X_pred)
                    pred = float(model.predict(X_pred_scaled)[0])
                    predictions.append(pred)

                model_predictions[name] = predictions

            except Exception:
                # Fallback to simple trend
                model_predictions[name] = self._simple_trend_prediction(
                    mission_data, predict_years
                )
                model_scores[name] = 0.1

        # Compute weighted ensemble
        total_weight = sum(model_scores.values()) or 1
        weights = {name: score / total_weight for name, score in model_scores.items()}

        ensemble_predictions = []
        for i, year in enumerate(predict_years):
            weighted_pred = sum(
                weights[name] * model_predictions[name][i]
                for name in model_predictions
            )

            # Apply growth constraints
            last_value = mission_data["Montant"].iloc[-1]
            years_ahead = year - mission_data["Annee"].max()
            min_value = last_value * MIN_VALUE_RETENTION
            max_value = last_value * (MAX_ANNUAL_GROWTH_RATE**years_ahead)

            constrained_pred = max(min_value, min(max_value, weighted_pred))
            ensemble_predictions.append(constrained_pred)

        # Format results
        avg_score = sum(model_scores.values()) / len(model_scores)
        results = [
            {
                "Annee": year,
                "Mission": mission,
                "Montant_Predit": ensemble_predictions[i],
                "Confiance": min(1.0, avg_score),
            }
            for i, year in enumerate(predict_years)
        ]

        return results

    def _simple_linear_prediction(
        self,
        df: pd.DataFrame,
        mission: str,
        predict_years: list[int],
    ) -> list[dict[str, Any]]:
        """
        Generate predictions using simple linear regression.

        Used as a fallback for missions with limited historical data.

        Args:
            df: Full budget DataFrame.
            mission: Mission name.
            predict_years: Years to predict.

        Returns:
            List of prediction dictionaries.
        """
        mission_data = df[df["Mission"] == mission].sort_values("Annee")

        if len(mission_data) < 2:
            # Only one data point - assume constant
            last_value = (
                mission_data["Montant"].iloc[-1] if not mission_data.empty else 1.0
            )
            return [
                {
                    "Annee": year,
                    "Mission": mission,
                    "Montant_Predit": last_value,
                    "Confiance": 0.3,
                }
                for year in predict_years
            ]

        # Fit simple linear regression
        X = mission_data["Annee"].values.reshape(-1, 1)
        y = mission_data["Montant"].values

        model = LinearRegression()
        model.fit(X, y)

        # Calculate confidence from R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        confidence = max(0.2, min(0.8, r2))

        # Generate predictions
        results = []
        last_value = mission_data["Montant"].iloc[-1]
        last_year = mission_data["Annee"].max()

        for year in predict_years:
            pred = model.predict([[year]])[0]

            # Apply growth constraints
            years_ahead = year - last_year
            max_growth = ML_CONFIG["simple_linear_max_growth"]
            max_decline = ML_CONFIG["simple_linear_max_decline"]
            max_pred = last_value * (max_growth**years_ahead)
            min_pred = last_value * (max_decline**years_ahead)

            constrained_pred = max(min_pred, min(max_pred, pred))

            results.append(
                {
                    "Annee": year,
                    "Mission": mission,
                    "Montant_Predit": max(MIN_PREDICTION_VALUE, constrained_pred),
                    "Confiance": confidence,
                }
            )

        return results

    def _simple_trend_prediction(
        self,
        mission_data: pd.DataFrame,
        predict_years: list[int],
    ) -> list[float]:
        """
        Generate predictions using polynomial trend fitting.

        Used as a last-resort fallback when ML models fail.

        Args:
            mission_data: Data for a single mission.
            predict_years: Years to predict.

        Returns:
            List of predicted values (floats).
        """
        if len(mission_data) < 2:
            return [mission_data["Montant"].iloc[-1]] * len(predict_years)

        # Simple linear fit
        x = mission_data["Annee"].values
        y = mission_data["Montant"].values
        coefficients = np.polyfit(x, y, 1)

        predictions = []
        for year in predict_years:
            pred = coefficients[0] * year + coefficients[1]
            predictions.append(max(MIN_PREDICTION_VALUE, pred))

        return predictions

    def get_prediction_summary(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate summary statistics for predictions.

        Args:
            df: Historical budget data.
            predictions_df: Prediction results.

        Returns:
            Dictionary with summary statistics including growth projections.
        """
        if predictions_df is None or predictions_df.empty:
            return {}

        latest_year = df["Annee"].max()
        latest_total = df[df["Annee"] == latest_year]["Montant"].sum()

        pred_2030 = predictions_df[predictions_df["Annee"] == 2030][
            "Montant_Predit"
        ].sum()

        years_ahead = 2030 - latest_year
        total_growth = ((pred_2030 - latest_total) / latest_total) * 100
        annual_growth = total_growth / years_ahead if years_ahead > 0 else 0

        # Find fastest growing missions
        growth_by_mission = []
        for mission in predictions_df["Mission"].unique():
            hist_value = df[
                (df["Mission"] == mission) & (df["Annee"] == latest_year)
            ]["Montant"]
            pred_value = predictions_df[
                (predictions_df["Mission"] == mission)
                & (predictions_df["Annee"] == 2030)
            ]["Montant_Predit"]

            if not hist_value.empty and not pred_value.empty:
                hist_val = hist_value.iloc[0]
                pred_val = pred_value.iloc[0]
                if hist_val > 0:
                    mission_growth = ((pred_val - hist_val) / hist_val) * 100
                    growth_by_mission.append(
                        {"Mission": mission, "Croissance_Predite": mission_growth}
                    )

        growth_df = pd.DataFrame(growth_by_mission)
        top_growing = (
            growth_df.nlargest(3, "Croissance_Predite")
            if not growth_df.empty
            else pd.DataFrame()
        )

        return {
            "budget_actuel": latest_total,
            "budget_predit_2030": pred_2030,
            "croissance_totale_pct": total_growth,
            "croissance_annuelle_pct": annual_growth,
            "missions_croissance_forte": (
                top_growing.to_dict("records") if not top_growing.empty else []
            ),
            "nombre_missions_analysees": len(predictions_df["Mission"].unique()),
            "confiance_moyenne": (
                predictions_df["Confiance"].mean()
                if "Confiance" in predictions_df.columns
                else 0.5
            ),
        }
