"""
Data fetching module for French government budget data.

This module handles all data retrieval from official French government APIs,
including budget data, CPI inflation indices, and debt interest calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from config import (
    API_ENDPOINTS,
    API_TIMEOUT,
    DEBT_CONFIG,
    DEFAULT_INFLATION_RATE,
    INFLATION_RATES,
    MAX_MONETARY_VALUE,
    MIN_DATA_YEAR,
    MIN_MONETARY_VALUE,
)


class DataFetcher:
    """
    Fetches French government budget data from official APIs.

    This class provides methods to retrieve budget data from data.economie.gouv.fr,
    as well as synthetic debt interest and CPI inflation data for analysis.

    Attributes:
        COMPTES: Dictionary mapping account codes to their descriptions.

    Example:
        >>> fetcher = DataFetcher()
        >>> df = fetcher.fetch_budget_data(start_year=2020, end_year=2024)
        >>> print(df.head())
    """

    def __init__(self) -> None:
        """Initialize the DataFetcher with account mappings."""
        self._load_account_mappings()

    def _load_account_mappings(self) -> None:
        """Load account code mappings from JSON file."""
        account_file = Path(__file__).parent / "account_name.json"
        try:
            with open(account_file, encoding="utf-8") as f:
                self.COMPTES: dict[str, str] = json.load(f)
        except FileNotFoundError:
            self.COMPTES = {}

    def fetch_budget_data(
        self,
        source: str = "data.economie.gouv.fr",
        start_year: int = 2015,
        end_year: int = 2025,
        base_compte: str = "",
        acc_level_range: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch budget data from the specified source.

        Args:
            source: Data source identifier (currently only 'data.economie.gouv.fr').
            start_year: Start year for data retrieval.
            end_year: End year for data retrieval.
            base_compte: Base account code to filter by (empty for all).
            acc_level_range: Account hierarchy level (1-3).

        Returns:
            DataFrame with columns: [Annee, Mission, Montant]
            - Annee: Year (integer)
            - Mission: Budget mission/account name (string)
            - Montant: Amount in billions EUR (float)

        Raises:
            Exception: If API request fails and no fallback is available.
        """
        try:
            return self._fetch_from_data_economie(
                start_year, end_year, base_compte, acc_level_range
            )
        except Exception as e:
            raise RuntimeError(f"API request failed for {source}: {e}") from e

    def _http_json(self, url: str) -> Optional[dict[str, Any]]:
        """
        Make an HTTP GET request and return JSON response.

        Args:
            url: The URL to fetch.

        Returns:
            Parsed JSON response as dictionary, or None on failure.
        """
        try:
            response = requests.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            return None

    def _parse_results_to_df(self, payload: Optional[dict[str, Any]]) -> pd.DataFrame:
        """
        Parse API v2.1 response into a DataFrame.

        Args:
            payload: JSON response from the API.

        Returns:
            DataFrame from the 'results' key, or empty DataFrame on failure.
        """
        if not isinstance(payload, dict):
            return pd.DataFrame()
        results = payload.get("results", [])
        return pd.DataFrame(results) if results else pd.DataFrame()

    def _fetch_balances_etat(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """
        Fetch state account balances from the API.

        Args:
            start_year: Start year for filtering.
            end_year: End year for filtering.

        Returns:
            DataFrame with harmonized columns including 'Annee' and 'Montant'.
        """
        payload = self._http_json(API_ENDPOINTS["balances"])
        df = self._parse_results_to_df(payload)

        if df.empty:
            return pd.DataFrame(columns=["Annee", "Montant"])

        # Harmonize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Find and process year column
        year_col = "annee" if "annee" in df.columns else "annee"
        if year_col in df.columns:
            df["Annee"] = pd.to_datetime(
                df[year_col], errors="coerce"
            ).dt.year.astype("Int64")

        return df.sort_values("Annee").reset_index(drop=True)

    def _fetch_from_data_economie(
        self,
        start_year: int,
        end_year: int,
        base_compte: str,
        acc_level_range: int,
    ) -> pd.DataFrame:
        """
        Fetch and process budget data from data.economie.gouv.fr.

        Args:
            start_year: Start year (minimum: MIN_DATA_YEAR).
            end_year: End year.
            base_compte: Base account code filter.
            acc_level_range: Account detail level.

        Returns:
            Processed DataFrame with [Annee, Mission, Montant].
        """
        # Apply year constraints
        if end_year is None:
            end_year = pd.Timestamp.now().year
        if start_year is None:
            start_year = end_year - 19
        start_year = max(start_year, MIN_DATA_YEAR)

        # Fetch data
        df = self._fetch_balances_etat(start_year, end_year)
        if df.empty:
            return pd.DataFrame(columns=["Annee", "Mission", "Montant"])

        # Clean and process
        df.columns = df.columns.map(str)
        df = df.loc[:, ~df.columns.duplicated()]

        # Process year column
        df["Annee"] = pd.to_datetime(
            df["annee"], errors="coerce"
        ).dt.year.astype("Int64")

        # Convert montant to billions EUR
        df["montant"] = (
            pd.to_numeric(df["montant"], errors="coerce").fillna(0.0) / 1_000_000_000
        )

        # Process account codes
        df["compte"] = df["compte"].astype(str).str.strip("0")

        # Apply account hierarchy filtering
        if base_compte == "":
            df["compte"] = df["compte"].str[:acc_level_range]
        else:
            df = df[df["compte"].str.startswith(base_compte)]
            df["compte"] = df["compte"].str[: len(base_compte) + acc_level_range]

        # Map account codes to mission names
        df["Mission"] = df["compte"].map(
            lambda x: self.COMPTES.get(x, f"Compte inconnu: {x}")
        )

        # Select and aggregate
        df = df[["Annee", "Mission", "montant"]].dropna(subset=["Annee"])
        df = df.groupby(["Annee", "Mission"], as_index=False)["montant"].sum()
        df.rename(columns={"montant": "Montant"}, inplace=True)

        # Final type conversions and filtering
        df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").astype("Int64")
        df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce").fillna(0.0)
        df = df.query("@start_year <= Annee <= @end_year")

        # Sanitize values
        df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["Montant"] = np.clip(df["Montant"], MIN_MONETARY_VALUE, MAX_MONETARY_VALUE)

        return df

    def fetch_revenue_20y(
        self,
        start_year: int = 2015,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """
        Fetch state revenue data for the specified period.

        This method retrieves revenue data from account class 7 (products/revenues)
        from the French government accounting system.

        Args:
            start_year: Start year for data retrieval.
            end_year: End year for data retrieval.

        Returns:
            DataFrame with columns: [Annee, Postes, Montant]
        """
        # Fetch revenue data (class 7 accounts)
        df = self.fetch_budget_data(
            start_year=start_year,
            end_year=end_year,
            base_compte="7",
            acc_level_range=1,
        )

        if df.empty:
            return pd.DataFrame(columns=["Annee", "Postes", "Montant"])

        # Rename Mission to Postes for revenue
        df = df.rename(columns={"Mission": "Postes"})

        return df

    def get_debt_interest_series(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """
        Generate synthetic debt interest series.

        Creates an approximate 'Charge de la dette de l'Etat' series based on
        historical patterns and economic cycle adjustments.

        Args:
            start_year: Start year for the series.
            end_year: End year for the series.

        Returns:
            DataFrame with columns: [Annee, Montant] where Montant is in billions EUR.
        """
        start_year = int(start_year)
        end_year = int(end_year)
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        years = list(range(start_year, end_year + 1))
        series: dict[int, float] = {start_year: DEBT_CONFIG["base_value_billions"]}

        for year in years[1:]:
            prev = series[year - 1]
            drift = DEBT_CONFIG["baseline_drift"]
            shock = 0.0

            # Apply crisis shocks
            if 2008 <= year <= 2012:
                shock += DEBT_CONFIG["crisis_shock_2008_2012"]
            if 2022 <= year <= 2024:
                shock += DEBT_CONFIG["crisis_shock_2022_2024"]

            series[year] = prev * (1.0 + drift + shock)

        return pd.DataFrame({
            "Annee": years,
            "Montant": [round(series[y], 2) for y in years],
        })

    def get_cpi_series(
        self,
        start_year: int,
        end_year: int,
        base_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate CPI (Consumer Price Index) series for France.

        Creates an inflation index based on historical and projected inflation rates.

        Args:
            start_year: Start year for the series.
            end_year: End year for the series.
            base_year: Base year for normalization (handled by deflator).

        Returns:
            DataFrame with columns: [Annee, CPI] where CPI is an index value.
        """
        start_year = int(start_year)
        end_year = int(end_year)
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        years = list(range(start_year, end_year + 1))
        cpi_values: dict[int, float] = {start_year: 100.0}

        for year in years[1:]:
            prev = cpi_values[year - 1]
            inflation_rate = INFLATION_RATES.get(year, DEFAULT_INFLATION_RATE) / 100.0
            cpi_values[year] = prev * (1.0 + inflation_rate)

        return pd.DataFrame({
            "Annee": years,
            "CPI": [cpi_values[y] for y in years],
        })


if __name__ == "__main__":
    # Simple test
    fetcher = DataFetcher()
    df = fetcher.fetch_budget_data(
        start_year=2015, end_year=2024, acc_level_range=1, base_compte=""
    )
    print(f"Fetched {len(df)} records")
    print(df.head())
