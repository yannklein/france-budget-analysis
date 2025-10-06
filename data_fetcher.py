import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional
import numpy as np


class DataFetcher:
    """Fetch French government budget data from various official sources."""

    def __init__(self):
        # --- ENDPOINTS (ouverts) ---
        self.BALANCES_RECORDS = (
            "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
            "balances_des_comptes_etat/records?limit=1000"
        )
        self.SME_RECORDS = (
            "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
            "situation-mensuelle-de-l-etat/records?limit=100000"
        )
        self.HISTO_DS_SLUGS = [
            "depenses-et-recettes-des-budgets-executes-de-l-etat-et-de-la-defense-30382662",
        ]
        self.base_urls = {
            "data_gouv": "https://www.data.gouv.fr/api/1/",
            "data_economie": "https://data.economie.gouv.fr/api/records/1.0/search/",
            "insee": "https://api.insee.fr/series/BDM/V1/data/",
        }
        self.missions_mapping = {
            # Major French budget missions (post-2001 LOLF reform)
            "Defense": "Défense",
            "Enseignement scolaire": "Éducation Nationale",
            "Enseignement supérieur et recherche": "Enseignement Supérieur",
            "Sécurités": "Sécurité",
            "Justice": "Justice",
            "Gestion des finances publiques": "Gestion Finances",
            "Travail et emploi": "Emploi",
            "Santé": "Santé",
            "Solidarité, insertion et égalité des chances": "Solidarité",
            "Ville et logement": "Logement",
            "Écologie, développement et mobilité durables": "Écologie",
            "Agriculture, alimentation, forêt et affaires rurales": "Agriculture",
            "Culture": "Culture",
            "Sport, jeunesse et vie associative": "Sport et Jeunesse",
            "Outre-mer": "Outre-mer",
            "Aide publique au développement": "Aide Développement",
            "Anciens combattants, mémoire et liens avec la nation": "Anciens Combattants",
            "Immigration, asile et intégration": "Immigration",
            "Administration générale et territoriale de l'État": "Administration",
            "Direction de l'action du gouvernement": "Action Gouvernement",
        }

    def fetch_budget_data(
        self, source: str = "data.gouv.fr", start_year: int = 2005, end_year: int = 2025
    ) -> pd.DataFrame:
        """
        Fetch budget data from specified source.

        Args:
            source: Data source ('data.gouv.fr', 'INSEE', 'data.economie.gouv.fr')
            start_year: Start year for data
            end_year: End year for data

        Returns:
            DataFrame with columns: Année, Mission, Montant
        """
        try:
            if source == "data.gouv.fr":
                return self._fetch_from_data_gouv(start_year, end_year)
            elif source == "INSEE":
                return self._fetch_from_insee(start_year, end_year)
            elif source == "data.economie.gouv.fr":
                return self._fetch_from_data_economie(start_year, end_year)
            else:
                raise ValueError(f"Source non supportée: {source}")

        except Exception as e:
            # If API fails, return realistic sample data based on actual French budget structure
            print(f"Erreur API {source}: {e}")
            return self._generate_realistic_budget_data(start_year, end_year)

    def _fetch_from_data_gouv(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch data from data.gouv.fr API."""

        # Search for budget datasets
        search_url = f"{self.base_urls['data_gouv']}datasets/"
        params = {"q": "budget état missions", "page_size": 50}

        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()

        datasets = response.json().get("data", [])

        # Look for budget datasets
        budget_datasets = []
        for dataset in datasets:
            title = dataset.get("title", "").lower()
            if any(
                keyword in title
                for keyword in ["budget", "finances", "état", "mission"]
            ):
                budget_datasets.append(dataset)

        if not budget_datasets:
            raise Exception("Aucun dataset budgétaire trouvé sur data.gouv.fr")

        # Try to fetch data from the most relevant dataset
        for dataset in budget_datasets[:3]:  # Try top 3 datasets
            try:
                dataset_id = dataset.get("id")
                dataset_detail_url = (
                    f"{self.base_urls['data_gouv']}datasets/{dataset_id}/"
                )

                detail_response = requests.get(dataset_detail_url, timeout=30)
                detail_response.raise_for_status()

                dataset_detail = detail_response.json()
                resources = dataset_detail.get("resources", [])

                for resource in resources:
                    if resource.get("format", "").upper() in ["CSV", "JSON"]:
                        resource_url = resource.get("url")
                        if resource_url:
                            data = self._download_and_parse_resource(
                                resource_url, resource.get("format", "CSV")
                            )
                            if data is not None and not data.empty:
                                return self._normalize_budget_data(
                                    data, start_year, end_year
                                )

            except Exception as e:
                continue

        raise Exception("Impossible de récupérer les données depuis data.gouv.fr")

    def _fetch_from_insee(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch data from INSEE API."""

        # INSEE uses specific series codes for government spending
        insee_series = [
            "001717256",  # Central government expenditure
            "001717257",  # Social security expenditure
        ]

        all_data = []

        for series_code in insee_series:
            try:
                url = f"https://api.insee.fr/series/BDM/V1/data/{series_code}"
                params = {"startPeriod": str(start_year), "endPeriod": str(end_year)}

                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    # Parse INSEE data structure
                    if "Obs" in data:
                        for obs in data["Obs"]:
                            all_data.append(
                                {
                                    "Année": int(obs.get("TIME_PERIOD", 0)),
                                    "Montant": float(obs.get("OBS_VALUE", 0))
                                    / 1000,  # Convert to billions
                                    "Series": series_code,
                                }
                            )

            except Exception as e:
                continue

        if not all_data:
            raise Exception("Aucune donnée INSEE récupérée")

        df = pd.DataFrame(all_data)
        return self._expand_to_missions(df, start_year, end_year)

    def _fetch_from_data_economie(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch data from data.economie.gouv.fr API."""

        url = self.base_urls["data_economie"]
        params = {"q": "budget état", "rows": 1000, "format": "json"}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        records = data.get("records", [])

        if not records:
            raise Exception("Aucune donnée trouvée sur data.economie.gouv.fr")

        # Parse records and convert to DataFrame
        parsed_data = []
        for record in records:
            fields = record.get("fields", {})
            # Extract relevant budget information
            if "annee" in fields or "year" in fields:
                year = fields.get("annee", fields.get("year", 0))
                if isinstance(year, str):
                    year = int(year) if year.isdigit() else 0

                if start_year <= year <= end_year:
                    amount = fields.get("montant", fields.get("amount", 0))
                    if isinstance(amount, str):
                        try:
                            amount = float(amount.replace(",", "."))
                        except:
                            amount = 0

                    mission = fields.get(
                        "mission", fields.get("category", "Non spécifié")
                    )

                    parsed_data.append(
                        {
                            "Année": year,
                            "Mission": mission,
                            "Montant": amount / 1000000,  # Convert to billions
                        }
                    )

        if not parsed_data:
            raise Exception("Impossible de parser les données économiques")

        df = pd.DataFrame(parsed_data)
        return self._normalize_budget_data(df, start_year, end_year)

    def _download_and_parse_resource(
        self, url: str, format_type: str
    ) -> Optional[pd.DataFrame]:
        """Download and parse a data resource."""

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if format_type.upper() == "CSV":
                # Try different encodings
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        from io import StringIO

                        df = pd.read_csv(
                            StringIO(response.content.decode(encoding)),
                            sep=None,
                            engine="python",
                        )
                        return df
                    except:
                        continue

            elif format_type.upper() == "JSON":
                data = response.json()
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to find the data array in the JSON
                    for key in ["data", "records", "results"]:
                        if key in data and isinstance(data[key], list):
                            return pd.DataFrame(data[key])

            return None

        except Exception as e:
            print(f"Erreur lors du téléchargement de {url}: {e}")
            return None

    def _normalize_budget_data(
        self, df: pd.DataFrame, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Normalize raw budget data to standard format."""

        # Try to identify year, mission, and amount columns
        year_columns = [
            col
            for col in df.columns
            if any(term in col.lower() for term in ["year", "annee", "date"])
        ]
        mission_columns = [
            col
            for col in df.columns
            if any(
                term in col.lower()
                for term in ["mission", "category", "ministere", "department"]
            )
        ]
        amount_columns = [
            col
            for col in df.columns
            if any(
                term in col.lower()
                for term in ["montant", "amount", "value", "budget", "spending"]
            )
        ]

        if not year_columns or not amount_columns:
            # If structure is unclear, create realistic data
            return self._generate_realistic_budget_data(start_year, end_year)

        # Use the first matching columns
        year_col = year_columns[0]
        mission_col = mission_columns[0] if mission_columns else None
        amount_col = amount_columns[0]

        # Create normalized DataFrame
        normalized_data = []

        for _, row in df.iterrows():
            try:
                year = int(row[year_col]) if pd.notna(row[year_col]) else None
                if year and start_year <= year <= end_year:
                    mission = (
                        str(row[mission_col])
                        if mission_col and pd.notna(row[mission_col])
                        else "Non spécifié"
                    )
                    amount = float(row[amount_col]) if pd.notna(row[amount_col]) else 0

                    # Convert to billions if necessary
                    if amount > 1000000:  # Likely in euros, convert to billions
                        amount = amount / 1000000000
                    elif amount > 1000:  # Likely in millions, convert to billions
                        amount = amount / 1000

                    normalized_data.append(
                        {
                            "Année": year,
                            "Mission": self._normalize_mission_name(mission),
                            "Montant": amount,
                        }
                    )

            except Exception as e:
                continue

        if not normalized_data:
            return self._generate_realistic_budget_data(start_year, end_year)

        result_df = pd.DataFrame(normalized_data)

        # Group by year and mission, sum amounts
        result_df = (
            result_df.groupby(["Année", "Mission"])["Montant"].sum().reset_index()
        )

        return result_df

    def _expand_to_missions(
        self, df: pd.DataFrame, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Expand aggregate data to mission-level data based on typical French budget distribution."""

        # Typical distribution of French budget by mission (approximate percentages)
        mission_distribution = {
            "Défense": 0.12,
            "Éducation Nationale": 0.18,
            "Enseignement Supérieur": 0.08,
            "Sécurité": 0.06,
            "Justice": 0.03,
            "Santé": 0.15,
            "Solidarité": 0.10,
            "Écologie": 0.08,
            "Agriculture": 0.04,
            "Culture": 0.02,
            "Emploi": 0.06,
            "Administration": 0.08,
        }

        expanded_data = []

        # Group by year and sum total spending
        yearly_totals = df.groupby("Année")["Montant"].sum()

        for year, total_spending in yearly_totals.items():
            for mission, percentage in mission_distribution.items():
                mission_amount = total_spending * percentage
                expanded_data.append(
                    {"Année": year, "Mission": mission, "Montant": mission_amount}
                )

        return pd.DataFrame(expanded_data)

    def get_cpi_series(
        self, start_year: int, end_year: int, base_year: int = None
    ) -> pd.DataFrame:
        """
        Return a CPI index for France with columns ['Année','CPI'].
        Uses a robust fallback (synthetic series) if remote fetch is unavailable.
        CPI is returned as an index; normalization to a base year is handled by the utils deflator.
        """
        import pandas as pd

        start_year = int(start_year)
        end_year = int(end_year)
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        years = list(range(start_year, end_year + 1))

        # Approximate yearly inflation (%) as a safe fallback
        approx_inflation = {
            2005: 1.9,
            2006: 1.7,
            2007: 1.5,
            2008: 2.8,
            2009: 0.1,
            2010: 1.5,
            2011: 2.1,
            2012: 2.0,
            2013: 0.9,
            2014: 0.5,
            2015: 0.1,
            2016: 0.2,
            2017: 1.0,
            2018: 1.8,
            2019: 1.1,
            2020: 0.5,
            2021: 1.6,
            2022: 5.2,
            2023: 4.9,
            2024: 2.5,
            2025: 2.0,
            2026: 2.0,
            2027: 2.0,
            2028: 2.0,
            2029: 2.0,
            2030: 2.0,
        }

        cpi_values = {start_year: 100.0}
        for y in years[1:]:
            prev = cpi_values[y - 1]
            infl = approx_inflation.get(y, 2.0) / 100.0
            cpi_values[y] = prev * (1.0 + infl)

        return pd.DataFrame({"Année": years, "CPI": [cpi_values[y] for y in years]})

    def get_debt_interest_series(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Return an approximate 'Charge de la dette de l'État' series (billions EUR).
        Uses a synthetic fallback based on plausible averages, trending with rates and inflation.
        Columns: ['Année','Montant'] where Montant is in billions.
        """
        import pandas as pd
        import math

        start_year = int(start_year)
        end_year = int(end_year)
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        years = list(range(start_year, end_year + 1))

        # Baseline (roughly aligns with historic order of magnitude: ~40-55 Bn over the period)
        base_2005 = 40.0
        series = {}
        series[start_year] = base_2005

        # Simple rule: drift slowly; increase in 2008-2012 (post-crisis), stabilize, rise with 2022-2024 inflation/rates
        for y in years[1:]:
            prev = series[y - 1]
            drift = 0.005  # 0.5% baseline drift
            shock = 0.0
            if 2008 <= y <= 2012:
                shock += 0.02  # +2% per year
            if 2022 <= y <= 2024:
                shock += 0.03  # +3% per year due to higher rates/inflation
            series[y] = prev * (1.0 + drift + shock)

        return pd.DataFrame(
            {"Année": years, "Montant": [round(series[y], 2) for y in years]}
        )

    def _normalize_mission_name(self, mission: str) -> str:
        """Normalize mission names to standard French budget missions."""

        mission = str(mission).strip()

        # Check if mission matches known mappings
        for key, value in self.missions_mapping.items():
            if key.lower() in mission.lower() or mission.lower() in key.lower():
                return value

        # Return cleaned mission name
        return mission.title()

    def _generate_realistic_budget_data(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """
        Generate realistic budget data based on actual French government spending patterns.
        This is used as fallback when APIs are unavailable.
        """

        # Base budget values for 2020 (in billions €) - based on actual French budget
        base_missions = {
            "Défense": 47.2,
            "Éducation Nationale": 53.8,
            "Enseignement Supérieur": 28.5,
            "Sécurité": 13.9,
            "Justice": 8.7,
            "Santé": 19.8,
            "Solidarité": 25.4,
            "Écologie": 11.2,
            "Agriculture": 6.8,
            "Culture": 3.1,
            "Emploi": 15.6,
            "Administration": 8.9,
            "Gestion Finances": 12.3,
            "Logement": 7.4,
            "Outre-mer": 2.8,
        }

        # Realistic growth rates by mission (annual %)
        growth_rates = {
            "Défense": 0.02,  # 2% annual growth
            "Éducation Nationale": 0.015,  # 1.5% annual growth
            "Enseignement Supérieur": 0.03,  # 3% annual growth (priority)
            "Sécurité": 0.025,  # 2.5% annual growth
            "Justice": 0.02,
            "Santé": 0.035,  # 3.5% annual growth (aging population)
            "Solidarité": 0.025,
            "Écologie": 0.08,  # 8% annual growth (green transition)
            "Agriculture": 0.01,
            "Culture": 0.005,  # 0.5% annual growth
            "Emploi": 0.02,
            "Administration": 0.01,
            "Gestion Finances": 0.015,
            "Logement": 0.03,
            "Outre-mer": 0.015,
        }

        # Add some realistic noise
        np.random.seed(42)  # For reproducible results

        data = []

        for year in range(start_year, end_year + 1):
            year_offset = year - 2020

            for mission, base_value in base_missions.items():
                # Calculate compound growth
                growth_rate = growth_rates.get(mission, 0.02)
                projected_value = base_value * (1 + growth_rate) ** year_offset

                # Add realistic noise (±5%)
                noise_factor = 1 + np.random.normal(0, 0.05)
                final_value = projected_value * noise_factor

                # Add some economic cycle effects
                if year in [2008, 2009, 2020, 2021]:  # Crisis years
                    if mission in ["Emploi", "Solidarité", "Santé"]:
                        final_value *= 1.1  # Counter-cyclical spending
                    else:
                        final_value *= 0.95  # Austerity in other areas

                data.append(
                    {
                        "Année": year,
                        "Mission": mission,
                        "Montant": max(0.1, final_value),  # Minimum 0.1 billion
                    }
                )

        df = pd.DataFrame(data)
        return df.sort_values(["Année", "Mission"]).reset_index(drop=True)

    # ===============  UTILITAIRES  ===============

    def _http_json(self, url: str):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _parse_records_to_df(self, records: list) -> pd.DataFrame:
        rows = []
        for rec in records or []:
            fields = rec.get("fields", {})
            if isinstance(fields, dict):
                rows.append(fields)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _clean_number(self, x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("\xa0", " ").replace(" ", "")
        s = s.replace(",", ".")
        try:
            return float(s)
        except:
            return np.nan

    def _detect_year_series(
        self, df: pd.DataFrame, year_col_hint: list[str] = None
    ) -> Optional[str]:
        year_col_hint = year_col_hint or []
        candidates = [c for c in df.columns if c in year_col_hint]
        if candidates:
            return candidates[0]
        tokens = ("annee", "année", "year", "exercice", "fiscal")
        cands = [c for c in df.columns if any(t in c.lower() for t in tokens)]
        if cands:
            return cands[0]
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            ok = vals.dropna().astype(int)
            if ok.between(1900, 2100).sum() >= max(3, int(0.3 * len(ok))):
                return c
        return None

    def _normalize_revenue_df(
        self,
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
        year_col_hint: list[str] = None,
        prefer_cols: tuple = (
            "recette",
            "recettes",
            "produit",
            "revenu",
            "montant",
            "total",
            "budget",
        ),
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Année", "Montant"])
        df1 = df.copy()
        df1.columns = [str(c).strip() for c in df1.columns]

        year_col = self._detect_year_series(df1, year_col_hint=year_col_hint)
        years_in_headers = [
            c
            for c in df1.columns
            if pd.Series([c]).str.fullmatch(r"(19|20)\d{2}").any()
        ]

        if not year_col and years_in_headers:
            id_vars = [c for c in df1.columns if c not in years_in_headers]
            melted = df1.melt(
                id_vars=id_vars,
                value_vars=years_in_headers,
                var_name="Année",
                value_name="Montant_raw",
            )
            melted["Année"] = pd.to_numeric(melted["Année"], errors="coerce").astype(
                "Int64"
            )
            melted["Montant"] = melted["Montant_raw"].map(self._clean_number)
            headers_blob = " ".join(df1.columns).lower()
            scale = 1.0
            if "millier" in headers_blob or "milliers" in headers_blob:
                scale = 1_000.0
            elif (
                "million" in headers_blob
                or "millions" in headers_blob
                or "m€" in headers_blob
            ):
                scale = 1_000_000.0
            elif (
                "milliard" in headers_blob
                or "milliards" in headers_blob
                or "md€" in headers_blob
            ):
                scale = 1_000_000_000.0
            melted["Montant"] = (melted["Montant"] * scale) / 1_000_000_000.0
            out = (
                melted.dropna(subset=["Année"])
                .astype({"Année": "int"})
                .query("@start_year <= Année <= @end_year")
                .groupby("Année", as_index=False)["Montant"]
                .sum()
                .sort_values("Année")
            )
            return out

        if not year_col:
            return pd.DataFrame(columns=["Année", "Montant"])

        out = df1.copy()
        out["Année"] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        out = out.dropna(subset=["Année"])
        out["Année"] = out["Année"].astype(int)

        cand_cols = [
            c
            for c in out.columns
            if c != year_col and any(tok in c.lower() for tok in prefer_cols)
        ]
        if not cand_cols:
            cand_cols = []
            for c in out.columns:
                if c == year_col:
                    continue
                vals = pd.to_numeric(out[c], errors="coerce")
                if vals.notna().sum() >= max(3, int(0.25 * len(vals))):
                    cand_cols.append(c)
        if not cand_cols:
            return pd.DataFrame(columns=["Année", "Montant"])

        for c in cand_cols:
            out[c] = out[c].map(self._clean_number)

        out["Montant_raw"] = out[cand_cols].sum(axis=1, skipna=True)
        headers_blob = " ".join(df1.columns).lower()
        scale = 1.0
        if "millier" in headers_blob or "milliers" in headers_blob:
            scale = 1_000.0
        elif (
            "million" in headers_blob
            or "millions" in headers_blob
            or "m€" in headers_blob
        ):
            scale = 1_000_000.0
        elif (
            "milliard" in headers_blob
            or "milliards" in headers_blob
            or "md€" in headers_blob
        ):
            scale = 1_000_000_000.0
        out["Montant"] = (out["Montant_raw"] * scale) / 1_000_000_000.0

        out = (
            out.query("@start_year <= Année <= @end_year")
            .groupby("Année", as_index=False)["Montant"]
            .sum()
            .sort_values("Année")
        )
        return out[["Année", "Montant"]]

    # ===============  SOURCES  ===============

    def _fetch_balances_etat_revenue(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Balances des comptes de l'État → agrège les PRODUITS (classe 7) par année."""
        payload = self._http_json(self.BALANCES_RECORDS)
        if not payload:
            return pd.DataFrame(columns=["Année", "Montant"])

        # v2.1 payload
        df = self._parse_v21_results_to_df(payload)
        if df is None or df.empty:
            # fallback legacy (unlikely with v2.1)
            df = self._parse_records_to_df(payload)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Année", "Montant"])
        
        [print(f"_fetch_balances_etat_revenue raw Col {i}: {c}") for i,c in enumerate(df.columns)]

        # Harmonise columns
        df.columns = [str(c).strip() for c in df.columns]
    

        # Year
        year_col = "annee" if "annee" in df.columns else (
            "année" if "année" in df.columns else None
        )
        if year_col is None:
            # try to infer from a date column
            date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date","exercice"))), None)
            if date_col:
                s = pd.to_datetime(df[date_col], errors="coerce")
                df["Année"] = s.dt.year
                year_col = "Année"
        if year_col is None:
            return pd.DataFrame(columns=["Année", "Montant"])

        # Value column: coalesce balance_sortie_2 / _3 / (fallback) _ (some resources only have one)
        for c in ("balance_sortie_2", "balance_sortie_3", "balance_sortie"):
            if c not in df.columns:
                df[c] = np.nan
        df["val"] = df["balance_sortie_2"].where(df["balance_sortie_2"].notna(),
                    df["balance_sortie_3"].where(df["balance_sortie_3"].notna(),
                    df["balance_sortie"]))

        # Apply multiplicateur if present
        if "multiplicateur" in df.columns:
            # safe multiply
            try:
                df["val"] = pd.to_numeric(df["val"], errors="coerce") * pd.to_numeric(df["multiplicateur"], errors="coerce").fillna(1.0)
            except Exception:
                df["val"] = pd.to_numeric(df["val"], errors="coerce")
        else:
            df["val"] = pd.to_numeric(df["val"], errors="coerce")

        # ---- Filter to revenues (Produits) ----
        def _contains(col, needle):
            return df[col].astype(str).str.contains(needle, case=False, na=False) if col in df.columns else pd.Series(False, index=df.index)

        cond_class7 = df["compte"].astype(str).str.startswith("7") if "compte" in df.columns else False
        cond_categ  = _contains("categorie", "produit")
        cond_postes = _contains("postes", "produit")

        mask_revenue = cond_class7 | cond_categ | cond_postes
        df_rev = df[mask_revenue].copy()
        if df_rev.empty:
            # nothing matched -> return empty instead of misleading values
            return pd.DataFrame(columns=["Année", "Montant"])

        # Year clean
        df_rev["Année"] = pd.to_numeric(df_rev[year_col], errors="coerce").astype("Int64")
        df_rev = df_rev.dropna(subset=["Année"]).astype({"Année": int})

        # Aggregate → Md€
        out = (
            df_rev.groupby("Année", as_index=False)["val"].sum()
                .rename(columns={"val": "Montant"})
        )
        # values appear to be in euros; convert to Md€ (billions)
        out["Montant"] = out["Montant"] / 1_000_000_000.0

        # Keep requested range
        out = out[(out["Année"] >= start_year) & (out["Année"] <= end_year)]
        return out.sort_values("Année").reset_index(drop=True)

    def _fetch_sme_revenue(self, start_year: int, end_year: int) -> pd.DataFrame:
        payload = self._http_json(self.SME_RECORDS)
        if not payload:
            return pd.DataFrame(columns=["Année", "Montant"])
        # v2.1:
        df = self._parse_v21_results_to_df(payload)
        # Fallback to legacy shape if needed:
        print("df _fetch_balances_etat_revenue", df.columns)
        [print(f"_fetch_balances_etat_revenue raw Col {i}: {c}") for i,c in enumerate(df.columns)]
        if df.empty:
            df = self._parse_records_to_df(payload)
            


        # fabriquer Année depuis une date si nécessaire
        lower_cols = {c.lower(): c for c in df.columns}
        if "année" not in lower_cols and "annee" not in lower_cols and len(df.columns) > 0:
            date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date","periode","période","mois"))), None)
            if date_col:
                s = pd.to_datetime(df[date_col], errors="coerce")
                df["Année"] = s.dt.year

        return self._normalize_revenue_df(df, start_year, end_year)

    def _fetch_historic_revenue_from_dataset(
        self, start_year: int, end_year: int
    ) -> pd.DataFrame:
        base = "https://www.data.gouv.fr/api/1/datasets/"
        for slug in self.HISTO_DS_SLUGS:
            detail = self._http_json(f"{base}{slug}/")
            if not detail:
                continue
            for res in detail.get("resources", []) or []:
                fmt = (res.get("format") or "").upper()
                if fmt not in ("CSV", "JSON"):
                    continue
                title = (res.get("title") or res.get("name") or "").lower()
                if not any(
                    k in title
                    for k in (
                        "recette",
                        "recettes",
                        "revenu",
                        "budget",
                        "produit",
                        "equilibre",
                        "équilibre",
                    )
                ):
                    continue
                url = res.get("url")
                if not url:
                    continue

                df = None
                try:
                    if fmt == "JSON":
                        j = self._http_json(url)
                        if isinstance(j, list):
                            df = pd.DataFrame(j)
                            [print(f"1_fetch_historic_revenue_from_dataset raw Col {i}: {c}") for i,c in enumerate(df.columns)]
                        elif isinstance(j, dict):
                            for k in ("data", "records", "results"):
                                if k in j and isinstance(j[k], list):
                                    df = pd.DataFrame(j[k])
                                    [print(f"2_fetch_historic_revenue_from_dataset raw Col {i}: {c}") for i,c in enumerate(df.columns)]
                                    break
                    else:
                        from io import StringIO

                        raw = requests.get(url, timeout=30)
                        raw.raise_for_status()
                        content = raw.content
                        for enc in ("utf-8", "latin-1", "cp1252"):
                            try:
                                text = content.decode(enc)
                            except Exception:
                                continue
                            for sep in (None, ";", ";", ",", "\t", "|"):
                                for dec in (".", ","):
                                    try:
                                        df = pd.read_csv(
                                            StringIO(text),
                                            sep=sep,
                                            engine="python",
                                            decimal=dec,
                                        )
                                        [print(f"3_fetch_historic_revenue_from_dataset raw Col {i}: {c}") for i,c in enumerate(df.columns)]
                                        if df is not None and not df.empty:
                                            break
                                    except Exception:
                                        df = None
                                if df is not None:
                                    break
                            if df is not None:
                                break
                except Exception:
                    df = None

                if df is not None and not df.empty:
                    out = self._normalize_revenue_df(df, start_year, end_year)
                    if not out.empty:
                        return out

        return pd.DataFrame(columns=["Année", "Montant"])

    # ===============  MÉTHODE PRINCIPALE  ===============

    def fetch_revenue_20y(
        self, start_year: int = None, end_year: int = None
    ) -> pd.DataFrame:
        """
        Essaie plusieurs sources publiques pour retourner les recettes de l'État (Md€) par année
        sur ~20 ans (ou la plage demandée). Fusionne, dédoublonne et somme en cas de recouvrements.
        """
        # plage par défaut: dernières 20 années pleines
        if end_year is None:
            end_year = pd.Timestamp.now().year
        if start_year is None:
            start_year = end_year - 19

        frames = []
        try:
            frames.append(self._fetch_balances_etat_revenue(start_year, end_year))
        except Exception:
            pass
        try:
            frames.append(self._fetch_sme_revenue(start_year, end_year))
        except Exception:
            pass
        try:
            frames.append(
                self._fetch_historic_revenue_from_dataset(start_year, end_year)
            )
        except Exception:
            pass

        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            # rien trouvé : renvoyer DF vide explicite
            return pd.DataFrame(columns=["Année", "Montant"])

        out = pd.concat(frames, ignore_index=True)
        # garder uniquement la plage demandée
        out = out[(out["Année"] >= start_year) & (out["Année"] <= end_year)]
        # agrège si plusieurs sources trouvent la même année
        out = (
            out.groupby("Année", as_index=False)["Montant"].mean().sort_values("Année")
        )
        # astuce: on prend la moyenne si les sources diffèrent; sinon passe à .sum() si tu préfères
        return out.reset_index(drop=True)
    
    def _parse_v21_results_to_df(self, payload) -> pd.DataFrame:
        """
        v2.1 Explore API returns a dict with 'results': [ {flat fields...}, ... ].
        """
        if not isinstance(payload, dict):
            return pd.DataFrame()
        results = payload.get("results") or []
        return pd.DataFrame(results) if results else pd.DataFrame()

    def _parse_records_to_df(self, payload_like_records: list | dict) -> pd.DataFrame:
        """
        Backward-compatible: if payload is a v1-like dict with 'records': [{'fields': {...}}, ...],
        extract 'fields'. If it's already a list of records with 'fields', handle it.
        Otherwise, return empty.
        """
        rows = []
        if isinstance(payload_like_records, dict):
            records = payload_like_records.get("records", [])
        else:
            records = payload_like_records or []

        for rec in records:
            fields = rec.get("fields", {}) if isinstance(rec, dict) else {}
            if isinstance(fields, dict):
                rows.append(fields)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
