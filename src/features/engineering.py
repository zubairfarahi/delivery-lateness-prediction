"""
Feature Engineering Module
"""

from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Creates new features from raw delivery data
    Follows Open/Closed Principle - can add new features without modifying existing
    """

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit feature engineer (for consistency with sklearn API)

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        logger.info("Fitting feature engineer")
        # No fitting required for these features, but keeping for extensibility
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw data

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features added
        """
        logger.info("Engineering features")

        df = df.copy()

        # 1. Duration-based features
        df = self._create_duration_features(df)

        # 2. Time-based features
        df = self._create_time_features(df)

        # 3. Load & efficiency features
        df = self._create_efficiency_features(df)

        # 4. Interaction features
        df = self._create_interaction_features(df)

        # 5. Categorical aggregations
        df = self._create_categorical_features(df)

        logger.info(f"Created {len(self.feature_names)} new features")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)

    def _create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create duration-based features"""

        # Difference between actual and estimated
        df["duration_diff"] = df["actual_duration_min"] - df["estimated_duration_min"]

        # Ratio (how much longer/shorter than expected)
        df["duration_ratio"] = df["actual_duration_min"] / df["estimated_duration_min"]

        # Boolean: was delivery ahead of schedule?
        df["is_ahead_of_schedule"] = (df["duration_diff"] < 0).astype(int)

        # Absolute percentage difference
        df["duration_pct_diff"] = abs(df["duration_diff"]) / df["estimated_duration_min"]

        # Minutes per km (speed proxy)
        df["min_per_km"] = df["actual_duration_min"] / df["distance_km"]

        self.feature_names.extend(
            [
                "duration_diff",
                "duration_ratio",
                "is_ahead_of_schedule",
                "duration_pct_diff",
                "min_per_km",
            ]
        )

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        # Rush hour (7-9 AM, 4-6 PM)
        df["is_rush_hour"] = (
            ((df["departure_hour"] >= 7) & (df["departure_hour"] <= 9))
            | ((df["departure_hour"] >= 16) & (df["departure_hour"] <= 18))
        ).astype(int)

        # Weekend
        df["is_weekend"] = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)

        # Time of day categories
        df["time_of_day"] = pd.cut(
            df["departure_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        )

        # Day type: weekday vs weekend
        df["day_type"] = df["day_of_week"].apply(
            lambda x: "weekend" if x in ["Sat", "Sun"] else "weekday"
        )

        self.feature_names.extend(["is_rush_hour", "is_weekend", "time_of_day", "day_type"])

        return df

    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create load and efficiency features"""

        # Load per km
        df["load_per_km"] = df["load_weight_kg"] / df["distance_km"]

        # Stops per km (delivery density)
        df["stops_per_km"] = df["num_stops"] / df["distance_km"]

        # Average distance per stop
        df["avg_distance_per_stop"] = df["distance_km"] / df["num_stops"]

        # Load weight categories
        df["load_category"] = pd.cut(
            df["load_weight_kg"],
            bins=[0, 500, 1000, 1500, 2000],
            labels=["light", "medium", "heavy", "very_heavy"],
        )

        # Distance categories
        df["distance_category"] = pd.cut(
            df["distance_km"],
            bins=[0, 15, 30, 50],
            labels=["short", "medium", "long"],
        )

        # Stops categories
        df["stops_category"] = pd.cut(
            df["num_stops"],
            bins=[0, 5, 10, 15, 25],
            labels=["few", "moderate", "many", "very_many"],
        )

        self.feature_names.extend(
            [
                "load_per_km",
                "stops_per_km",
                "avg_distance_per_stop",
                "load_category",
                "distance_category",
                "stops_category",
            ]
        )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features (medium complexity)"""

        # Bad weather in rural areas (challenging conditions)
        df["bad_weather_rural"] = (
            df["weather_condition"].isin(["snow", "rain"]) & (df["region"] == "rural")
        ).astype(int)

        # Heavy load with old vehicle
        median_load = df["load_weight_kg"].median()
        median_age = df["vehicle_age_years"].median()

        df["heavy_load_old_vehicle"] = (
            (df["load_weight_kg"] > median_load) & (df["vehicle_age_years"] > median_age)
        ).astype(int)

        # Many stops in urban (potential traffic delays)
        median_stops = df["num_stops"].median()
        df["many_stops_urban"] = (
            (df["num_stops"] > median_stops) & (df["region"] == "urban")
        ).astype(int)

        # Long distance with many stops (complex route)
        df["complex_route"] = (
            (df["distance_km"] > df["distance_km"].median()) & (df["num_stops"] > median_stops)
        ).astype(int)

        # Bad weather during rush hour
        df["bad_weather_rush_hour"] = (
            df["weather_condition"].isin(["snow", "rain"]) & (df["is_rush_hour"] == 1)
        ).astype(int)

        self.feature_names.extend(
            [
                "bad_weather_rural",
                "heavy_load_old_vehicle",
                "many_stops_urban",
                "complex_route",
                "bad_weather_rush_hour",
            ]
        )

        return df

    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from categorical combinations"""

        # Vehicle-region combination (e.g., truck in urban)
        df["vehicle_region"] = df["vehicle_type"] + "_" + df["region"]

        # Weather-region combination
        df["weather_region"] = df["weather_condition"] + "_" + df["region"]

        # Vehicle capacity indicator (larger vehicles = higher capacity)
        vehicle_capacity_map = {
            "van": 1,
            "truck_small": 2,
            "truck_large": 3,
        }
        df["vehicle_capacity"] = df["vehicle_type"].map(vehicle_capacity_map)

        # Load efficiency (load relative to vehicle capacity)
        df["load_efficiency"] = df["load_weight_kg"] / (
            df["vehicle_capacity"] * 500
        )  # Assuming base capacity

        self.feature_names.extend(
            ["vehicle_region", "weather_region", "vehicle_capacity", "load_efficiency"]
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of created feature names"""
        return self.feature_names
