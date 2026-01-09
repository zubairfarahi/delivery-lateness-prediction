"""
Data Validation Module using Great Expectations
"""

from typing import Dict, List

import pandas as pd
from great_expectations.core import ExpectationConfiguration, ExpectationSuite
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import EphemeralDataContext
from great_expectations.dataset import PandasDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates delivery trips data using Great Expectations"""

    def __init__(self):
        """Initialize validator with expected schema"""
        self.expected_columns = [
            "trip_id",
            "distance_km",
            "estimated_duration_min",
            "actual_duration_min",
            "num_stops",
            "vehicle_type",
            "vehicle_age_years",
            "load_weight_kg",
            "departure_hour",
            "day_of_week",
            "weather_condition",
            "region",
            "is_late_delivery",
        ]

        self.categorical_columns = [
            "vehicle_type",
            "day_of_week",
            "weather_condition",
            "region",
        ]

        self.numerical_columns = [
            "distance_km",
            "estimated_duration_min",
            "actual_duration_min",
            "num_stops",
            "vehicle_age_years",
            "load_weight_kg",
            "departure_hour",
        ]

    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate DataFrame against expectations

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting data validation")

        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Check 1: Column presence
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing columns: {missing_cols}")
            return validation_results

        # Check 2: No null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            validation_results["is_valid"] = False
            validation_results["errors"].append(
                f"Null values found: {null_counts[null_counts > 0].to_dict()}"
            )

        # Check 3: No duplicate trip_ids
        if df["trip_id"].duplicated().any():
            validation_results["is_valid"] = False
            validation_results["errors"].append(
                f"Duplicate trip_ids found: {df['trip_id'].duplicated().sum()}"
            )

        # Check 4: Value ranges
        range_checks = {
            "distance_km": (1, 50),
            "num_stops": (1, 25),
            "vehicle_age_years": (1, 14),
            "load_weight_kg": (50, 2000),
            "departure_hour": (0, 23),
            "is_late_delivery": (0, 1),
        }

        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    validation_results["warnings"].append(
                        f"{col}: {out_of_range.sum()} values out of range [{min_val}, {max_val}]"
                    )

        # Check 5: Categorical values
        expected_values = {
            "vehicle_type": ["van", "truck_small", "truck_large"],
            "weather_condition": ["clear", "rain", "snow", "windy"],
            "region": ["urban", "suburban", "rural"],
            "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        }

        for col, valid_values in expected_values.items():
            if col in df.columns:
                invalid = ~df[col].isin(valid_values)
                if invalid.any():
                    validation_results["warnings"].append(
                        f"{col}: {invalid.sum()} invalid values found: {df.loc[invalid, col].unique()}"
                    )

        # Check 6: Class distribution (for target variable)
        if "is_late_delivery" in df.columns:
            class_dist = df["is_late_delivery"].value_counts(normalize=True)
            validation_results["stats"]["class_distribution"] = class_dist.to_dict()

            # Warn if severely imbalanced
            min_class_ratio = class_dist.min()
            if min_class_ratio < 0.1:
                validation_results["warnings"].append(
                    f"Severe class imbalance detected: {min_class_ratio:.2%} minority class"
                )

        # Check 7: Data types
        expected_dtypes = {
            "trip_id": "int64",
            "num_stops": "int64",
            "vehicle_age_years": "int64",
            "departure_hour": "int64",
            "is_late_delivery": "int64",
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                validation_results["warnings"].append(
                    f"{col}: expected dtype {expected_dtype}, got {df[col].dtype}"
                )

        # Log results
        if validation_results["is_valid"]:
            logger.info("✓ Data validation passed")
        else:
            logger.error(
                f"✗ Data validation failed with {len(validation_results['errors'])} errors"
            )

        if validation_results["warnings"]:
            logger.warning(f"⚠ {len(validation_results['warnings'])} warnings during validation")

        return validation_results

    def validate_with_great_expectations(self, df: pd.DataFrame) -> bool:
        """
        Comprehensive validation using Great Expectations

        Args:
            df: DataFrame to validate

        Returns:
            True if all expectations pass, False otherwise
        """
        logger.info("Running Great Expectations validation")

        # Create ephemeral data context
        context = EphemeralDataContext()

        # Create expectation suite
        suite = ExpectationSuite(expectation_suite_name="delivery_trips_suite")

        # Add expectations
        expectations = [
            # Column expectations
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": self.expected_columns},
            ),
            # Null checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "trip_id"},
            ),
            # Uniqueness
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "trip_id"},
            ),
            # Range checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "distance_km", "min_value": 1, "max_value": 50},
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "departure_hour", "min_value": 0, "max_value": 23},
            ),
            # Categorical checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "vehicle_type",
                    "value_set": ["van", "truck_small", "truck_large"],
                },
            ),
        ]

        for exp in expectations:
            suite.add_expectation(exp)

        # Add suite to context
        context.add_expectation_suite(expectation_suite=suite)

        # Create validator
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="delivery_trips",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )

        # Validate
        try:
            # Add pandas datasource
            context.sources.add_pandas(name="pandas_datasource")

            # Get validator
            validator = context.get_validator(
                batch_request=batch_request, expectation_suite_name=suite.expectation_suite_name
            )

            # Run validation
            results = validator.validate()

            success = results.success
            logger.info(f"Great Expectations validation: {'PASSED' if success else 'FAILED'}")

            return success

        except Exception as e:
            logger.error(f"Great Expectations validation error: {e}")
            return False
