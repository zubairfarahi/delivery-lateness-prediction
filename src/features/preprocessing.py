"""
Preprocessing Module
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesses data for ML model
    Follows Dependency Inversion - can swap out different preprocessing strategies
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: str = "is_late_delivery",
    ):
        """
        Initialize preprocessor

        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            target_column: Target variable name
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create sklearn ColumnTransformer for preprocessing

        Returns:
            Configured ColumnTransformer
        """
        logger.info("Creating preprocessing pipeline")

        transformers = []

        # Numerical features: StandardScaler
        if self.numerical_features:
            transformers.append(("num", StandardScaler(), self.numerical_features))
            logger.info(
                f"Added StandardScaler for {len(self.numerical_features)} numerical features"
            )

        # Categorical features: OneHotEncoder
        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                )
            )
            logger.info(
                f"Added OneHotEncoder for {len(self.categorical_features)} categorical features"
            )

        # Create ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop columns not specified
            verbose_feature_names_out=False,
        )

        return self.preprocessor

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor on training data

        Args:
            X: Training features

        Returns:
            self
        """
        logger.info("Fitting preprocessor")

        if self.preprocessor is None:
            self.create_preprocessor()

        self.preprocessor.fit(X)

        # Get feature names after transformation
        self.feature_names_out = self.preprocessor.get_feature_names_out().tolist()
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_out)}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor

        Args:
            X: Features to transform

        Returns:
            Transformed numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        logger.info("Transforming data")
        X_transformed = self.preprocessor.transform(X)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        """Get transformed feature names"""
        return self.feature_names_out

    @staticmethod
    def split_features_target(
        df: pd.DataFrame, target_column: str = "is_late_delivery"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target

        Args:
            df: Input DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (features, target)
        """
        # Drop trip_id (identifier, not a feature)
        feature_cols = [col for col in df.columns if col not in [target_column, "trip_id"]]

        X = df[feature_cols]
        y = df[target_column]

        logger.info(f"Split into {X.shape[1]} features and 1 target")

        return X, y

    @staticmethod
    def train_test_split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = "is_late_delivery",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with stratification

        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            random_state: Random seed
            stratify_column: Column to stratify on

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")

        stratify = df[stratify_column] if stratify_column else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

        # Log class distribution
        if stratify_column:
            train_dist = train_df[stratify_column].value_counts(normalize=True)
            test_dist = test_df[stratify_column].value_counts(normalize=True)

            logger.info(f"Train class distribution: {train_dist.to_dict()}")
            logger.info(f"Test class distribution: {test_dist.to_dict()}")

        return train_df, test_df

    @staticmethod
    def create_cv_splits(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Create stratified K-Fold cross-validation splits

        Args:
            X: Features
            y: Target
            n_splits: Number of folds
            random_state: Random seed

        Returns:
            StratifiedKFold object
        """
        logger.info(f"Creating {n_splits}-fold stratified CV splits")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        return skf

    def get_numerical_features(self) -> List[str]:
        """Get list of numerical features"""
        return self.numerical_features

    def get_categorical_features(self) -> List[str]:
        """Get list of categorical features"""
        return self.categorical_features
