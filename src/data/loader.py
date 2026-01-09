"""
Data Loading Module
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders (Interface Segregation)"""

    @abstractmethod
    def load(self, path: Path) -> pd.DataFrame:
        """Load data from source"""
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Save data to destination"""
        pass


class CSVDataLoader(BaseDataLoader):
    """Concrete implementation for CSV files"""

    def load(self, path: Path) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {path}")

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        return df

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to CSV file"""
        logger.info(f"Saving data to {path}")

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")


class DataLoader:
    """
    Main DataLoader class (Dependency Inversion - depends on abstraction)
    Provides high-level interface for loading delivery data
    """

    def __init__(self, loader: Optional[BaseDataLoader] = None):
        """
        Initialize with a specific loader implementation

        Args:
            loader: BaseDataLoader implementation (defaults to CSV)
        """
        self.loader = loader or CSVDataLoader()

    def load_raw_data(self, path: Path) -> pd.DataFrame:
        """Load raw delivery trips data"""
        return self.loader.load(path)

    def load_train_test_data(
        self, train_path: Path, test_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-split train and test data"""
        logger.info("Loading train and test datasets")

        train_df = self.loader.load(train_path)
        test_df = self.loader.load(test_path)

        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

        return train_df, test_df

    def save_train_test_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """Save train and test splits"""
        logger.info("Saving train and test datasets")

        self.loader.save(train_df, output_dir / "train.csv")
        self.loader.save(test_df, output_dir / "test.csv")

        logger.info(f"Saved splits to {output_dir}")
