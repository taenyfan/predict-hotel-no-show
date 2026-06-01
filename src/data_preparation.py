# Standard library imports
import logging
from typing import Any, Dict

# Related third-party imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def month_to_num(month_series: pd.Series) -> pd.Series:
    """
    Map month names ('January'...'December') to numbers 1-12.

    Args:
    -----
    month_series (pd.Series): A Series of month-name strings.

    Returns:
    --------
    pd.Series: A Series of month numbers (1-12), NaN for unrecognised names.
    """
    month_map = {m: i for i, m in enumerate([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'], start=1)}
    return month_series.map(month_map)


def cyclical_encode(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode month-name columns as sin/cos pairs.

    Months are cyclical (December is adjacent to January), so each month
    column is mapped to a point on a circle via sine and cosine. Each input
    column 'col' produces '{col}_sin' and '{col}_cos'.

    Args:
    -----
    X (pd.DataFrame): DataFrame of month-name columns.

    Returns:
    --------
    pd.DataFrame: DataFrame of sin/cos columns, aligned to X's index.
    """
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        m = month_to_num(X[col])
        out[f"{col}_sin"] = np.sin(2 * np.pi * m / 12)
        out[f"{col}_cos"] = np.cos(2 * np.pi * m / 12)
    return out


def cyclical_names(transformer, input_features):
    """
    Output feature names for cyclical_encode, in the same order the columns
    are created (one '_sin'/'_cos' pair per input column). Passed to
    FunctionTransformer(feature_names_out=...) so get_feature_names_out works.

    Args:
    -----
    transformer: The FunctionTransformer instance (unused, required by sklearn).
    input_features: The names of the input columns.

    Returns:
    --------
    np.ndarray: The output feature names.
    """
    names = []
    for col in input_features:
        names += [f"{col}_sin", f"{col}_cos"]
    return np.array(names)


def build_preprocessor(
    config: Dict[str, Any], scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Builds a preprocessor for transforming numerical, nominal, and cyclical
    features.

    Args:
    -----
    config (Dict[str, Any]): Configuration dictionary containing the feature
        lists.
    scale_numeric (bool): If True, numerical features are standardised with
        StandardScaler. If False, they are passed through unchanged. Set to
        False for tree-based models, which are invariant to feature scaling.

    Returns:
    --------
    sklearn.compose.ColumnTransformer: A ColumnTransformer for preprocessing.
    """
    numerical_transformer = StandardScaler() if scale_numeric else "passthrough"

    nominal_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    cyclical_transformer = Pipeline(
        steps=[
            (
                "cyclical",
                FunctionTransformer(
                    cyclical_encode, feature_names_out=cyclical_names
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, config["numerical_features"]),
            ("nom", nominal_transformer, config["nominal_features"]),
            ("cyc", cyclical_transformer, config["cyclical_features"]),
        ],
        remainder="drop",
        n_jobs=2,
    )
    return preprocessor


class DataPreparation:
    """
    A class used to clean hotel no-show classification data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters
            for data cleaning.
        """
        self.config = config

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several cleaning, imputation,
        and feature-engineering steps.

        Steps:
          1. Drop the fully-null row and reset the index.
          2. Standardise casing in 'arrival_month'.
          3. Fix negative 'checkout_day' values via absolute value.
          4. Parse 'num_adults' (words mixed with digits) to integers.
          5. Convert 'price' from 'SGD$/USD$' strings to numeric SGD.
          6. Impute missing 'price' from the (branch, room) median.
          7. Impute missing 'room' from branch + price band.
          8. Engineer 'lead_time' (months between booking and arrival).
          9. Drop low/no-signal columns and the row identifier.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df = df.copy()

        # 1. Drop the fully-null row (all columns except booking_id are NaN)
        null_row = df.drop(columns="booking_id").isnull().all(axis=1)
        df = df[~null_row].reset_index(drop=True)

        # 2. Standardise month casing
        df["arrival_month"] = df["arrival_month"].str.title()

        # 3. Fix negative checkout days
        df["checkout_day"] = df["checkout_day"].abs()

        # 4. Parse num_adults (words -> digits)
        df["num_adults"] = df["num_adults"].replace({"one": 1, "two": 2}).astype(int)

        # 5. Standardise price to SGD
        df["price"] = self._convert_price_to_sgd(
            df["price"], self.config["usd_to_sgd"]
        )

        # 6. Impute missing price from (branch, room) median
        df = self._impute_price(df)

        # 7. Impute missing room from branch + price band
        df = self._impute_room(df)

        # 8. Engineer lead_time (needs booking_month, so before the drop)
        df["lead_time"] = self._compute_lead_time(df)

        # 9. Drop low/no-signal columns and the row identifier
        df = df.drop(columns=self.config["drop_features"])

        logging.info("Data cleaning completed.")
        return df

    @staticmethod
    def _convert_price_to_sgd(
        price: pd.Series, usd_to_sgd: float = 1.37
    ) -> pd.Series:
        """
        Convert a price column of 'SGD$ x' / 'USD$ x' strings to numeric SGD.

        USD rows are scaled by 'usd_to_sgd'; SGD rows are left unchanged.
        Non-parseable entries become NaN.

        Args:
        -----
        price (pd.Series): The raw price column of currency strings.
        usd_to_sgd (float): The USD-to-SGD exchange rate.

        Returns:
        --------
        pd.Series: Numeric prices in SGD.
        """
        currency = price.str.extract(r"^([A-Z]+)")[0]
        amount = pd.to_numeric(
            price.str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
        )
        return amount.where(currency != "USD", amount * usd_to_sgd)

    @staticmethod
    def _impute_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing price with the median price of its (branch, room) group.

        Args:
        -----
        df (pd.DataFrame): The DataFrame with a numeric 'price' column.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing prices imputed.
        """
        df = df.copy()
        group_median = df.groupby(["branch", "room"])["price"].transform("median")
        df["price"] = df["price"].fillna(group_median)
        return df

    @staticmethod
    def _impute_room(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing room from branch + price band; assign 'Unknown' where
        ambiguous.

        A (branch, room) price band is the 1st-99th percentile range computed
        from rows where room is known. A missing-room row is assigned a room
        only if its price falls in exactly one band within its branch;
        otherwise 'Unknown'.

        Args:
        -----
        df (pd.DataFrame): The DataFrame with a numeric 'price' column.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing rooms imputed.
        """
        df = df.copy()

        # Build price bands per (branch, room) from rows where room is KNOWN
        known = df.dropna(subset=["room"])
        bands = (
            known.groupby(["branch", "room"])["price"]
            .quantile([0.01, 0.99])
            .unstack()
        )
        bands.columns = ["lo", "hi"]

        # For one row, find which room(s) its price falls into, within its branch
        def lookup_room(branch, price):
            if pd.isna(price):
                return "Unknown"
            candidates = [
                room
                for (b, room), band in bands.iterrows()
                if b == branch and band["lo"] <= price <= band["hi"]
            ]
            return candidates[0] if len(candidates) == 1 else "Unknown"

        # Apply only to the missing-room rows
        missing = df["room"].isna()
        df.loc[missing, "room"] = df.loc[missing].apply(
            lambda r: lookup_room(r["branch"], r["price"]), axis=1
        )
        return df

    @staticmethod
    def _compute_lead_time(df: pd.DataFrame) -> pd.Series:
        """
        Months between booking and arrival (0-11, year-wrap handled).

        If arrival precedes booking within the year (e.g. booked December,
        arrive February), the booking was the previous year, so 12 is added.

        Args:
        -----
        df (pd.DataFrame): The DataFrame with 'arrival_month' and
            'booking_month' columns.

        Returns:
        --------
        pd.Series: The lead time in months (0-11).
        """
        lead = month_to_num(df["arrival_month"]) - month_to_num(df["booking_month"])
        return lead.where(lead >= 0, lead + 12)
