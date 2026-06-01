# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

# Local application imports
from src.data_preparation import build_preprocessor


class ModelTraining:
    """
    A class used to train and evaluate classification models on hotel
    no-show data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training
        and evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelTraining class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters
            for model training and evaluation.
        """
        self.config = config

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into stratified training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        logging.info("Starting data splitting.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config["val_test_size"],
            random_state=42,
            stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config["val_size"],
            random_state=42,
            stratify=y_temp,
        )
        logging.info("Data split into train, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_pipeline(self, model, scale_numeric: bool) -> Pipeline:
        """
        Build a sklearn Pipeline pairing a preprocessor with a classifier.

        Args:
        -----
        model: An unfitted sklearn classifier instance.
        scale_numeric (bool): Whether to scale numerical features. Set False
            for tree-based models.

        Returns:
        --------
        sklearn.pipeline.Pipeline: The full preprocessing + model pipeline.
        """
        preprocessor = build_preprocessor(self.config, scale_numeric=scale_numeric)
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models with default
        hyperparameters. Includes a most-frequent DummyClassifier as the
        no-skill floor that all tuned models must beat.

        Args:
        -----
        X_train, y_train: Training features and target.
        X_val, y_val: Validation features and target.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple
            containing the trained pipelines and their validation metrics.
        """
        logging.info("Training and evaluating baseline models.")

        # (model_instance, scale_numeric) per model
        baseline_specs = {
            "dummy": (DummyClassifier(strategy="most_frequent"), False),
            "logistic_regression": (LogisticRegression(max_iter=1000), True),
            "knn": (KNeighborsClassifier(n_jobs=2), True),
            "random_forest": (
                RandomForestClassifier(random_state=42, n_jobs=2),
                False,
            ),
        }

        pipelines = {}
        metrics = {}

        for model_name, (model, scale_numeric) in baseline_specs.items():
            pipeline = self._build_pipeline(model, scale_numeric=scale_numeric)
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning via GridSearchCV for all classification
        models and evaluate them on the validation set.

        Args:
        -----
        X_train, y_train: Training features and target.
        X_val, y_val: Validation features and target.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple
            containing the tuned pipelines and their validation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        cv = self.config["cv"]
        scoring = self.config["scoring"]
        param_grids = self.config["param_grids"]

        # (model_instance, scale_numeric, grid_key) per model
        tuned_specs = {
            "logistic_regression_tuned": (
                LogisticRegression(max_iter=1000, solver="liblinear"),
                True,
                "logistic_regression",
            ),
            "knn_tuned": (KNeighborsClassifier(n_jobs=2), True, "knn"),
            "random_forest_tuned": (
                RandomForestClassifier(random_state=42, n_jobs=2),
                False,
                "random_forest",
            ),
        }

        tuned_models = {}
        tuned_metrics = {}

        for model_name, (model, scale_numeric, grid_key) in tuned_specs.items():
            pipeline = self._build_pipeline(model, scale_numeric=scale_numeric)
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grids[grid_key],
                cv=cv,
                scoring=scoring,
                n_jobs=2,
            )
            grid_search.fit(X_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            logging.info(
                f"{model_name} best params: {grid_search.best_params_}"
            )
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate the final selected model on the held-out test set.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test, y_test: Test features and target.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the test set metrics.
        """
        metrics = self._compute_metrics(model, X_test, y_test)
        logging.info(f"Final test metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.5f}")
        return metrics

    def _evaluate_model(
        self,
        model: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val, y_val: Validation features and target.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the validation metrics.
        """
        metrics = self._compute_metrics(model, X_val, y_val)
        logging.info(f"{model_name} validation metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.5f}")
        return metrics

    @staticmethod
    def _compute_metrics(
        model: Pipeline, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Compute classification metrics for a fitted model.

        ROC-AUC uses predicted probabilities for the positive class
        (predict_proba[:, 1]), not hard predictions. All other metrics use
        the hard predictions from predict(). zero_division=0 keeps precision
        well-defined for the dummy, which predicts no positives.

        Args:
        -----
        model (Pipeline): The fitted model pipeline.
        X (pd.DataFrame): The features.
        y (pd.Series): The true target labels.

        Returns:
        --------
        Dict[str, float]: A dictionary containing accuracy, precision, recall,
            F1, and ROC-AUC.
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
        }
