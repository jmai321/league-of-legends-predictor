"""
LOL XGBoost models: lineup model + full-game realtime model + mid-game realtime models (10 / 15 / 20 / 25).

Usage:
1. Make sure the current directory contains:
   - data_clean/game_result.csv
   - data_clean/realtime.csv

2. Install dependencies:
   pip install xgboost scikit-learn pandas matplotlib joblib

3. Run:
   python model.py

This script will:
- Train and validate:
  * Lineup model (game_result)
  * Full-game realtime model (realtime global stats)
  * Mid-game realtime models at 10 / 15 / 20 / 25 minutes
    (based on *at10 / *at15 / *at20 / *at25 feature columns)
- Print validation AUC / Accuracy for each model
- Print top 20 feature importances
- Print example win probabilities for a sample game (Blue / Red, raw + normalized)
- Save ROC curves under model/ directory
- Save confusion matrices under model/ directory
- Save trained models under model/ directory
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)


# =========================
# Plot utilities
# =========================
def plot_roc(y_true, y_score, title: str, save_path: str) -> None:
    """
    Plot and save a ROC curve given true labels and predicted scores.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_score : array-like
        Predicted probability scores.
    title : str
        Title of the ROC figure.
    save_path : str
        Path to save the ROC PNG image.
    """
    assert y_true is not None and y_score is not None, "Input arrays cannot be None"
    assert isinstance(title, str), "title must be a string"
    assert isinstance(save_path, str), "save_path must be a string"
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ROC] {title} saved to {save_path}, AUC = {auc_val:.4f}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    save_path: str,
    class_names: Tuple[str, str] = ("0", "1"),
) -> None:
    """
    Plot and save a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels (0/1).
    title : str
        Title of the confusion matrix figure.
    save_path : str
        Path to save the confusion matrix image.
    class_names : tuple of str, optional
        Names of the classes in order (negative, positive).
    """
    assert y_true is not None and y_pred is not None, "Input arrays cannot be None"
    assert isinstance(title, str), "title must be a string"
    assert isinstance(save_path, str), "save_path must be a string"
    assert isinstance(class_names, tuple) and len(class_names) == 2, "class_names must be tuple of 2 strings"
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"[CM] {title}")
    print(cm)
    print(f"      TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[CM] {title} saved to {save_path}")


# =========================
# Lineup model: based on game_result.csv
# =========================
class LineupXGBModel:
    """
    XGBoost model for predicting Blue side win probability from lineups.

    The model is built on champion combinations:
    One row = one game (blue 5 champions + red 5 champions + win).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize the lineup model and configure the XGBoost classifier.

        Parameters
        ----------
        n_estimators : int
            Number of boosting trees.
        max_depth : int
            Maximum depth of each tree.
        learning_rate : float
            Learning rate (eta) for boosting.
        subsample : float
            Subsample ratio of the training instances.
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree.
        random_state : int
            Random seed.
        """
        self.categorical_cols = [
            "bot_blue",
            "jng_blue",
            "mid_blue",
            "sup_blue",
            "top_blue",
            "bot_red",
            "jng_red",
            "mid_red",
            "sup_red",
            "top_red",
        ]
        self.target_col = "win"

        self.model: Pipeline = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.fitted: bool = False

    def train_val_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and validation sets.

        Parameters
        ----------
        df : pd.DataFrame
            Lineup dataset containing champion columns and target.
        test_size : float, optional
            Proportion of the data to use as validation, by default 0.2.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
            X_train, X_valid, y_train, y_valid.
        """
        X = df[self.categorical_cols]
        y = df[self.target_col]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_valid, y_train, y_valid

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the lineup model on the given training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target labels.
        """
        assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame"
        assert isinstance(y_train, pd.Series), "y_train must be a pandas Series"
        assert len(X_train) == len(y_train), "X_train and y_train must have same length"
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict Blue side win probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with lineup categorical columns.

        Returns
        -------
        np.ndarray
            Probability of Blue side winning for each row.
        """
        assert self.fitted, "Model must be fit before calling predict_proba."
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        return self.model.predict_proba(X)[:, 1]

    def predict_game(self, row: pd.Series) -> dict:
        """
        Predict win probabilities for a single game from its lineup.

        Parameters
        ----------
        row : pd.Series
            A single row containing champion columns for both sides.

        Returns
        -------
        dict
            Dictionary with keys:
                p_blue : float
                p_red  : float
        """
        assert self.fitted, "Model must be fit before calling predict_game."
        assert isinstance(row, pd.Series), "row must be a pandas Series"

        X = row[self.categorical_cols].to_frame().T
        p_blue = float(self.predict_proba(X)[0])
        p_red = 1.0 - p_blue
        return {"p_blue": p_blue, "p_red": p_red}

    def feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """
        Get global feature importances from the lineup model.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to return, by default 30.

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature names and importances.
        """
        assert self.fitted, "Model must be fit before calling feature_importance."

        onehot: OneHotEncoder = self.model.named_steps["onehot"]
        xgb: XGBClassifier = self.model.named_steps["xgb"]

        feature_names = onehot.get_feature_names_out(self.categorical_cols)
        importances = xgb.feature_importances_

        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        if top_n is not None:
            fi = fi.head(top_n)

        return fi

    def save(self, path: str) -> None:
        """
        Save the lineup model pipeline to disk (OneHotEncoder + XGBClassifier).

        Parameters
        ----------
        path : str
            File path to save the joblib model.
        """
        assert self.fitted, "Cannot save an unfitted model."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[Lineup] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LineupXGBModel":
        """
        Load a pre-trained lineup model from disk.

        Parameters
        ----------
        path : str
            File path where the joblib model is stored.

        Returns
        -------
        LineupXGBModel
            Loaded lineup model instance.
        """
        model = joblib.load(path)
        obj = cls()
        obj.model = model
        obj.fitted = True
        print(f"[Lineup] Model loaded from {path}")
        return obj


def tune_and_train_lineup_model(
    lineup_df: pd.DataFrame,
    model_dir: str = "model/",
) -> LineupXGBModel:
    """
    Tune and train the lineup model using RandomizedSearchCV.

    The search is performed on the training split (80% of data) with
    3-fold cross validation, optimizing ROC-AUC. The best estimator is
    then evaluated on the held-out validation split, ROC curve and
    confusion matrix are plotted, and the final model is saved to disk.

    Parameters
    ----------
    lineup_df : pd.DataFrame
        Dataset containing lineup columns and 'win' target.
    model_dir : str, optional
        Directory to store ROC / CM figures and trained model, by default "model/".

    Returns
    -------
    LineupXGBModel
        Lineup model instance with the best estimator fitted.
    """
    os.makedirs(model_dir, exist_ok=True)

    lineup_model = LineupXGBModel()
    X_train, X_valid, y_train, y_valid = lineup_model.train_val_split(lineup_df)

    # Hyperparameter search space (on the XGB inside the pipeline)
    param_distributions = {
        "xgb__n_estimators": [200, 300, 400, 500],
        "xgb__max_depth": [3, 4, 5, 6, 7],
        "xgb__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
        "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__gamma": [0.0, 0.1, 0.2],
    }

    print("\n[Lineup] Starting hyperparameter tuning with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=lineup_model.model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train)

    print("\n[Lineup] Best CV AUC: {:.4f}".format(search.best_score_))
    print("[Lineup] Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_pipeline: Pipeline = search.best_estimator_
    lineup_model.model = best_pipeline
    lineup_model.fitted = True

    # Evaluate on validation set
    y_valid_pred_proba = lineup_model.predict_proba(X_valid)
    y_valid_pred_label = (y_valid_pred_proba >= 0.5).astype(int)
    auc_val = roc_auc_score(y_valid, y_valid_pred_proba)
    acc_val = accuracy_score(y_valid, y_valid_pred_label)
    print(
        "[Lineup] Validation AUC with best params: {:.4f}, Accuracy: {:.4f}".format(
            auc_val, acc_val
        )
    )

    # ROC curve
    roc_path = os.path.join(model_dir, "lineup_roc_tuned.png")
    plot_roc(
        y_true=y_valid,
        y_score=y_valid_pred_proba,
        title="Lineup Model ROC (tuned, Blue win)",
        save_path=roc_path,
    )

    # Confusion matrix
    cm_path = os.path.join(model_dir, "lineup_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_valid,
        y_pred=y_valid_pred_label,
        title="Lineup Model Confusion Matrix (Blue win)",
        save_path=cm_path,
        class_names=("Lose", "Win"),
    )

    # Feature importance
    print("\n[Lineup] Top 20 feature importance (tuned model):")
    print(lineup_model.feature_importance(top_n=20))

    # Save model pipeline
    model_path = os.path.join(model_dir, "lineup_model.joblib")
    lineup_model.save(model_path)

    return lineup_model


# =========================
# Full-game realtime model: based on realtime.csv
# =========================
class RealtimeXGBModel:
    """
    XGBoost model for predicting team win probability from full-game stats.

    One row = one team in one game.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize the realtime model and configure the XGBoost classifier.

        Parameters
        ----------
        n_estimators : int
            Number of boosting trees.
        max_depth : int
            Maximum depth of each tree.
        learning_rate : float
            Learning rate (eta) for boosting.
        subsample : float
            Subsample ratio of the training instances.
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree.
        random_state : int
            Random seed.
        """
        self.target_col = "result"
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
        )
        self.feature_cols: List[str] = []
        self.fitted: bool = False

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process dataframe for training/prediction.

        Steps:
        - Encode side as 0 (Blue) / 1 (Red).
        - Drop non-feature columns.
        - Convert all non-target columns to numeric and fill missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Original realtime dataframe.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe prepared for modeling.
        """
        df = df.copy()

        # encode side as numeric
        df["side"] = df["side"].map({"Blue": 0, "Red": 1})

        # drop non-numerical / unneeded fields
        drop_cols = [
            "gameid",
            "datacompleteness",
            "teamname",
            "teamid",
            "top",
            "jng",
            "mid",
            "bot",
            "sup",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

        # cast to numeric
        for col in df.columns:
            if col == "result":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # fill missing values
        df = df.fillna(0)

        return df

    def train_val_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split realtime data into training and validation sets.

        Parameters
        ----------
        df : pd.DataFrame
            Full realtime dataframe.
        test_size : float, optional
            Proportion of the data to use as validation, by default 0.2.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
            X_train, X_valid, y_train, y_valid.
        """
        df_prep = self._prepare_dataframe(df)

        y = df_prep[self.target_col]
        X = df_prep.drop(columns=[self.target_col])

        self.feature_cols = X.columns.tolist()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        return X_train, X_valid, y_train, y_valid

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the realtime XGBoost model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training labels.
        """
        assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame"
        assert isinstance(y_train, pd.Series), "y_train must be a pandas Series"
        assert len(X_train) == len(y_train), "X_train and y_train must have same length"
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities for each team row.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as used in training).

        Returns
        -------
        np.ndarray
            Array of predicted win probabilities.
        """
        assert self.fitted, "Model must be fit before calling predict_proba."
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"

        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(0)

        return self.model.predict_proba(X)[:, 1]

    def _predict_team_proba(self, row: pd.Series) -> float:
        """
        Predict win probability for a single team row.

        Parameters
        ----------
        row : pd.Series
            One row of realtime data, already cleaned.

        Returns
        -------
        float
            Predicted probability that this team wins.
        """
        X = row[self.feature_cols].to_frame().T
        p_win = float(self.predict_proba(X)[0])
        return p_win

    def predict_game(self, raw_df: pd.DataFrame, gameid) -> dict:
        """
        Predict Blue and Red win probabilities for a specific game.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Original realtime dataframe with both teams.
        gameid :
            Game identifier matching the 'gameid' column.

        Returns
        -------
        dict
            Dictionary containing:
                gameid, blue_team, red_team,
                p_blue_raw, p_red_raw,
                p_blue_norm, p_red_norm.
        """
        assert self.fitted, "Model must be fit before calling predict_game."

        df = raw_df.copy()
        game_mask = df["gameid"] == gameid
        game_df = df[game_mask]

        if game_df.shape[0] != 2:
            raise ValueError(
                f"gameid={gameid} does not have exactly two rows (found {game_df.shape[0]})."
            )

        blue_row_raw = game_df[game_df["side"] == "Blue"].iloc[0]
        red_row_raw = game_df[game_df["side"] == "Red"].iloc[0]

        blue_teamname = blue_row_raw.get("teamname", "Blue")
        red_teamname = red_row_raw.get("teamname", "Red")

        df_clean = self._prepare_dataframe(df)
        game_df_clean = df_clean[game_mask]

        blue_row = game_df_clean.loc[blue_row_raw.name]
        red_row = game_df_clean.loc[red_row_raw.name]

        p_blue_raw = self._predict_team_proba(blue_row)
        p_red_raw = self._predict_team_proba(red_row)

        def _logit(p):
            eps = 1e-6
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        l_blue = _logit(p_blue_raw)
        l_red = _logit(p_red_raw)
        exp_blue = np.exp(l_blue)
        exp_red = np.exp(l_red)
        p_blue_norm = float(exp_blue / (exp_blue + exp_red))
        p_red_norm = float(exp_red / (exp_blue + exp_red))

        return {
            "gameid": gameid,
            "blue_team": blue_teamname,
            "red_team": red_teamname,
            "p_blue_raw": p_blue_raw,
            "p_red_raw": p_red_raw,
            "p_blue_norm": p_blue_norm,
            "p_red_norm": p_red_norm,
        }

    def feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """
        Get global feature importances for the realtime model.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to return, by default 30.

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature names and importances.
        """
        assert self.fitted, "Model must be fit before calling feature_importance."
        assert self.feature_cols is not None

        importances = self.model.feature_importances_
        fi = (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        if top_n is not None:
            fi = fi.head(top_n)

        return fi

    def save(self, path: str) -> None:
        """
        Save the realtime model to disk (model + feature columns).

        Parameters
        ----------
        path : str
            File path to save the joblib payload.
        """
        assert self.fitted, "Cannot save an unfitted model."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"model": self.model, "feature_cols": self.feature_cols}
        joblib.dump(payload, path)
        print(f"[Realtime] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RealtimeXGBModel":
        """
        Load a pre-trained realtime model from disk.

        Parameters
        ----------
        path : str
            File path where the joblib payload is stored.

        Returns
        -------
        RealtimeXGBModel
            Loaded realtime model instance.
        """
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.feature_cols = payload["feature_cols"]
        obj.fitted = True
        print(f"[Realtime] Model loaded from {path}")
        return obj


# =========================
# Mid-game models (10 / 15 / 20 / 25 minutes)
# =========================
def train_midgame_models(
    realtime_df: pd.DataFrame,
    model_dir: str = "model/",
    time_points: List[int] = None,
) -> None:
    """
    Train mid-game realtime models at given time points (minutes).

    For each time point T, this function:
    - selects columns containing 'atT',
    - builds a team-level dataset with side + mid-game stats,
    - splits data by gameid using GroupShuffleSplit,
    - trains an XGBoost classifier,
    - evaluates AUC and Accuracy,
    - saves ROC plots and confusion matrices,
    - saves joblib model,
    - prints an example prediction for a sample game.

    Parameters
    ----------
    realtime_df : pd.DataFrame
        Full realtime dataset.
    model_dir : str, optional
        Directory to save mid-game models and plots, by default "model/".
    time_points : List[int], optional
        List of mid-game minutes to train, by default [10, 15, 20, 25].
    """
    if time_points is None:
        time_points = [10, 15, 20, 25]

    os.makedirs(model_dir, exist_ok=True)

    for T in time_points:
        print(f"\n===== Training {T}-minute mid-game model =====")

        mid_cols = [c for c in realtime_df.columns if f"at{T}" in c]
        base_cols = ["gameid", "side", "result"]

        missing = [c for c in base_cols if c not in realtime_df.columns]
        if missing:
            raise ValueError(f"[Mid{T}] Missing required columns in realtime: {missing}")

        if not mid_cols:
            print(f"[Mid{T}] No feature column containing 'at{T}' was found, skipping.")
            continue

        use_cols = base_cols + mid_cols
        df_mid = realtime_df[use_cols].copy()

        # encode side
        df_mid["side"] = df_mid["side"].map({"Blue": 0, "Red": 1})

        # numeric conversion
        for col in mid_cols + ["side"]:
            df_mid[col] = pd.to_numeric(df_mid[col], errors="coerce")
        df_mid[mid_cols + ["side"]] = df_mid[mid_cols + ["side"]].fillna(0)

        X_all = df_mid[["side"] + mid_cols]
        y_all = df_mid["result"]
        groups = df_mid["gameid"]

        # Group-wise split by gameid
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(gss.split(X_all, y_all, groups))

        X_train, X_valid = X_all.iloc[train_idx], X_all.iloc[valid_idx]
        y_train, y_valid = y_all.iloc[train_idx], y_all.iloc[valid_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        y_valid_pred = (y_valid_proba >= 0.5).astype(int)

        auc_val = roc_auc_score(y_valid, y_valid_proba)
        acc_val = accuracy_score(y_valid, y_valid_pred)
        print(f"[Mid{T}] Validation AUC: {auc_val:.4f}, Accuracy: {acc_val:.4f}")

        # ROC curve
        roc_path = os.path.join(model_dir, f"realtime_mid{T}_roc.png")
        plot_roc(
            y_true=y_valid,
            y_score=y_valid_proba,
            title=f"Realtime Mid {T}min ROC",
            save_path=roc_path,
        )

        # Confusion matrix
        cm_path = os.path.join(model_dir, f"realtime_mid{T}_confusion_matrix.png")
        plot_confusion_matrix(
            y_true=y_valid,
            y_pred=y_valid_pred,
            title=f"Realtime Mid {T}min Confusion Matrix",
            save_path=cm_path,
            class_names=("Lose", "Win"),
        )

        # Feature importances
        importances = model.feature_importances_
        feature_names = X_all.columns.tolist()
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        print(f"[Mid{T}] Top 20 feature importance:")
        print(fi.head(20))

        # Save joblib model
        payload = {"model": model, "feature_cols": feature_names}
        model_path = os.path.join(model_dir, f"realtime_mid{T}_model.joblib")
        joblib.dump(payload, model_path)
        print(f"[Mid{T}] Model saved to {model_path}")

        # Sanity check: reload & re-evaluate
        loaded_payload = joblib.load(model_path)
        loaded_model = loaded_payload["model"]
        y_valid_proba_loaded = loaded_model.predict_proba(X_valid)[:, 1]
        auc_val_loaded = roc_auc_score(y_valid, y_valid_proba_loaded)
        acc_val_loaded = accuracy_score(
            y_valid, (y_valid_proba_loaded >= 0.5).astype(int)
        )
        print(
            f"[Mid{T}] Validation after reload - AUC: {auc_val_loaded:.4f}, "
            f"Accuracy: {acc_val_loaded:.4f}"
        )

        # Example per-game prediction
        valid_gameids = (
            realtime_df.groupby("gameid")
            .filter(lambda x: set(x["side"]) == {"Blue", "Red"})
            ["gameid"]
            .unique()
        )
        if len(valid_gameids) == 0:
            print(
                f"[Mid{T}] No valid game with both Blue and Red sides found for example output."
            )
            continue

        example_gameid = valid_gameids[0]
        game_raw = realtime_df[realtime_df["gameid"] == example_gameid]

        blue_raw = game_raw[game_raw["side"] == "Blue"].iloc[0]
        red_raw = game_raw[game_raw["side"] == "Red"].iloc[0]
        blue_team = blue_raw.get("teamname", "Blue")
        red_team = red_raw.get("teamname", "Red")

        game_mid = df_mid[df_mid["gameid"] == example_gameid]
        if game_mid.shape[0] != 2:
            print(
                f"[Mid{T}] Example gameid={example_gameid} does not have exactly two rows "
                f"in mid-game dataframe, skipping example output."
            )
            continue

        blue_mid = game_mid[game_mid["side"] == 0].iloc[0]
        red_mid = game_mid[game_mid["side"] == 1].iloc[0]

        X_blue = blue_mid[["side"] + mid_cols].to_frame().T
        X_red = red_mid[["side"] + mid_cols].to_frame().T

        for col in X_blue.columns:
            X_blue[col] = pd.to_numeric(X_blue[col], errors="coerce")
        for col in X_red.columns:
            X_red[col] = pd.to_numeric(X_red[col], errors="coerce")
        X_blue = X_blue.fillna(0)
        X_red = X_red.fillna(0)

        p_blue_raw = float(model.predict_proba(X_blue)[:, 1][0])
        p_red_raw = float(model.predict_proba(X_red)[:, 1][0])

        def _logit(p):
            eps = 1e-6
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))

        l_blue = _logit(p_blue_raw)
        l_red = _logit(p_red_raw)
        exp_blue = np.exp(l_blue)
        exp_red = np.exp(l_red)
        p_blue_norm = float(exp_blue / (exp_blue + exp_red))
        p_red_norm = float(exp_red / (exp_blue + exp_red))

        print(f"\n[Mid{T}] Example gameid: {example_gameid}")
        print(f"[Mid{T}] Team Blue: {blue_team}")
        print(f"[Mid{T}] Team Red : {red_team}")
        print(
            f"[Mid{T}] Raw P(Blue win): {p_blue_raw:.3f}, "
            f"Raw P(Red win): {p_red_raw:.3f}"
        )
        print(
            f"[Mid{T}] Norm P(Blue win): {p_blue_norm:.3f}, "
            f"Norm P(Red win): {p_red_norm:.3f}"
        )


# =========================
# Main script
# =========================
def main() -> None:
    """
    Entry point for training all LOL XGBoost models.

    This function:
    - loads cleaned datasets,
    - trains and evaluates the lineup model (with tuning),
    - trains and evaluates the full-game realtime model,
    - prints example predictions and feature importances,
    - trains mid-game models (10 / 15 / 20 / 25 minutes).
    """
    data_path = "data_clean/"
    model_dir = "model/"

    realtime_df = pd.read_csv(os.path.join(data_path, "realtime.csv"))
    lineup_df = pd.read_csv(os.path.join(data_path, "game_result.csv"))

    print("realtime shape:", realtime_df.shape)
    print("game_result shape:", lineup_df.shape)

    # 1) Lineup model with tuning + ROC + CM
    print(
        "\n===== Training Lineup Model (LineupXGBModel, with hyperparameter tuning) ====="
    )
    lineup_model = tune_and_train_lineup_model(
        lineup_df=lineup_df,
        model_dir=model_dir,
    )

    # 2) Full-game realtime model
    print("\n===== Training Full-game Realtime Model (RealtimeXGBModel) =====")
    realtime_model = RealtimeXGBModel()

    X_train_r, X_valid_r, y_train_r, y_valid_r = realtime_model.train_val_split(
        realtime_df
    )
    realtime_model.fit(X_train_r, y_train_r)

    y_valid_pred_r = realtime_model.predict_proba(X_valid_r)
    y_valid_label_r = (y_valid_pred_r >= 0.5).astype(int)
    auc_r = roc_auc_score(y_valid_r, y_valid_pred_r)
    acc_r = accuracy_score(y_valid_r, y_valid_label_r)
    print(f"[Realtime] Validation AUC: {auc_r:.4f}, Accuracy: {acc_r:.4f}")

    # ROC
    plot_roc(
        y_true=y_valid_r,
        y_score=y_valid_pred_r,
        title="Realtime Model ROC (Team win)",
        save_path=os.path.join(model_dir, "realtime_roc.png"),
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true=y_valid_r,
        y_pred=y_valid_label_r,
        title="Realtime Model Confusion Matrix (Team win)",
        save_path=os.path.join(model_dir, "realtime_confusion_matrix.png"),
        class_names=("Lose", "Win"),
    )

    # Save realtime model
    realtime_model.save(os.path.join(model_dir, "realtime_model.joblib"))

    # Example full-game prediction
    example_gameid = (
        realtime_df.groupby("gameid")
        .filter(lambda x: set(x["side"]) == {"Blue", "Red"})
        ["gameid"]
        .iloc[0]
    )
    rt_pred = realtime_model.predict_game(realtime_df, example_gameid)
    print("\n[Realtime] Example gameid:", example_gameid)
    print("[Realtime] Team Blue:", rt_pred["blue_team"])
    print("[Realtime] Team Red :", rt_pred["red_team"])
    print(
        "[Realtime] Raw P(Blue win): {:.3f}, Raw P(Red win): {:.3f}".format(
            rt_pred["p_blue_raw"], rt_pred["p_red_raw"]
        )
    )
    print(
        "[Realtime] Norm P(Blue win): {:.3f}, Norm P(Red win): {:.3f}".format(
            rt_pred["p_blue_norm"], rt_pred["p_red_norm"]
        )
    )

    print("\n[Realtime] Top 20 feature importance:")
    print(realtime_model.feature_importance(top_n=20))

    # 3) Mid-game models: 10 / 15 / 20 / 25
    train_midgame_models(realtime_df, model_dir=model_dir, time_points=[10, 15, 20, 25])


if __name__ == "__main__":
    main()
