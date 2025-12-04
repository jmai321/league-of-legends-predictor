from typing import List, Dict, Any, Optional, Union
import os

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb  # 用于构造 DMatrix + pred_contribs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_PATH = "model/"


# =========================
# Pydantic request models
# =========================
class LineupItem(BaseModel):
    """
    Request schema for a single game lineup.

    Fields represent champions for both sides and each position.
    """

    bot_blue: str
    jng_blue: str
    mid_blue: str
    sup_blue: str
    top_blue: str
    bot_red: str
    jng_red: str
    mid_red: str
    sup_red: str
    top_red: str


class RealtimeRow(BaseModel):
    """
    Request schema for a single team row of realtime data.

    At minimum it must contain gameid and side, and any other numeric
    columns (gold, kills, at10/15/20/25 features, etc.) will be passed
    through to the model.
    """

    gameid: str
    side: str  # "Blue" or "Red"
    teamname: Optional[str] = None

    class Config:
        extra = "allow"  # allow arbitrary other fields (gold, kills, etc.)


class RealtimeGame(BaseModel):
    """
    Request schema for a full game consisting of two rows (Blue and Red).
    """

    rows: List[RealtimeRow] = Field(
        ...,
        description="Two rows per game: one Blue and one Red team.",
    )


# =========================
# Utility functions
# =========================
def logit(p: float) -> float:
    """Numerically-stable logit."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return float(np.log(p / (1 - p)))


def get_tree_top_features(
    model, feature_cols: List[str], top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Extract the top N most important features from a tree model
    with feature_importances_ attribute.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df.head(top_n).to_dict(orient="records")


def get_xgb_contribs_single(
    model, X_row: pd.DataFrame, feature_cols: List[str], top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Compute per-input feature contributions (SHAP values) for a single row
    using XGBoost Booster + DMatrix with pred_contribs=True.

    如果模型不是 XGBoost（没有 get_booster），会返回 [] 安全退化。
    """
    # 只保留一行
    if X_row.shape[0] != 1:
        X_row = X_row.iloc[[0]]

    # 不是 XGBoost 的话直接空
    if not hasattr(model, "get_booster"):
        return []

    # 按 feature_cols 顺序取列，避免列顺序错乱
    X_used = X_row[feature_cols].copy()
    for c in X_used.columns:
        X_used[c] = pd.to_numeric(X_used[c], errors="coerce")
    X_used = X_used.fillna(0.0)

    dmatrix = xgb.DMatrix(X_used.values, feature_names=feature_cols)
    booster = model.get_booster()
    try:
        contribs_all = booster.predict(dmatrix, pred_contribs=True)[0]
    except Exception:
        return []

    # XGBoost 返回 N+1（最后一列为 bias）
    if len(contribs_all) == len(feature_cols) + 1:
        contribs = contribs_all[:-1]
    else:
        contribs = contribs_all[: len(feature_cols)]

    contribs = np.asarray(contribs, dtype=float)
    abs_order = np.argsort(-np.abs(contribs))

    results = []
    for idx in abs_order[:top_n]:
        results.append(
            {
                "feature": feature_cols[idx],
                "contribution": float(contribs[idx]),
            }
        )
    return results


# =========================
# Load models
# =========================
# 注意：这里用的是你改后的文件名（不带 xgb）
#   - model/lineup_model.joblib
#   - model/realtime_model.joblib
#   - model/realtime_mid10_model.joblib
#   - model/realtime_mid15_model.joblib
#   - model/realtime_mid20_model.joblib
#   - model/realtime_mid25_model.joblib

# Lineup model: Pipeline(onehot + tree model)
lineup_model_path = os.path.join(DATA_PATH, "lineup_model.joblib")
if not os.path.exists(lineup_model_path):
    raise RuntimeError(f"Lineup model file not found: {lineup_model_path}")
lineup_pipeline = joblib.load(lineup_model_path)

lineup_categorical_cols = [
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


def _get_lineup_inner_model_and_onehot():
    """内部工具：从 pipeline 中取出 onehot 和 tree 模型。"""
    onehot = lineup_pipeline.named_steps.get("onehot")
    # 模型这一步的名字不一定叫 xgb，所以先尝试 xgb，再 fallback 任意非 onehot 的 step
    model = lineup_pipeline.named_steps.get("xgb", None)
    if model is None:
        for name, step in lineup_pipeline.named_steps.items():
            if name != "onehot":
                model = step
                break
    return onehot, model


def get_lineup_top_features(top_n: int = 10) -> List[Dict[str, Any]]:
    """全局特征重要性（针对 one-hot 后的 champion 组合）。"""
    onehot, model = _get_lineup_inner_model_and_onehot()
    if onehot is None or model is None:
        return []

    feature_names = list(onehot.get_feature_names_out(lineup_categorical_cols))
    return get_tree_top_features(model, feature_names, top_n=top_n)


def get_lineup_contribs_for_input(
    item_dict: Dict[str, Any], top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    计算单局阵容的 per-input 特征贡献（SHAP-like），在 one-hot 空间下。
    """
    df = pd.DataFrame([item_dict], columns=lineup_categorical_cols)
    onehot, model = _get_lineup_inner_model_and_onehot()
    if onehot is None or model is None or not hasattr(model, "get_booster"):
        return []

    X_ohe = onehot.transform(df)
    feature_names = list(onehot.get_feature_names_out(lineup_categorical_cols))

    # X_ohe 是 sparse，DMatrix 可以直接吃
    dmatrix = xgb.DMatrix(X_ohe, feature_names=feature_names)
    booster = model.get_booster()
    try:
        contribs_all = booster.predict(dmatrix, pred_contribs=True)[0]
    except Exception:
        return []

    if len(contribs_all) == len(feature_names) + 1:
        contribs = contribs_all[:-1]
    else:
        contribs = contribs_all[: len(feature_names)]

    contribs = np.asarray(contribs, dtype=float)
    abs_order = np.argsort(-np.abs(contribs))

    results = []
    for idx in abs_order[:top_n]:
        results.append(
            {
                "feature": feature_names[idx],
                "contribution": float(contribs[idx]),
            }
        )
    return results


# Full-game realtime model
realtime_full_path = os.path.join(DATA_PATH, "realtime_model.joblib")
if not os.path.exists(realtime_full_path):
    raise RuntimeError(f"Full-game realtime model file not found: {realtime_full_path}")
realtime_full_payload = joblib.load(realtime_full_path)
realtime_full_model = realtime_full_payload["model"]
realtime_full_feature_cols: List[str] = realtime_full_payload["feature_cols"]

# Mid-game models: 10 / 15 / 20 / 25
mid_models: Dict[int, Dict[str, Any]] = {}
for minute in [10, 15, 20, 25]:
    mid_path = os.path.join(DATA_PATH, f"realtime_mid{minute}_model.joblib")
    if os.path.exists(mid_path):
        payload = joblib.load(mid_path)
        mid_models[minute] = {
            "model": payload["model"],
            "feature_cols": payload["feature_cols"],
        }


# =========================
# FastAPI app
# =========================
app = FastAPI(title="LOL Models API", version="1.2.0")


# =========================
# Lineup prediction: single or batch
# =========================
@app.post("/predict/lineup")
def predict_lineup(
    req: Union[LineupItem, List[LineupItem]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    选人阶段胜率预测。

    返回：
    - p_blue, p_red
    - top_features：全局 feature_importances
    - feature_contribs：当前这局阵容在 one-hot 空间的 per-input 贡献
    """
    if isinstance(req, LineupItem):
        items = [req]
        single_input = True
    else:
        items = req
        single_input = False

    data = [item.dict() for item in items]
    df = pd.DataFrame(data, columns=lineup_categorical_cols)

    proba = lineup_pipeline.predict_proba(df)[:, 1]
    top_feats_global = get_lineup_top_features(top_n=15)

    results: List[Dict[str, Any]] = []
    for row_dict, p in zip(data, proba):
        p_blue = float(p)
        p_red = 1.0 - p_blue
        per_input_contribs = get_lineup_contribs_for_input(row_dict, top_n=15)

        results.append(
            {
                "p_blue": p_blue,
                "p_red": p_red,
                "top_features": top_feats_global,
                "feature_contribs": per_input_contribs,
            }
        )

    return results[0] if single_input else results


# =========================
# Full-game realtime prediction: single or batch
# =========================
@app.post("/predict/realtime/full")
def predict_realtime_full(
    req: Union[RealtimeGame, List[RealtimeGame]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    全局经济 / 数据结束后的胜率预测。
    """
    if isinstance(req, RealtimeGame):
        games = [req]
        single_input = True
    else:
        games = req
        single_input = False

    results: List[Dict[str, Any]] = []

    for game in games:
        if len(game.rows) != 2:
            raise HTTPException(
                status_code=400,
                detail="Each game must contain exactly two rows (Blue and Red).",
            )

        rows = [r.dict() for r in game.rows]
        df_raw = pd.DataFrame(rows)

        sides = set(df_raw["side"])
        if sides != {"Blue", "Red"}:
            raise HTTPException(
                status_code=400,
                detail="Both Blue and Red sides must appear exactly once.",
            )

        gameid = df_raw["gameid"].iloc[0]
        blue_team = df_raw[df_raw["side"] == "Blue"]["teamname"].iloc[0]
        red_team = df_raw[df_raw["side"] == "Red"]["teamname"].iloc[0]

        # 特征对齐
        df_feat = df_raw.copy()
        df_feat["side"] = df_feat["side"].map({"Blue": 0, "Red": 1})

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
            "result",
        ]
        for c in drop_cols:
            if c in df_feat.columns:
                df_feat = df_feat.drop(columns=[c])

        for c in realtime_full_feature_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0.0

        extra_cols = [c for c in df_feat.columns if c not in realtime_full_feature_cols]
        if extra_cols:
            df_feat = df_feat.drop(columns=extra_cols)

        df_feat = df_feat[realtime_full_feature_cols]

        for col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
        df_feat = df_feat.fillna(0.0)

        blue_idx = df_raw[df_raw["side"] == "Blue"].index[0]
        red_idx = df_raw[df_raw["side"] == "Red"].index[0]

        X_blue = df_feat.loc[[blue_idx]]
        X_red = df_feat.loc[[red_idx]]

        # raw prob
        p_blue_raw = float(realtime_full_model.predict_proba(X_blue)[:, 1][0])
        p_red_raw = float(realtime_full_model.predict_proba(X_red)[:, 1][0])

        # logit + softmax 归一化
        l_blue = logit(p_blue_raw)
        l_red = logit(p_red_raw)
        exp_blue = np.exp(l_blue)
        exp_red = np.exp(l_red)
        p_blue_norm = float(exp_blue / (exp_blue + exp_red))
        p_red_norm = float(exp_red / (exp_blue + exp_red))

        # 全局重要性
        top_feats = get_tree_top_features(
            realtime_full_model, realtime_full_feature_cols, top_n=15
        )

        # per-input 贡献
        contribs_blue = get_xgb_contribs_single(
            realtime_full_model, X_blue, realtime_full_feature_cols, top_n=15
        )
        contribs_red = get_xgb_contribs_single(
            realtime_full_model, X_red, realtime_full_feature_cols, top_n=15
        )

        results.append(
            {
                "gameid": gameid,
                "blue_team": blue_team,
                "red_team": red_team,
                "p_blue_raw": p_blue_raw,
                "p_red_raw": p_red_raw,
                "p_blue_norm": p_blue_norm,
                "p_red_norm": p_red_norm,
                "top_features": top_feats,
                "feature_contribs_blue": contribs_blue,
                "feature_contribs_red": contribs_red,
            }
        )

    return results[0] if single_input else results


# =========================
# Mid-game realtime prediction: single or batch
# =========================
@app.post("/predict/realtime/mid/{minute}")
def predict_realtime_mid(
    minute: int, req: Union[RealtimeGame, List[RealtimeGame]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    T 分钟时（10/15/20/25）根据局势预测胜率。
    """
    if minute not in mid_models:
        raise HTTPException(
            status_code=404,
            detail=f"Mid-game model for minute={minute} is not available.",
        )

    if isinstance(req, RealtimeGame):
        games = [req]
        single_input = True
    else:
        games = req
        single_input = False

    payload = mid_models[minute]
    model = payload["model"]
    feature_cols: List[str] = payload["feature_cols"]

    results: List[Dict[str, Any]] = []

    for game in games:
        if len(game.rows) != 2:
            raise HTTPException(
                status_code=400,
                detail="Each game must contain exactly two rows (Blue and Red).",
            )

        rows = [r.dict() for r in game.rows]
        df_raw = pd.DataFrame(rows)

        sides = set(df_raw["side"])
        if sides != {"Blue", "Red"}:
            raise HTTPException(
                status_code=400,
                detail="Both Blue and Red sides must appear exactly once.",
            )

        gameid = df_raw["gameid"].iloc[0]
        blue_team = df_raw[df_raw["side"] == "Blue"]["teamname"].iloc[0]
        red_team = df_raw[df_raw["side"] == "Red"]["teamname"].iloc[0]

        # 根据请求里的列找 atT 字段（用于检查）
        mid_cols_req = [c for c in df_raw.columns if f"at{minute}" in c]
        if not mid_cols_req:
            raise HTTPException(
                status_code=400,
                detail=f"No mid-game columns containing 'at{minute}' were found in request.",
            )

        # side + 所有 mid 特征列
        df_mid = df_raw[["side"] + mid_cols_req].copy()
        df_mid["side"] = df_mid["side"].map({"Blue": 0, "Red": 1})

        for col in df_mid.columns:
            df_mid[col] = pd.to_numeric(df_mid[col], errors="coerce")
        df_mid = df_mid.fillna(0.0)

        # 对齐训练特征
        for c in feature_cols:
            if c not in df_mid.columns:
                df_mid[c] = 0.0
        extra_cols = [c for c in df_mid.columns if c not in feature_cols]
        if extra_cols:
            df_mid = df_mid.drop(columns=extra_cols)
        df_mid = df_mid[feature_cols]

        blue_idx = df_raw[df_raw["side"] == "Blue"].index[0]
        red_idx = df_raw[df_raw["side"] == "Red"].index[0]

        X_blue = df_mid.loc[[blue_idx]]
        X_red = df_mid.loc[[red_idx]]

        # raw prob
        p_blue_raw = float(model.predict_proba(X_blue)[:, 1][0])
        p_red_raw = float(model.predict_proba(X_red)[:, 1][0])

        l_blue = logit(p_blue_raw)
        l_red = logit(p_red_raw)
        exp_blue = np.exp(l_blue)
        exp_red = np.exp(l_red)
        p_blue_norm = float(exp_blue / (exp_blue + exp_red))
        p_red_norm = float(exp_red / (exp_blue + exp_red))

        # 全局重要性
        top_feats = get_tree_top_features(model, feature_cols, top_n=15)

        # per-input 贡献
        contribs_blue = get_xgb_contribs_single(
            model, X_blue, feature_cols, top_n=15
        )
        contribs_red = get_xgb_contribs_single(
            model, X_red, feature_cols, top_n=15
        )

        results.append(
            {
                "gameid": gameid,
                "minute": minute,
                "blue_team": blue_team,
                "red_team": red_team,
                "p_blue_raw": p_blue_raw,
                "p_red_raw": p_red_raw,
                "p_blue_norm": p_blue_norm,
                "p_red_norm": p_red_norm,
                "top_features": top_feats,
                "feature_contribs_blue": contribs_blue,
                "feature_contribs_red": contribs_red,
            }
        )

    return results[0] if single_input else results
