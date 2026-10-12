"""Feature definitions and conventional regression baselines."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASELINE_FEATURES = [
    "PBS_length",
    "RTT_length",
    "extension_length",
    "edit_position",
    "edit_length",
    "right_homology_arm_length",
    "edit_type_substitution",
    "edit_type_insertion",
    "edit_type_deletion",
    "Tm1",
    "Tm2",
    "Tm2new",
    "Tm3",
    "Tm4",
    "TmD",
    "GC_count_PBS",
    "GC_count_RTT",
    "GC_count_extension",
    "GC_fraction_PBS_percent",
    "GC_fraction_RTT_percent",
    "GC_fraction_extension_percent",
    "MFE_extension",
    "MFE_spacer",
    "DeepSpCas9_score",
]


def build_baseline_models() -> dict:
    return {
        "ElasticNet": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=20_000, random_state=42),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=160,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=260,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=8,
                random_state=42,
            ),
        ),
        "SmallMLP": TransformedTargetRegressor(
            regressor=make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(96, 48),
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=250,
                    batch_size=128,
                    random_state=42,
                ),
            ),
            transformer=StandardScaler(),
        ),
    }
