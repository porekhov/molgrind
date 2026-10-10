def train_martini_bead_models(
    csv_path: str,
    out_pkl_path: str = "martini_bead_classifiers.pkl",
    random_state: int = 42,
    cv_splits: int = 3,
    max_k: int | None = None,
    n_estimators: int = 400,
):
    """
    Train and save Martini bead endpoint classifiers:

      - Bead_type: ExtraTrees (trained on full data EXCLUDING class 'X')
      - Bead_size: Multinomial Logistic Regression (trained on full data)
      - Bead_subtype: separate model per Bead_type in {'C','N','P','X'}
          C: ExtraTrees
          N: ExtraTrees
          P: Logistic Regression
          X: ExtraTrees

    Descriptor selection:
      For each endpoint/model we rank descriptors (by ExtraTrees importance),
      then pick the best top-k subset by stratified CV macro-F1.

    Saves a single pickle bundle to out_pkl_path with:
      bundle['bead_type'], bundle['bead_size'], bundle['bead_subtype'][bt]

    Requirements:
      pandas, numpy, scikit-learn

    Returns:
      (bundle, cv_summary_df)
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

    # -------------------------
    # Helpers
    # -------------------------
    def _get_feature_columns(df_):
        # Adjust these if your CSV has different ID columns
        id_cols = [
            "Unnamed: 0.1", "Unnamed: 0", "Molecule", "Bead", "Type", "SMILES"
        ]
        target_cols = ["Bead_type", "Bead_size", "Bead_subtype"]
        feat_cols = [
            c for c in df_.columns
            if c not in id_cols + target_cols and pd.api.types.is_numeric_dtype(df_[c])
        ]
        return feat_cols

    def _cv_metrics(X, y, estimator, cv):
        y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1)
        return {
            "n": int(len(y)),
            "classes": int(len(np.unique(y))),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro")),
        }

    def _rank_features_by_extratrees(X, y, feature_names, seed):
        # ExtraTrees is a robust default for ranking in mixed-scale descriptor sets
        ranker = ExtraTreesClassifier(
            n_estimators=max(300, n_estimators // 2),
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        )
        ranker.fit(X, y)
        importances = getattr(ranker, "feature_importances_", None)
        if importances is None:
            # Should not happen, but safe fallback
            return list(feature_names)

        order = np.argsort(importances)[::-1]
        return [feature_names[i] for i in order]

    def _select_topk_by_cv(X_df, y, feature_ranking, estimator, cv, max_k_local):
        # pick best k by macro-F1 over prefixes of ranked features
        if max_k_local is None:
            max_k_local = len(feature_ranking)
        max_k_local = int(min(max_k_local, len(feature_ranking)))

        best = {"k": None, "f1_macro": -np.inf, "features": None, "metrics": None}

        # start from k=2 to avoid degenerate single-feature cases
        for k in range(2, max_k_local + 1):
            feats = feature_ranking[:k]
            X = X_df[feats].to_numpy()
            m = _cv_metrics(X, y, estimator, cv)
            if m["f1_macro"] > best["f1_macro"]:
                best = {"k": k, "f1_macro": m["f1_macro"], "features": feats, "metrics": m}

        # If the dataset is tiny, k loop might be too aggressive; ensure non-empty
        if best["features"] is None:
            feats = feature_ranking[: min(2, len(feature_ranking))]
            X = X_df[feats].to_numpy()
            m = _cv_metrics(X, y, estimator, cv)
            best = {"k": len(feats), "f1_macro": m["f1_macro"], "features": feats, "metrics": m}

        return best

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(csv_path)
    feat_cols = _get_feature_columns(df)

    # Basic cleanup: replace inf with nan, drop rows with nan in any descriptor/target used
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    # CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # -------------------------
    # Define suggested models
    # -------------------------
    bead_type_model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    bead_size_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
            random_state=random_state,
        ))
    ])
    bead_subtype_models = {
        "C": ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
        "N": ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
        "P": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=12000,
                class_weight="balanced",
                random_state=random_state,
            ))
        ]),
        "X": ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    # -------------------------
    # Train Bead_type (exclude X)
    # -------------------------
    df_bt = df[df["Bead_type"].notna()].copy()
    df_bt = df_bt[df_bt["Bead_type"] != "X"].copy()
    df_bt = df_bt.dropna(subset=feat_cols + ["Bead_type"])
    y_bt = df_bt["Bead_type"].to_numpy()

    # Rank + select features
    bt_ranking = _rank_features_by_extratrees(
        df_bt[feat_cols].to_numpy(), y_bt, feat_cols, random_state
    )
    bt_best = _select_topk_by_cv(
        df_bt, y_bt, bt_ranking, bead_type_model, cv,
        max_k_local=max_k or min(40, len(feat_cols))
    )
    bt_feats = bt_best["features"]

    # Fit final
    bead_type_model.fit(df_bt[bt_feats].to_numpy(), y_bt)

    # -------------------------
    # Train Bead_size (all)
    # -------------------------
    df_bs = df[df["Bead_size"].notna()].copy()
    df_bs = df_bs.dropna(subset=feat_cols + ["Bead_size"])
    y_bs = df_bs["Bead_size"].to_numpy()

    # Rank using ExtraTrees (even though final is LR) for stable feature ordering
    bs_ranking = _rank_features_by_extratrees(
        df_bs[feat_cols].to_numpy(), y_bs, feat_cols, random_state + 1
    )
    bs_best = _select_topk_by_cv(
        df_bs, y_bs, bs_ranking, bead_size_model, cv,
        max_k_local=max_k or min(40, len(feat_cols))
    )
    bs_feats = bs_best["features"]

    bead_size_model.fit(df_bs[bs_feats].to_numpy(), y_bs)

    # -------------------------
    # Train Bead_subtype (per Bead_type)
    # -------------------------
    subtype_bundle = {}
    subtype_cv_rows = []

    for bt in ["C", "N", "P", "X"]:
        df_sub = df[(df["Bead_type"] == bt) & df["Bead_subtype"].notna()].copy()
        df_sub = df_sub.dropna(subset=feat_cols + ["Bead_subtype"])

        # If too small, keep all features to avoid unstable selection,
        # or still attempt selection with capped k.
        y_sub = df_sub["Bead_subtype"].to_numpy()
        model_sub = bead_subtype_models[bt]

        if len(df_sub) < max(12, cv_splits * 3) or len(np.unique(y_sub)) < 2:
            # Tiny set: skip selection (use all descriptors)
            sub_feats = feat_cols
            sub_metrics = _cv_metrics(df_sub[sub_feats].to_numpy(), y_sub, model_sub, cv) if len(np.unique(y_sub)) > 1 else {
                "n": int(len(y_sub)), "classes": int(len(np.unique(y_sub))),
                "accuracy": float("nan"), "balanced_accuracy": float("nan"), "f1_macro": float("nan")
            }
        else:
            # Rank + select
            sub_ranking = _rank_features_by_extratrees(
                df_sub[feat_cols].to_numpy(), y_sub, feat_cols, random_state + 10 + ord(bt)
            )
            sub_best = _select_topk_by_cv(
                df_sub, y_sub, sub_ranking, model_sub, cv,
                max_k_local=max_k or min(40, len(feat_cols))
            )
            sub_feats = sub_best["features"]
            sub_metrics = sub_best["metrics"]

        # Fit final subset model
        
        model_sub.fit(df_sub[sub_feats].to_numpy(), y_sub)

        subtype_bundle[bt] = {
            "model": model_sub,
            "features": sub_feats,
            "n_train": int(len(df_sub)),
            "n_classes": int(len(np.unique(y_sub))),
        }

        subtype_cv_rows.append({
            "endpoint": f"Bead_subtype | Bead_type={bt}",
            "model": type(model_sub).__name__ if not isinstance(model_sub, Pipeline) else "Pipeline(LogReg)",
            "features": int(len(sub_feats)),
            **sub_metrics
        })

    # -------------------------
    # Assemble bundle + CV summary
    # -------------------------
    import pandas as pd

    cv_rows = []
    cv_rows.append({
        "endpoint": "Bead_type (exclude X)",
        "model": "ExtraTreesClassifier",
        "features": int(len(bt_feats)),
        **bt_best["metrics"],
    })
    cv_rows.append({
        "endpoint": "Bead_size",
        "model": "Pipeline(LogReg)",
        "features": int(len(bs_feats)),
        **bs_best["metrics"],
    })
    cv_rows.extend(subtype_cv_rows)

    cv_summary = pd.DataFrame(cv_rows)

    bundle = {
        "feature_pool": feat_cols,
        "meta": {
            "csv_path": os.path.abspath(csv_path),
            "random_state": int(random_state),
            "cv": f"StratifiedKFold(n_splits={cv_splits}, shuffle=True, random_state={random_state})",
            "selection": "ExtraTrees importance ranking + best top-k by CV macro-F1",
            "note": "Bead_type trained excluding class X (halogens). Bead_subtype models trained per Bead_type.",
        },
        "bead_type": {
            "model": bead_type_model,
            "features": bt_feats,
            "excluded_class": "X",
        },
        "bead_size": {
            "model": bead_size_model,
            "features": bs_feats,
        },
        "bead_subtype": subtype_bundle,
    }

    # Save
    os.makedirs(os.path.dirname(out_pkl_path) or ".", exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(bundle, f)

    return bundle, cv_summary

if __name__ == "__main__":
    bundle, cv_summary = train_martini_bead_models(
        csv_path="martini3_all_beads.csv",
        out_pkl_path="martini_bead_classifiers.pkl",
        random_state=42,
        cv_splits=3,
    )
    print("Summary:")
    print(cv_summary)
