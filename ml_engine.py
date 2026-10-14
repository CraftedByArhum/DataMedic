import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataMedicEngine:
    def __init__(self, df: pd.DataFrame, target_col: str, drop_cols: list = None):
        self.df = df.copy()
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols else []
        
        # Drop rows where target itself is missing
        self.df = self.df.dropna(subset=[self.target_col])
        
        # Encode target column (handles both text like 'Yes'/'No' and numbers)
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[self.target_col].astype(str))
        self.class_names = self.target_encoder.classes_

        # Automatically preprocess arbitrary user features (Numeric + Categorical)
        self.feature_cols = [c for c in self.df.columns if c != target_col and c not in self.drop_cols]
        self.X_encoded = self._preprocess_features(self.df[self.feature_cols])
        
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_encoded),
            columns=self.X_encoded.columns,
            index=self.df.index
        )

    def _preprocess_features(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Converts any user dataset (text, categories, NaNs) into clean numeric matrices for ML."""
        X_proc = pd.DataFrame(index=X_df.index)
        for col in X_df.columns:
            # Skip likely unique ID columns (high cardinality strings)
            if X_df[col].dtype == "object" and X_df[col].nunique() > 0.85 * len(X_df):
                continue
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_proc[col] = X_df[col].fillna(X_df[col].median() if not X_df[col].dropna().empty else 0)
            else:
                # Encode categorical/text columns safely
                filled = X_df[col].astype(str).fillna("MISSING")
                X_proc[col] = LabelEncoder().fit_transform(filled)
        return X_proc.fillna(0)

    def detect_data_leakage(self, corr_threshold=0.85):
        """Detects features that suspiciously give away the target label."""
        leakage_report = []
        for col in self.X_encoded.columns:
            if self.X_encoded[col].std() == 0:
                continue
            corr = abs(np.corrcoef(self.X_encoded[col], self.y)[0, 1])
            if corr >= corr_threshold:
                leakage_report.append({
                    "Feature": col,
                    "Target_Correlation": round(float(corr), 4),
                    "AI_Explanation": f"Feature '{col}' has an extreme correlation ({corr:.2f}) with '{self.target_col}', indicating likely target leakage.",
                    "Recommendation": "Drop Feature"
                })
        return pd.DataFrame(leakage_report)

    def detect_label_errors(self, clean_feature_cols=None, confidence_threshold=0.75):
        """Uses 5-fold Stratified Out-of-Fold probabilities to find mislabeled rows on any dataset."""
        cols = clean_feature_cols if clean_feature_cols else list(self.X_encoded.columns)
        if not cols or len(self.class_names) < 2:
            return pd.DataFrame()

        X_use = self.X_encoded[cols].values
        min_class_count = pd.Series(self.y).value_counts().min()
        n_splits = min(5, max(2, int(min_class_count)))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros((len(self.df), len(self.class_names)))

        for train_idx, val_idx in skf.split(X_use, self.y):
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_use[train_idx], self.y[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_use[val_idx])

        flagged_rows = []
        for i, idx in enumerate(self.df.index):
            given_idx = int(self.y[i])
            pred_idx = int(np.argmax(oof_probs[i]))
            pred_conf = float(oof_probs[i][pred_idx])

            if pred_idx != given_idx and pred_conf >= confidence_threshold:
                given_label_str = self.class_names[given_idx]
                pred_label_str = self.class_names[pred_idx]
                flagged_rows.append({
                    "Row_Index": idx,
                    "Given_Label": given_label_str,
                    "Suggested_Label": pred_label_str,
                    "Model_Confidence": round(pred_conf * 100, 1),
                    "AI_Explanation": (
                        f"Row {idx} is labeled '{given_label_str}', but out-of-fold cross-validation predicts "
                        f"'{pred_label_str}' with {pred_conf*100:.1f}% confidence."
                    ),
                    "Action": f"Relabel to '{pred_label_str}'"
                })
        return pd.DataFrame(flagged_rows).sort_values(by="Model_Confidence", ascending=False) if flagged_rows else pd.DataFrame()

    def detect_outliers(self, clean_feature_cols=None, contamination=0.03):
        """Uses Isolation Forest to detect multivariate anomalies across user features."""
        cols = clean_feature_cols if clean_feature_cols else list(self.X_scaled.columns)
        if not cols:
            return pd.DataFrame()

        X_use = self.X_scaled[cols]
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_use)
        scores = iso.decision_function(X_use)

        outlier_rows = []
        for i, idx in enumerate(self.df.index):
            if preds[i] == -1:
                row_z = X_use.iloc[i].abs()
                top_feat = row_z.idxmax()
                z_val = row_z.max()
                raw_val = self.df.loc[idx, top_feat]

                outlier_rows.append({
                    "Row_Index": idx,
                    "Anomaly_Severity": round(abs(float(scores[i])), 3),
                    "Primary_Driver": top_feat,
                    "Raw_Value": str(raw_val),
                    "AI_Explanation": (
                        f"Row {idx} flagged by Isolation Forest. Primarily driven by '{top_feat}' "
                        f"(value: {raw_val}), which deviates {z_val:.1f} standard deviations from normal."
                    ),
                    "Action": "Remove Outlier Row"
                })
        return pd.DataFrame(outlier_rows).sort_values(by="Anomaly_Severity", ascending=False) if outlier_rows else pd.DataFrame()

    def detect_duplicates(self, clean_feature_cols=None, distance_threshold=0.05):
        """Finds exact and near-duplicate records using NearestNeighbors."""
        cols = clean_feature_cols if clean_feature_cols else list(self.X_scaled.columns)
        if not cols or len(self.df) < 2:
            return pd.DataFrame()

        X_use = self.X_scaled[cols].values
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(X_use)
        distances, indices = nn.kneighbors(X_use)

        seen_pairs = set()
        dup_rows = []
        for i in range(len(X_use)):
            dist = distances[i][1]
            neighbor_i = indices[i][1]
            if dist <= distance_threshold and i != neighbor_i:
                pair = tuple(sorted((int(self.df.index[i]), int(self.df.index[neighbor_i]))))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    dup_type = "Exact Duplicate" if dist == 0 else "Near-Duplicate"
                    dup_rows.append({
                        "Row_Index": pair[1],
                        "Matches_Row": pair[0],
                        "Feature_Distance": round(float(dist), 5),
                        "AI_Explanation": (
                            f"{dup_type}: Row {pair[1]} has a Euclidean distance of {dist:.5f} "
                            f"from Row {pair[0]} across all normalized features."
                        ),
                        "Action": f"Drop Row {pair[1]}"
                    })
        return pd.DataFrame(dup_rows)