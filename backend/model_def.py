import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsEnsembleModel:
    def __init__(self):
        """Initialize ensemble model with multiple classifiers"""
        self.models = {
            'xgb': XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss'
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
    
        self.model_weights = {
            'xgb': 0.35,
            'rf': 0.25,
            'gb': 0.25,
            'svm': 0.15
        }
        
        self.scaler = RobustScaler()  
        self.feature_names = None
        
    def engineer_features(self, df):
        df = df.copy()
        
        # Jitter-Shimmer interactions (voice instability indicators)
        if 'MDVP:Jitter(%)' in df.columns and 'MDVP:Shimmer' in df.columns:
            df['jitter_shimmer_ratio'] = df['MDVP:Jitter(%)'] / (df['MDVP:Shimmer'] + 1e-6)
            df['jitter_shimmer_product'] = df['MDVP:Jitter(%)'] * df['MDVP:Shimmer']
        
        # HNR-based features (harmonicity indicators)
        if 'HNR' in df.columns:
            df['hnr_squared'] = df['HNR'] ** 2
            df['hnr_log'] = np.log1p(df['HNR'] + 20)  # Shift to positive
        
        # Spread features (voice range indicators)
        spread_cols = [col for col in df.columns if 'spread' in col.lower()]
        if len(spread_cols) >= 2:
            df['spread_mean'] = df[spread_cols].mean(axis=1)
            df['spread_std'] = df[spread_cols].std(axis=1)
        
        # PPE (Pitch Period Entropy) interactions
        if 'PPE' in df.columns and 'MDVP:Fo(Hz)' in df.columns:
            df['ppe_fo_ratio'] = df['PPE'] / (df['MDVP:Fo(Hz)'] + 1e-6)
        
        # DFA (Detrended Fluctuation Analysis) features
        if 'DFA' in df.columns:
            df['dfa_squared'] = df['DFA'] ** 2
        
        return df
    
    def train(self, X, y):
        """Train all models in the ensemble"""
        print("Engineering features...")
        X = self.engineer_features(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Total features: {len(self.feature_names)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\nTraining ensemble models...")
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Validation performance
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            val_acc = accuracy_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba)
            
            val_scores[name] = {'accuracy': val_acc, 'auc': val_auc}
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Validation ROC-AUC: {val_auc:.4f}")
        
        # Retrain on full dataset for final model
        print("\nRetraining on full dataset...")
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        return val_scores
    
    def predict_proba(self, X):
        """Predict probabilities using weighted ensemble"""
        X = self.engineer_features(X)
        
        # Ensure same features as training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        
        # Weighted ensemble prediction
        ensemble_proba = np.zeros((X_scaled.shape[0], 2))
        
        for name, model in self.models.items():
            proba = model.predict_proba(X_scaled)
            ensemble_proba += proba * self.model_weights[name]
        
        return ensemble_proba
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

