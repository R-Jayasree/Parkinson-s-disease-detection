import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')
from model_def import ParkinsonsEnsembleModel
from imblearn.over_sampling import SMOTE

def load_and_prepare_data():
    """Load and prepare datasets"""
    print("Loading UCI Parkinson's dataset...")

    # Load UCI dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['status'].value_counts()}")

    # Separate features and target
    X = df.drop(['name', 'status'], axis=1)
    y = df['status'].values

    return X, y



def main():
    """Main training pipeline"""
    print("=" * 60)
    print("PARKINSON'S DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)
    X, y = load_and_prepare_data()

    print("\nSplitting into Train/Test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")


    print("\nApplying SMOTE to balance the training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled training shape: {X_train_res.shape}, Class distribution: {np.bincount(y_train_res)}")

    model = ParkinsonsEnsembleModel()
    val_scores = model.train(X_train_res, y_train_res)  
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (Unseen Data)")
    print("=" * 60)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Healthy', 'Parkinsons']))

    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    print("\nRetraining ensemble on full dataset for deployment...")
    
    # Apply SMOTE on full dataset
    smote_full = SMOTE(random_state=42)
    X_res, y_res = smote_full.fit_resample(X, y)
    model.train(X_res, y_res)

    print("\nSaving full dataset predictions to 'predictions.csv'...")
    all_preds = model.predict(X)
    all_probs = model.predict_proba(X)[:, 1]

    df_preds = X.copy()
    df_preds['Actual'] = y
    df_preds['Prediction'] = all_preds
    df_preds['Risk_Score'] = all_probs * 100  
    df_preds.to_csv('predictions.csv', index=False)
    print("predictions.csv saved successfully!")

    print("\nSaving model...")
    with open('parkinsons_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\nModel saved as 'parkinsons_model.pkl'")
    print("\n" + "=" * 60)
    print("Training complete with proper test evaluation!")
    print("=" * 60)



if __name__ == "__main__":
    main()
