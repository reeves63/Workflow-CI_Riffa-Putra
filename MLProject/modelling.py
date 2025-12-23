import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline


def load_data(filepath):
    """Load preprocessed dataset."""
    print("LOADING PREPROCESSED DATASET")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts()}")
    print(f"[INFO] Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")
    
    return X, y


def create_preprocessor():
    """Create preprocessing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [
                'age', 'bmi', 'HbA1c_level', 
                'blood_glucose_level', 'hypertension', 'heart_disease'
            ]),
            ('cat', OneHotEncoder(), ['gender', 'smoking_history'])
        ]
    )
    return preprocessor


def create_model_pipeline():
    """Create complete model pipeline."""
    preprocessor = create_preprocessor()
    over = SMOTE(sampling_strategy=0.1, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    
    pipeline = imbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('over', over),
        ('under', under),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    
    return pipeline


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def train_model(X_train, X_test, y_train, y_test, experiment_name="Diabetes-Prediction-CI"):
    """Train model with MLflow tracking."""
    print("\n" + "="*80)
    print("MODEL TRAINING WITH MLFLOW")
    print("="*80)
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Enable autolog for sklearn
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest-CI-Auto"):
        print("[INFO] MLflow autolog enabled")
        print("[INFO] Training Random Forest model...")
        
        # Create and train model
        model = create_model_pipeline()
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="DiabetesRF_CI"
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n[RESULTS] Model trained successfully!")
        print(f"[RESULTS] Accuracy:  {accuracy:.4f}")
        print(f"[RESULTS] Precision: {precision:.4f}")
        print(f"[RESULTS] Recall:    {recall:.4f}")
        print(f"[RESULTS] F1-Score:  {f1:.4f}")
        
        # Generate and log confusion matrix
        plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        # Log additional tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("dataset", "diabetes_prediction")
        mlflow.set_tag("student", "Riffa Putra")
        mlflow.set_tag("environment", "CI/CD")
        mlflow.set_tag("pipeline", "GitHub Actions")
        
        print(f"\n[INFO] MLflow run completed")
        
    return model, y_pred


def display_classification_report(y_test, y_pred):
    """Display detailed classification report."""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, 
                                target_names=['No Diabetes', 'Diabetes']))


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Diabetes Prediction Model')
    parser.add_argument('--data_path', type=str, default='diabetes_dataset_processed.csv',
                        help='Path to preprocessed dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--experiment_name', type=str, default='Diabetes-Prediction-CI',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DIABETES PREDICTION - CI/CD PIPELINE")
    print("Student: Riffa Putra")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Test Size: {args.test_size}")
    print(f"Random State: {args.random_state}")
    print(f"Experiment Name: {args.experiment_name}")
    print("="*80)
    
    # Step 1: Load data
    X, y = load_data(args.data_path)
    
    # Step 2: Split data
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"[INFO] Training set size: {X_train.shape}")
    print(f"[INFO] Test set size:     {X_test.shape}")
    
    # Step 3: Train model with MLflow
    model, y_pred = train_model(X_train, X_test, y_train, y_test, args.experiment_name)
    
    # Step 4: Display detailed results
    display_classification_report(y_test, y_pred)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()