import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def custom_score(y_true, y_pred):
    """Calculate the competition metric score."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 100 * (1 - rmse)

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_base_models():
    """Get dictionary of base models for training."""
    return {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'Ridge': Ridge()
    }

def optimize_xgboost(X_train, y_train, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'tree_method': 'gpu_hist',  # Enable GPU acceleration
            'predictor': 'gpu_predictor' # Use GPU predictor
        }
        
        model = XGBRegressor(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_root_mean_squared_error')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_lightgbm(X_train, y_train, n_trials=100):
    """Optimize LightGBM hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'device': 'gpu' # Enable GPU acceleration
        }
        
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1) # Use all available cores
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_root_mean_squared_error')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_random_forest(X_train, y_train, n_trials=50):
    """Optimize RandomForest hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8, 1.0])
        }
        
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1) # Use all available cores
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_root_mean_squared_error')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_models(X_train, y_train, X_val, y_val):
    """Train multiple models and evaluate their performance."""
    models = get_base_models()
    results = {}
    
    # No longer training individual models here as they will be tuned and trained within KFold
    # This function will primarily be for initial checks or if individual model scores are needed outside the ensemble
    # For the KFold loop, models are instantiated with optimized parameters.
    
    # For now, return empty results to proceed with KFold logic
    return results

def create_ensemble(models_dict, X_train, y_train, optimized_params_xgb, optimized_params_lgb, optimized_params_rf):
    """
    Create a stacking ensemble from the best models with optimized hyperparameters.
    
    Args:
        models_dict (dict): Dictionary of trained base models and their results (can be empty if models are re-instantiated)
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        optimized_params_xgb (dict): Optimized hyperparameters for XGBoost
        optimized_params_lgb (dict): Optimized hyperparameters for LightGBM
        optimized_params_rf (dict): Optimized hyperparameters for RandomForest
        
    Returns:
        sklearn.ensemble.StackingRegressor: Trained stacking ensemble model
    """
    base_models = [
        ('xgb', XGBRegressor(**optimized_params_xgb, random_state=42)),
        ('lgb', LGBMRegressor(**optimized_params_lgb, random_state=42)),
        ('rf', RandomForestRegressor(**optimized_params_rf, random_state=42))
    ]

    meta_learner = Ridge(alpha=0.1)

    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5
    )

    stacking_regressor.fit(X_train, y_train)
    return stacking_regressor

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze and plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
        
        return importance_df
    return None

def create_shap_analysis(model, X_test, feature_names):
    """Create SHAP analysis for model interpretation."""
    if isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
        # SHAP explainer should be fit on the model, not just estimators_[0]
        # For StackingRegressor, it's more complex to get SHAP values directly for the ensemble.
        # We can analyze the meta-learner or one of the base learners for interpretation.
        # For simplicity, we'll use the first base model as before for now.
        
        # If you want SHAP for the full ensemble, you might need to use shap.KernelExplainer
        # which is slower but model-agnostic.
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig('plots/shap_summary.png')
        plt.close()
        
        return explainer, shap_values
    return None, None

def main():
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Load and preprocess data
    from data_exploration import load_data
    from feature_engineering import preprocess_data
    
    train_df, test_df, sample_submission = load_data()
    
    # Identify categorical columns
    categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocess data
    train_processed, test_processed = preprocess_data(train_df, test_df, categorical_columns)
    # Preprocess test_df separately for feature creation
    test_processed_for_features = preprocess_data(test_df.copy(), test_df.copy(), categorical_columns)[0]
    
    # Prepare features and target
    X = train_processed.drop(['efficiency', 'id'], axis=1)
    y = train_processed['efficiency']
    
    # --- Hyperparameter Tuning (run once before K-Fold if resources are limited) ---
    print("\n--- Starting Hyperparameter Optimization for Base Models ---")
    # For demonstration, limiting n_trials. Increase for better optimization.
    optimized_params_xgb = optimize_xgboost(X, y, n_trials=200)
    print(f"Best XGBoost params: {optimized_params_xgb}")
    
    optimized_params_lgb = optimize_lightgbm(X, y, n_trials=200)
    print(f"Best LightGBM params: {optimized_params_lgb}")

    optimized_params_rf = optimize_random_forest(X, y, n_trials=100) # RF is slower, so fewer trials
    print(f"Best RandomForest params: {optimized_params_rf}")
    
    # --- K-Fold Cross-Validation for Final Model --- 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])  # Out-of-fold predictions
    test_preds = np.zeros(test_processed_for_features.shape[0]) # Test predictions
    
    # No longer storing all models, only the last one for feature importance/SHAP
    last_trained_ensemble = None
    last_trained_xgb_model = None
    
    val_scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{kf.n_splits} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train base models and get results (train_models is now a placeholder, will be removed if not needed)
        model_results = train_models(X_train, y_train, X_val, y_val) # This will be empty
        
        # Create and train ensemble for this fold with optimized hyperparameters
        ensemble = create_ensemble(model_results, X_train, y_train, 
                                   optimized_params_xgb, optimized_params_lgb, optimized_params_rf)
        
        # Store the last trained ensemble model (for analysis outside the loop)
        last_trained_ensemble = ensemble
        
        # Store the last trained XGBoost model from the ensemble for SHAP (as it's tree-based)
        # Note: This assumes XGBoost is the first estimator in the ensemble list.
        # If the order changes, this might need adjustment.
        if 'xgb' in ensemble.named_estimators_:
            last_trained_xgb_model = ensemble.named_estimators_['xgb']
        
        # Make out-of-fold predictions for validation set
        oof_preds[val_index] = ensemble.predict(X_val)
        
        # Make predictions on test set (average over folds later)
        test_preds += ensemble.predict(test_processed_for_features.drop(['id'], axis=1)) / kf.n_splits
        
        # Calculate and store validation score for this fold
        fold_val_score = custom_score(y_val, oof_preds[val_index])
        val_scores.append(fold_val_score)
        print(f"Fold {fold+1} Ensemble Validation Score: {fold_val_score:.2f}")
        
    print("\n=== K-Fold Cross-Validation Summary ===")
    print(f"Mean Ensemble Validation Score: {np.mean(val_scores):.2f}")
    print(f"Std Ensemble Validation Score: {np.std(val_scores):.2f}")
    
    # Final predictions from averaged test predictions
    final_test_predictions = test_preds
    
    # Clip predictions to ensure they are within the realistic efficiency bounds (0 to 1)
    final_test_predictions = np.clip(final_test_predictions, 0, 1)

    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'efficiency': final_test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    # Optional: Feature importance and SHAP analysis using one of the trained models
    if last_trained_xgb_model:
        print("\nPerforming Feature Importance and SHAP analysis...")
        feature_names = X.columns
        
        analyze_feature_importance(last_trained_xgb_model, feature_names)
        # For SHAP, using the last validation fold's X_val as X_test_sample
        create_shap_analysis(last_trained_xgb_model, X_val, feature_names)

if __name__ == "__main__":
    main() 