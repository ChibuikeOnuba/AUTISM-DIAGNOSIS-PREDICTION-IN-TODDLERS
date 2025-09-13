import joblib
import pandas as pd
import os

def save_your_models(rf_pipeline, log_reg_pipeline, xgb_pipeline, nb_pipeline, X):
    """
    Save your existing trained pipelines - Run this after training your models
    """
    # Create directory for saved models
    os.makedirs('saved_models', exist_ok=True)
    
    # Save each model
    joblib.dump(rf_pipeline, 'saved_models/random_forest_pipeline.joblib')
    joblib.dump(log_reg_pipeline, 'saved_models/logistic_regression_pipeline.joblib')
    joblib.dump(xgb_pipeline, 'saved_models/xgboost_pipeline.joblib')
    joblib.dump(nb_pipeline, 'saved_models/naive_bayes_pipeline.joblib')
    
    print("✅ All models saved successfully!")
    
    # Save feature information for the Streamlit app
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns
    
    feature_info = {
        'categorical_cols': categorical_cols.tolist(),
        'numeric_cols': numeric_cols.tolist(),
        'all_columns': X.columns.tolist()
    }
    
    joblib.dump(feature_info, 'saved_models/feature_info.joblib')
    print("✅ Feature information saved!")
    
    # Create performance data based on your results
    model_performance = {
        'Random Forest': {
            'f1_score': 74.29,
            'precision': 70.91,
            'recall': 78.0,
            'auc_score': 64.52
        },
        'Logistic Regression': {
            'f1_score': 64.52,
            'precision': 54.55,
            'recall': 78.95,
            'auc_score': 59.35
        },
        'XGBoost': {
            'f1_score': 76.5,
            'precision': 75.45,
            'recall': 77.57,
            'auc_score': 62.82
        },
        'Naive Bayes': {
            'f1_score': 6.96,  # Add your Naive Bayes scores here
            'precision': 3.64,
            'recall': 80.0,
            'auc_score': 55.58
        }
    }
    
    # Save model performance
    performance_df = pd.DataFrame(model_performance).T
    performance_df.to_csv('saved_models/model_performance.csv')
    print("✅ Model performance saved!")
    
    print("\n🚀 Ready to run Streamlit app!")
    print("Run: streamlit run streamlit_app.py")

# USAGE:
# After training all your models, simply call:
# save_your_models(rf_pipeline, log_reg_pipeline, xgb_pipeline, nb_pipeline, X)