import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Autism Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .positive-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .negative-prediction {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models and feature information"""
    models = {}
    model_files = {
        'Random Forest': 'saved_models/random_forest_pipeline.joblib',
        'Logistic Regression': 'saved_models/logistic_regression_pipeline.joblib',
        'XGBoost': 'saved_models/xgboost_pipeline.joblib',
        'Naive Bayes': 'saved_models/naive_bayes_pipeline.joblib'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
    
    # Load feature information
    if os.path.exists('saved_models/feature_info.joblib'):
        feature_info = joblib.load('saved_models/feature_info.joblib')
    else:
        # Fallback feature information based on your dataset
        feature_info = {
            'categorical_cols': ['Sex', 'Ethnicity', 'Who completed the test'],
            'numeric_cols': ['A10', 'Jaundice', 'Family_mem_with_ASD', 'Age'],
            'all_columns': ['A10', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'Age']
        }
    
    # Load model performance if available
    performance_data = None
    if os.path.exists('saved_models/model_performance.csv'):
        performance_data = pd.read_csv('saved_models/model_performance.csv', index_col=0)
    
    return models, feature_info, performance_data

def create_input_form(feature_info):
    """Create input form based on dataset features"""
    st.sidebar.header("📊 Patient Information")
    
    inputs = {}
    
    # Numeric inputs
    if 'A10' in feature_info['all_columns']:
        inputs['A10'] = st.sidebar.selectbox(
            "A10 - Does your child/you get easily upset by minor changes in routine?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
    
    if 'Jaundice' in feature_info['all_columns']:
        inputs['Jaundice'] = st.sidebar.selectbox(
            "Jaundice - Was your child/you born with jaundice?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
    
    if 'Family_mem_with_ASD' in feature_info['all_columns']:
        inputs['Family_mem_with_ASD'] = st.sidebar.selectbox(
            "Family Member with ASD",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
    
    if 'Age' in feature_info['all_columns']:
        inputs['Age'] = st.sidebar.slider(
            "Age",
            min_value=1,
            max_value=5,
            value=1,
            help="Age of the individual being assessed"
        )
    
    # Categorical inputs
    if 'Sex' in feature_info['all_columns']:
        inputs['Sex'] = st.sidebar.selectbox(
            "Sex",
            options=['m', 'f'],
            format_func=lambda x: "Male" if x == 'm' else "Female"
        )
    
    if 'Ethnicity' in feature_info['all_columns']:
        inputs['Ethnicity'] = st.sidebar.selectbox(
            "Ethnicity",
            options=['White European', 'middle eastern', 'Hispanic', 'black', 'asian', 'others'],
            index=0
        )
    
    if 'Who completed the test' in feature_info['all_columns']:
        inputs['Who completed the test'] = st.sidebar.selectbox(
            "Who completed the test?",
            options=['family member', 'self', 'health care professional', 'others'],
            index=0
        )
    
    return inputs

def create_prediction_visualization(prediction, probability):
    """Create interactive visualization for prediction results"""
    
    # Prediction result with custom styling
    if prediction == 1:
        st.markdown("""
        <div class="prediction-box positive-prediction">
            ⚠️ HIGH RISK: ASD Traits Detected<br>
            <small>Recommendation: Consult with a healthcare professional</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-box negative-prediction">
            ✅ LOW RISK: No ASD Traits Detected<br>
            <small>Continue regular monitoring and development</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#ff6b6b" if prediction == 1 else "#4ecdc4"},
            'steps': [
                {'range': [0, 50], 'color': "#e8f5e8"},
                {'range': [50, 80], 'color': "#fff3cd"},
                {'range': [80, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_model_performance(performance_data):
    """Display model performance comparison"""
    if performance_data is not None:
        st.subheader("📈 Model Performance Comparison")
        
        # Create performance comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1-Score', 'Precision', 'Recall', 'AUC Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['f1_score', 'precision', 'recall', 'auc_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, (metric, pos) in enumerate(zip(metrics, positions)):
            fig.add_trace(
                go.Bar(
                    x=performance_data.index,
                    y=performance_data[metric],
                    name=metric.title(),
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance table
        st.subheader("📊 Detailed Performance Metrics")
        performance_display = performance_data.round(4)
        st.dataframe(performance_display, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Autism Spectrum Disorder Prediction System</h1>', unsafe_allow_html=True)
    
    # Load models
    models, feature_info, performance_data = load_models()
    
    if not models:
        st.error("No models found! Please run the model training script first.")
        return
    
    # Sidebar for model selection and inputs
    st.sidebar.title("🎯 Model Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Prediction Model",
        options=list(models.keys()),
        help="Choose which machine learning model to use for prediction"
    )
    
    # Display model info
    st.sidebar.info(f"**Selected Model:** {selected_model}")
    
    # Input form
    inputs = create_input_form(feature_info)
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Create input dataframe
            input_df = pd.DataFrame([inputs])
            
            # Make prediction
            selected_pipeline = models[selected_model]
            prediction = selected_pipeline.predict(input_df)[0]
            prediction_proba = selected_pipeline.predict_proba(input_df)[0]
            
            # Get the probability for the predicted class
            prob = prediction_proba[prediction]
            
            st.subheader(f"🎯 Prediction Results - {selected_model}")
            
            # Create prediction visualization
            create_prediction_visualization(prediction, prob)
            
            # Detailed results
            st.subheader("📋 Detailed Analysis")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("Prediction", "ASD Traits Present" if prediction == 1 else "No ASD Traits", 
                         delta=None)
                st.metric("Confidence", f"{prob:.2%}", delta=None)
            
            with result_col2:
                st.metric("Risk Level", "High" if prob > 0.7 else "Moderate" if prob > 0.5 else "Low")
                st.metric("Model Used", selected_model, delta=None)
            
            # Show input summary
            st.subheader("📝 Input Summary")
            input_display = pd.DataFrame(inputs.items(), columns=['Feature', 'Value'])
            st.dataframe(input_display, use_container_width=True, hide_index=True)
            
        else:
            st.info("👈 Please fill in the information in the sidebar and click 'Predict' to get results.")
            
            # Show sample information
            st.subheader("ℹ️ About This System")
            st.markdown("""
            This system uses machine learning to assess the likelihood of Autism Spectrum Disorder (ASD) traits 
            based on various demographic and behavioral factors.
            
            **Available Models:**
            - **Random Forest**: Ensemble method using multiple decision trees
            - **Logistic Regression**: Statistical model for binary classification
            - **XGBoost**: Gradient boosting framework for high performance
            
            **Important Note:** This tool is for screening purposes only and should not replace professional 
            medical diagnosis. Always consult with healthcare professionals for proper evaluation.
            """)
    
    with col2:
        st.subheader("🔍 Model Information")
        
        if selected_model in models:
            model_info = {
                'Random Forest': {
                    'description': 'Ensemble learning method that combines multiple decision trees',
                    'pros': ['High accuracy', 'Handles missing values well', 'Feature importance'],
                    'cons': ['Can overfit', 'Less interpretable']
                },
                'Logistic Regression': {
                    'description': 'Statistical model that uses logistic function for binary classification',
                    'pros': ['Highly interpretable', 'Fast training', 'Probabilistic output'],
                    'cons': ['Assumes linear relationship', 'Sensitive to outliers']
                },
                'XGBoost': {
                    'description': 'Gradient boosting framework optimized for performance',
                    'pros': ['High performance', 'Handles missing data', 'Feature importance'],
                    'cons': ['Complex tuning', 'Can overfit']
                }
            }
            
            if selected_model in model_info:
                info = model_info[selected_model]
                st.write(f"**Description:** {info['description']}")
                st.write("**Pros:**")
                for pro in info['pros']:
                    st.write(f"• {pro}")
                st.write("**Cons:**")
                for con in info['cons']:
                    st.write(f"• {con}")
    
    # Model performance section
    if performance_data is not None:
        st.markdown("---")
        display_model_performance(performance_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <small>
            Developed for ASD screening • Not a substitute for professional medical advice<br>
            Last updated: {}
        </small>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()