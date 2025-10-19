# 🧠 Autism Spectrum Disorder Prediction in Toddlers

An interactive machine learning system for early detection and screening of Autism Spectrum Disorder (ASD) traits in toddlers using behavioral and demographic indicators.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This project implements a comprehensive machine learning solution for autism screening in toddlers. The system uses multiple algorithms to analyze behavioral patterns, demographic information, and family history to provide early indication of potential ASD traits.

**Key Objectives:**
- Early detection of autism spectrum disorder traits
- Comparison of multiple ML algorithms for optimal accuracy
- User-friendly interface for healthcare professionals and families
- Evidence-based screening tool to support clinical decision-making

## 📊 Dataset

The dataset contains screening information for toddlers with the following features:

### Input Features:
- **Case_No**: Unique case identifier
- **A10**: Response to routine changes (binary: 0/1)
- **Sex**: Gender (categorical: m/f)
- **Ethnicity**: Ethnic background (categorical: multiple options)
- **Jaundice**: Born with jaundice (binary: 0/1)
- **Family_mem_with_ASD**: Family history of ASD (binary: 0/1)
- **Who completed the test**: Test administrator (categorical)
- **Age**: Age of the child (numeric: 1-100)

### Target Variable:
- **Class/ASD Traits**: Presence of ASD traits (binary: 0/1)

**Dataset Source**: Toddler Autism Dataset (July 2018)
**Total Records**: [Add your dataset size]
**Features**: 8 input features + 1 target variable

## 🤖 Models

The system implements and compares four machine learning algorithms:

### 1. **Random Forest Classifier**
- **F1-Score**: 74.29%
- **Precision**: 70.91%
- **Recall**: 78.00%
- **AUC Score**: 64.52%

### 2. **XGBoost Classifier**
- **F1-Score**: 76.50%
- **Precision**: 75.45%
- **Recall**: 77.57%
- **AUC Score**: 62.82%

### 3. **Logistic Regression**
- **F1-Score**: 64.52%
- **Precision**: 54.55%
- **Recall**: 78.95%
- **AUC Score**: 59.35%

### 4. **Naive Bayes**
- **F1-Score**: 6.96%
- **Precision**: 3.64%
- **Recall**: 80.0%
- **AUC Score**: 55.58%

**Best Performing Model**: XGBoost with 76.50% F1-Score

## ✨ Features

### Interactive Web Application
- 🎛️ **Model Selection**: Choose between 4 different ML algorithms
- 📝 **User-Friendly Input Form**: Intuitive widgets for data entry
- 📊 **Real-Time Predictions**: Instant results with confidence scores
- 📈 **Performance Visualization**: Interactive charts comparing model metrics
- 🎯 **Risk Assessment**: Visual gauge showing prediction confidence
- 📋 **Detailed Analysis**: Comprehensive breakdown of results

### Technical Features
- **Automated Preprocessing**: Built-in scaling and encoding pipelines
- **Model Persistence**: Saved models for consistent predictions
- **Cross-Platform**: Runs on any system with Python
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Clean, medical-grade interface design

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/ChibuikeOnuba/AUTISM-DIAGNOSIS-PREDICTION-IN-TODDLERS.git
cd AUTISM-DIAGNOSIS-PREDICTION-IN-TODDLERS
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
streamlit --version
```

## 💻 Usage

### Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Use the application:**
   - Select a model from the sidebar dropdown
   - Fill in the patient information
   - Click "Predict" to get results
   - View detailed analysis and confidence scores

### Model Training (Optional)

If you want to retrain the models with new data:

```python
# Load your data
df = pd.read_csv('toddler_autism_dataset_july_2018.csv')
X = df.drop('Class/ASD Traits', axis=1)
y = df['Class/ASD Traits']

# Train models (use your existing training code)
# ...

# Save models
from save_models_script import save_your_models
save_your_models(rf_pipeline, log_reg_pipeline, xgb_pipeline, nb_pipeline, X)
```

## 📈 Model Performance

### Performance Comparison
The models were evaluated using cross-validation and the following metrics:

| Model | F1-Score | Precision | Recall | AUC Score |
|-------|----------|-----------|--------|-----------|
| XGBoost | **76.50%** | **75.45%** | 77.57% | 62.82% |
| Random Forest | 74.29% | 70.91% | **78.00%** | **64.52%** |
| Logistic Regression | 64.52% | 54.55% | 78.95% | 59.35% |
| Naive Bayes | 6.96% | 3.64% | 80.0% | 55.58% |

### Key Insights:
- **XGBoost** provides the best overall performance with balanced precision and recall
- **Random Forest** shows strong recall, making it good for not missing positive cases
- **Logistic Regression** has the highest recall but lower precision
- All models show good performance for early screening purposes

## 📁 Project Structure

```
autism-diagnosis-prediction-toddlers/
├── 📁 docs/                          # Documentation and presentations
│   ├── ASD Presentation.pdf
│   ├── Cert.pdf
│   ├── ethnicity_distribution.jpg
│   ├── fig1.jpg
│   └── Toddler data description.docx
├── 📁 saved_models/                   # Trained model files
│   ├── feature_info.joblib
│   ├── logistic_regression_pipeline.joblib
│   ├── model_performance.csv
│   ├── naive_bayes_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── xgboost_pipeline.joblib
├── 📄 app.py                         # Main Streamlit application
├── 📄 EDA on Autism prediction.ipynb # Exploratory data analysis
├── 📄 model.ipynb                    # Model training notebook
├── 📄 README.md                      # Project documentation
├── 📄 requirements.txt               # Python dependencies
├── 📄 save_models_script.py          # Script to save trained models
└── 📄 toddler_autism_dataset_july_2018.csv  # Training dataset
```

## 🤝 Contributing

We welcome contributions to improve this autism screening tool:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Areas for Contribution:
- Model performance improvements
- Additional visualization features
- UI/UX enhancements
- Documentation improvements
- Testing and validation
- Mobile responsiveness

## ⚠️ Important Disclaimer

**This tool is designed for screening and research purposes only.**

- ❌ **NOT a diagnostic tool** - Cannot replace professional medical evaluation
- ❌ **NOT a substitute** for clinical assessment by qualified healthcare providers
- ❌ **NOT medical advice** - Always consult with pediatricians or autism specialists
- ✅ **Screening aid only** - May help identify children who need further evaluation
- ✅ **Research purposes** - Useful for understanding autism detection patterns

### Recommended Next Steps:
If the tool indicates potential ASD traits:
1. **Consult a pediatrician** immediately
2. **Seek specialist evaluation** from developmental pediatricians or child psychologists
3. **Consider comprehensive testing** including ADOS, ADI-R, or other validated instruments
4. **Early intervention** - If diagnosed, early therapy significantly improves outcomes

## 📞 Support & Contact

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/ChibuikeOnuba/AUTISM-DIAGNOSIS-PREDICTION-IN-TODDLERS/issues)
- **Documentation**: Check the `docs/` folder for additional resources
- **Email**: [onubawinner042@gmail.com]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the researchers who provided the toddler autism dataset
- Streamlit community for the amazing framework
- scikit-learn and XGBoost teams for excellent ML libraries
- Healthcare professionals working in autism research and diagnosis

---

**Made with ❤️ for early autism detection and intervention**

*Last updated: September 2025*
