import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="MediScan AI — Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS (MATCHING HTML DESIGN) ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Dark theme background */
    .stApp {
        background: linear-gradient(180deg, #06080F 0%, #0D1117 100%);
        color: #F0F4FF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        text-align: center;
        padding: 60px 0 40px;
    }
    
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #00D9B8;
        background: rgba(0,217,184,0.15);
        border: 1px solid rgba(0,217,184,0.25);
        padding: 8px 20px;
        border-radius: 20px;
        margin-bottom: 24px;
    }
    
    .main-title {
        font-size: 64px;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -2px;
        margin: 20px 0;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00D9B8 0%, #0099FF 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 18px;
        color: #8892A4;
        max-width: 600px;
        margin: 20px auto;
        line-height: 1.6;
    }
    
    .disclaimer {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #5A6478;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 8px 16px;
        border-radius: 8px;
        margin-top: 16px;
    }
    
    /* Disease selector cards */
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin: 40px 0;
    }
    
    .disease-card {
        background: #0D1117;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        border-color: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    
    .disease-icon {
        font-size: 40px;
        margin-bottom: 12px;
    }
    
    .disease-name {
        font-size: 16px;
        font-weight: 600;
        color: #F0F4FF;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: #0D1117 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #F0F4FF !important;
        padding: 12px !important;
        font-size: 14px !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #00D9B8 !important;
        box-shadow: 0 0 0 2px rgba(0,217,184,0.15) !important;
    }
    
    /* Labels */
    .stNumberInput > label,
    .stSelectbox > label {
        color: #F0F4FF !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D9B8, #0099FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,217,184,0.3) !important;
    }
    
    /* Results card */
    .results-card {
        background: linear-gradient(135deg, #0D1117, #111827);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 32px;
        margin-top: 32px;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 24px;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8892A4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Risk indicator */
    .risk-bar {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        overflow: hidden;
        margin: 16px 0;
    }
    
    .risk-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Helper text */
    .helper-text {
        font-size: 12px;
        color: #5A6478;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DISEASE CONFIGURATIONS ====================
DISEASE_CONFIG = {
    'diabetes': {
        'name': 'Diabetes',
        'icon': '🩸',
        'color': '#F59E0B',
        'fields': [
            {'name': 'Pregnancies', 'min': 0, 'max': 20, 'default': 0, 'help': 'Number of pregnancies (0-17)', 'type': 'number'},
            {'name': 'Glucose', 'min': 0, 'max': 300, 'default': 120, 'help': 'Plasma glucose (mg/dL). Normal: 70-140', 'type': 'number'},
            {'name': 'BloodPressure', 'min': 0, 'max': 200, 'default': 80, 'help': 'Diastolic BP (mm Hg). Normal: 60-90', 'type': 'number'},
            {'name': 'SkinThickness', 'min': 0, 'max': 100, 'default': 20, 'help': 'Triceps skin fold (mm). Normal: 10-40', 'type': 'number'},
            {'name': 'Insulin', 'min': 0, 'max': 1000, 'default': 79, 'help': '2-Hour serum insulin (µU/mL). Normal: 15-276', 'type': 'number'},
            {'name': 'BMI', 'min': 0, 'max': 80, 'default': 24.5, 'step': 0.1, 'help': 'Body Mass Index (kg/m²). Normal: 18.5-24.9', 'type': 'number'},
            {'name': 'DiabetesPedigreeFunction', 'min': 0, 'max': 3, 'default': 0.47, 'step': 0.01, 'help': 'Genetic influence score (0.08-2.42)', 'type': 'number'},
            {'name': 'Age', 'min': 21, 'max': 100, 'default': 33, 'help': 'Age in years (21-81)', 'type': 'number'},
        ]
    },
    'anemia': {
        'name': 'Anemia',
        'icon': '🔬',
        'color': '#F43F5E',
        'fields': [
            {'name': 'Hemoglobin', 'min': 0, 'max': 30, 'default': 13.5, 'step': 0.1, 'help': 'Hemoglobin (g/dL). Normal: F 12-16, M 13.5-17.5', 'type': 'number'},
            {'name': 'MCH', 'min': 0, 'max': 50, 'default': 30.0, 'step': 0.1, 'help': 'Mean Corpuscular Hemoglobin (pg). Normal: 27-33', 'type': 'number'},
            {'name': 'MCHC', 'min': 0, 'max': 50, 'default': 34.0, 'step': 0.1, 'help': 'Mean Corpuscular Hemoglobin Concentration (g/dL). Normal: 32-36', 'type': 'number'},
            {'name': 'MCV', 'min': 0, 'max': 150, 'default': 90.0, 'step': 0.1, 'help': 'Mean Corpuscular Volume (fL). Normal: 80-100', 'type': 'number'},
        ]
    },
    'kidney': {
        'name': 'Kidney Disease',
        'icon': '💜',
        'color': '#A78BFA',
        'fields': [
            {'name': 'age', 'min': 0, 'max': 120, 'default': 48, 'help': 'Age in years', 'type': 'number'},
            {'name': 'bp', 'min': 0, 'max': 200, 'default': 80, 'help': 'Blood Pressure (mm Hg). Normal: 70-90', 'type': 'number'},
            {'name': 'sg', 'min': 1.0, 'max': 1.03, 'default': 1.015, 'step': 0.001, 'help': 'Specific Gravity. Normal: 1.005-1.025', 'type': 'number'},
            {'name': 'al', 'min': 0, 'max': 5, 'default': 0, 'help': 'Albumin (0-5). 0=None, 5=Highest', 'type': 'number'},
            {'name': 'su', 'min': 0, 'max': 5, 'default': 0, 'help': 'Sugar (0-5). 0=None, 5=Highest', 'type': 'number'},
            {'name': 'bgr', 'min': 0, 'max': 500, 'default': 120, 'help': 'Blood Glucose Random (mg/dL). Normal: 70-140', 'type': 'number'},
            {'name': 'bu', 'min': 0, 'max': 200, 'default': 15, 'help': 'Blood Urea (mg/dL). Normal: 7-25', 'type': 'number'},
            {'name': 'sc', 'min': 0, 'max': 20, 'default': 1.0, 'step': 0.1, 'help': 'Serum Creatinine (mg/dL). Normal: 0.6-1.2', 'type': 'number'},
            {'name': 'sod', 'min': 0, 'max': 200, 'default': 140, 'help': 'Sodium (mEq/L). Normal: 135-145', 'type': 'number'},
            {'name': 'pot', 'min': 0, 'max': 20, 'default': 4.5, 'step': 0.1, 'help': 'Potassium (mEq/L). Normal: 3.5-5.0', 'type': 'number'},
            {'name': 'hemo', 'min': 0, 'max': 30, 'default': 15.0, 'step': 0.1, 'help': 'Hemoglobin (g/dL). Normal: 12-17', 'type': 'number'},
            {'name': 'rbc', 'min': 0, 'max': 1, 'default': 0, 'help': 'Red Blood Cell Count. 0=Normal, 1=Abnormal', 'type': 'select', 'options': ['Normal', 'Abnormal']},
            {'name': 'pc', 'min': 0, 'max': 1, 'default': 0, 'help': 'Pus Cell. 0=Normal, 1=Abnormal', 'type': 'select', 'options': ['Normal', 'Abnormal']},
            {'name': 'ba', 'min': 0, 'max': 1, 'default': 0, 'help': 'Bacteria. 0=Not Present, 1=Present', 'type': 'select', 'options': ['Not Present', 'Present']},
        ]
    },
    'liver': {
        'name': 'Liver Disease',
        'icon': '🫀',
        'color': '#10B981',
        'fields': [
            {'name': 'Age', 'min': 0, 'max': 120, 'default': 45, 'help': 'Age in years (4-90)', 'type': 'number'},
            {'name': 'Gender', 'min': 0, 'max': 1, 'default': 0, 'help': 'Gender: 0=Female, 1=Male', 'type': 'select', 'options': ['Female', 'Male']},
            {'name': 'Total_Bilirubin', 'min': 0, 'max': 100, 'default': 0.9, 'step': 0.1, 'help': 'Total Bilirubin (mg/dL). Normal: 0.1-1.2', 'type': 'number'},
            {'name': 'Direct_Bilirubin', 'min': 0, 'max': 50, 'default': 0.2, 'step': 0.1, 'help': 'Direct Bilirubin (mg/dL). Normal: 0.0-0.3', 'type': 'number'},
            {'name': 'Alkaline_Phosphotase', 'min': 0, 'max': 2500, 'default': 90, 'help': 'Alkaline Phosphatase (IU/L). Normal: 44-147', 'type': 'number'},
            {'name': 'Alamine_Aminotransferase', 'min': 0, 'max': 2500, 'default': 25, 'help': 'ALT/SGPT (U/L). Normal: 7-56', 'type': 'number'},
            {'name': 'Aspartate_Aminotransferase', 'min': 0, 'max': 2500, 'default': 20, 'help': 'AST/SGOT (U/L). Normal: 10-40', 'type': 'number'},
            {'name': 'Total_Protiens', 'min': 0, 'max': 15, 'default': 7.0, 'step': 0.1, 'help': 'Total Proteins (g/dL). Normal: 6.3-8.2', 'type': 'number'},
            {'name': 'Albumin', 'min': 0, 'max': 10, 'default': 4.2, 'step': 0.1, 'help': 'Albumin (g/dL). Normal: 3.5-5.5', 'type': 'number'},
            {'name': 'Albumin_and_Globulin_Ratio', 'min': 0, 'max': 5, 'default': 1.3, 'step': 0.1, 'help': 'A/G Ratio. Normal: 0.9-2.0', 'type': 'number'},
        ]
    },
    'hypothyroid': {
        'name': 'Hypothyroid',
        'icon': '🦋',
        'color': '#FF6B6B',
        'fields': [
            {'name': 'age', 'min': 0, 'max': 120, 'default': 45, 'help': 'Age in years', 'type': 'number'},
            {'name': 'sex', 'min': 0, 'max': 1, 'default': 0, 'help': 'Sex: 0=Female, 1=Male', 'type': 'select', 'options': ['Female', 'Male']},
            {'name': 'TSH', 'min': 0, 'max': 500, 'default': 2.5, 'step': 0.1, 'help': 'Thyroid Stimulating Hormone (mU/L). Normal: 0.4-4.0', 'type': 'number'},
            {'name': 'T3', 'min': 0, 'max': 20, 'default': 1.5, 'step': 0.1, 'help': 'Triiodothyronine (nmol/L). Normal: 1.2-2.8', 'type': 'number'},
            {'name': 'TT4', 'min': 0, 'max': 500, 'default': 100, 'help': 'Total Thyroxine (nmol/L). Normal: 70-150', 'type': 'number'},
            {'name': 'T4U', 'min': 0, 'max': 3, 'default': 1.0, 'step': 0.01, 'help': 'T4 Uptake. Normal: 0.7-1.8', 'type': 'number'},
            {'name': 'FTI', 'min': 0, 'max': 500, 'default': 100, 'help': 'Free Thyroxine Index. Normal: 60-180', 'type': 'number'},
        ]
    }
}

# ==================== HELPER FUNCTIONS ====================

def load_model_artifacts(disease_name):
    """Load all model artifacts for a given disease"""
    try:
        # Check if model files exist
        model_path = Path(f'{disease_name}_model.pkl')
        if not model_path.exists():
            return None
        
        with open(f'{disease_name}_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{disease_name}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{disease_name}_imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        with open(f'{disease_name}_feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open(f'{disease_name}_threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'feature_names': feature_names,
            'threshold': threshold
        }
    except Exception as e:
        st.error(f"Error loading model for {disease_name}: {str(e)}")
        return None

def predict_disease(patient_data, artifacts):
    """Make prediction using loaded artifacts"""
    try:
        # Convert to DataFrame with correct column order
        patient_df = pd.DataFrame([patient_data])[artifacts['feature_names']]
        
        # Apply preprocessing pipeline: impute → scale
        patient_imputed = pd.DataFrame(
            artifacts['imputer'].transform(patient_df),
            columns=artifacts['feature_names']
        )
        patient_scaled = artifacts['scaler'].transform(patient_imputed)
        
        # Get prediction and probability
        probability = artifacts['model'].predict_proba(patient_scaled)[0, 1]
        prediction = 1 if probability >= artifacts['threshold'] else 0
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
            risk_color = '#EF4444'
        elif probability >= 0.4:
            risk_level = 'MEDIUM'
            risk_color = '#F59E0B'
        else:
            risk_level = 'LOW'
            risk_color = '#10B981'
        
        return {
            'prediction': 'POSITIVE (Disease Detected)' if prediction == 1 else 'NEGATIVE (No Disease)',
            'confidence': round(probability * 100, 2),
            'probability': probability,
            'threshold': artifacts['threshold'],
            'risk_level': risk_level,
            'risk_color': risk_color,
            'is_positive': prediction == 1
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="hero-tag">
            🏥 AI-POWERED MEDICAL DIAGNOSTICS
        </div>
        <h1 class="main-title">
            MediScan <span class="gradient-text">AI</span>
        </h1>
        <p class="subtitle">
            Advanced machine learning models for rapid disease screening and risk assessment
        </p>
        <div class="disclaimer">
            ⚠️ For screening purposes only — consult a healthcare professional for diagnosis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disease Selection
    st.markdown("### 🔍 Select Disease to Screen")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    disease_buttons = {
        'diabetes': col1,
        'anemia': col2,
        'kidney': col3,
        'liver': col4,
        'hypothyroid': col5
    }
    
    selected_disease = None
    
    for disease_key, col in disease_buttons.items():
        with col:
            config = DISEASE_CONFIG[disease_key]
            if st.button(f"{config['icon']} {config['name']}", key=f"btn_{disease_key}", use_container_width=True):
                st.session_state.selected_disease = disease_key
    
    # Check if a disease is selected
    if 'selected_disease' not in st.session_state:
        st.info("👆 Please select a disease from the options above to begin screening")
        return
    
    selected_disease = st.session_state.selected_disease
    config = DISEASE_CONFIG[selected_disease]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display selected disease header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0D1117, #111827); 
                border: 1px solid rgba(255,255,255,0.1); 
                border-radius: 16px; 
                padding: 24px; 
                margin-bottom: 24px;">
        <h2 style="color: {config['color']}; margin: 0;">
            {config['icon']} {config['name']} Prediction
        </h2>
        <p style="color: #8892A4; margin: 8px 0 0 0; font-size: 14px;">
            Enter patient laboratory values below for AI analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    st.markdown("### 📋 Patient Information")
    
    # Create input fields based on disease configuration
    patient_data = {}
    
    # Organize fields in columns (3 per row)
    fields = config['fields']
    num_cols = 3
    
    for i in range(0, len(fields), num_cols):
        cols = st.columns(num_cols)
        for j, field in enumerate(fields[i:i+num_cols]):
            with cols[j]:
                if field.get('type') == 'select':
                    # Select box for categorical fields
                    selected_option = st.selectbox(
                        field['name'],
                        options=field['options'],
                        help=field['help'],
                        key=f"{selected_disease}_{field['name']}"
                    )
                    # Convert to numeric (0, 1)
                    patient_data[field['name']] = field['options'].index(selected_option)
                else:
                    # Number input for numeric fields
                    patient_data[field['name']] = st.number_input(
                        field['name'],
                        min_value=float(field['min']),
                        max_value=float(field['max']),
                        value=float(field['default']),
                        step=field.get('step', 1.0),
                        help=field['help'],
                        key=f"{selected_disease}_{field['name']}"
                    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    if st.button("⚡ Run AI Prediction", use_container_width=True, type="primary"):
        # Load model artifacts
        with st.spinner("🔄 Loading AI model..."):
            artifacts = load_model_artifacts(selected_disease)
        
        if artifacts is None:
            st.error(f"""
            ❌ **Model files not found for {config['name']}**
            
            Please ensure the following files exist in the current directory:
            - `{selected_disease}_model.pkl`
            - `{selected_disease}_scaler.pkl`
            - `{selected_disease}_imputer.pkl`
            - `{selected_disease}_feature_names.pkl`
            - `{selected_disease}_threshold.pkl`
            
            You need to run the Jupyter notebook first to train and save the models.
            """)
            return
        
        # Make prediction
        with st.spinner("🧠 Analyzing patient data..."):
            result = predict_disease(patient_data, artifacts)
        
        if result is None:
            return
        
        # Display Results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="results-card">
            <h2 class="result-title" style="color: {config['color']};">
                📊 Analysis Results — {config['name']}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {result['risk_color']};">
                <div class="metric-label">Prediction</div>
                <div class="metric-value" style="color: {result['risk_color']};">
                    {'POSITIVE' if result['is_positive'] else 'NEGATIVE'}
                </div>
                <p style="font-size: 12px; color: #8892A4; margin-top: 8px;">
                    {result['prediction']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {config['color']};">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value" style="color: {config['color']};">
                    {result['confidence']}%
                </div>
                <p style="font-size: 12px; color: #8892A4; margin-top: 8px;">
                    Model certainty level
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {result['risk_color']};">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="color: {result['risk_color']};">
                    {result['risk_level']}
                </div>
                <p style="font-size: 12px; color: #8892A4; margin-top: 8px;">
                    Overall assessment
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Progress Bar
        st.markdown("<br>", unsafe_allow_html=True)
        
        risk_percentage = int(result['probability'] * 100)
        gradient_color = f"linear-gradient(90deg, #10B981, {result['risk_color']})"
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); 
                    border: 1px solid rgba(255,255,255,0.07); 
                    border-radius: 12px; 
                    padding: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span style="font-size: 14px; color: #8892A4;">DISEASE PROBABILITY</span>
                <span style="font-size: 14px; font-weight: 600; color: {result['risk_color']};">
                    {risk_percentage}%
                </span>
            </div>
            <div class="risk-bar">
                <div class="risk-fill" style="width: {risk_percentage}%; background: {gradient_color};"></div>
            </div>
            <p style="font-size: 12px; color: #5A6478; margin-top: 12px;">
                Decision threshold: {round(result['threshold'] * 100, 1)}% 
                | Ensemble ML model prediction
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        
        if result['is_positive']:
            st.warning(f"""
            **⚠️ Elevated Risk Detected**
            
            The AI model indicates a potential risk for **{config['name']}** based on the provided parameters. 
            
            **Recommended Actions:**
            - Consult with a healthcare professional immediately
            - Schedule comprehensive laboratory testing
            - Discuss these results with your doctor
            - Do not self-diagnose or self-medicate
            
            *This is a screening tool only and not a substitute for professional medical advice.*
            """)
        else:
            st.success(f"""
            **✅ Low Risk Assessment**
            
            The AI model indicates low risk for **{config['name']}** based on current parameters.
            
            **Recommendations:**
            - Continue regular health monitoring
            - Maintain healthy lifestyle practices
            - Schedule routine check-ups as recommended
            - Stay aware of any symptom changes
            
            *Regular medical check-ups are important even with low risk assessments.*
            """)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #5A6478; font-size: 12px;">
        <p>
            🏥 MediScan AI | Powered by Machine Learning Ensemble Models<br>
            Random Forest • XGBoost • LightGBM
        </p>
        <p style="margin-top: 8px;">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. 
            Always consult qualified healthcare professionals for medical advice and diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
