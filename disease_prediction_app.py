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

# ==================== CUSTOM CSS (EXACT HTML MATCH) ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    /* Dark theme background */
    .stApp {
        background: #06080F;
        color: #F0F4FF;
        overflow-x: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background Orbs & Grid */
    .bg-orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(120px);
        pointer-events: none;
        z-index: 0;
        animation: orbFloat 8s ease-in-out infinite;
    }
    .bg-orb-1 {
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(0,217,184,0.07), transparent 70%);
        top: -150px;
        left: -100px;
    }
    .bg-orb-2 {
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(167,139,250,0.06), transparent 70%);
        bottom: -100px;
        right: -100px;
        animation-delay: -4s;
    }
    .bg-grid {
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -20px) scale(1.05); }
        66% { transform: translate(-20px, 30px) scale(0.97); }
    }
    
    /* Navigation Bar */
    .nav-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background: rgba(6, 8, 15, 0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        padding: 0;
        margin: 0;
    }
    .nav-inner {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 64px;
    }
    .logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: -0.5px;
        text-decoration: none;
        color: #F0F4FF;
    }
    .logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #00D9B8, #0099FF);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .logo-text { color: #00D9B8; }
    .nav-badge {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 3px 8px;
        border-radius: 20px;
        background: rgba(0, 217, 184, 0.15);
        color: #00D9B8;
        border: 1px solid rgba(0, 217, 184, 0.3);
        text-transform: uppercase;
    }
    .nav-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .nav-chip {
        font-size: 13px;
        color: #8892A4;
        background: #0D1117;
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 6px 14px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-chip:hover {
        color: #F0F4FF;
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Hero Section */
    .hero {
        padding: 80px 0 48px;
        text-align: center;
        position: relative;
        z-index: 1;
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
        background: rgba(0, 217, 184, 0.15);
        border: 1px solid rgba(0, 217, 184, 0.25);
        padding: 6px 16px;
        border-radius: 20px;
        margin-bottom: 28px;
        animation: fadeUp 0.5s ease both;
    }
    .hero-tag::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #00D9B8;
        animation: pulse 2s ease infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.7); }
    }
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-title {
        font-size: clamp(36px, 6vw, 68px);
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -2px;
        margin: 20px 0;
        animation: fadeUp 0.5s ease 0.1s both;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00D9B8 0%, #0099FF 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-sub {
        margin-top: 20px;
        font-size: 18px;
        font-weight: 400;
        color: #8892A4;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
        animation: fadeUp 0.5s ease 0.2s both;
    }
    
    .disclaimer {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 24px;
        font-size: 12px;
        color: #5A6478;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 8px 16px;
        border-radius: 8px;
        animation: fadeUp 0.5s ease 0.3s both;
    }
    
    /* Section Label */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8892A4;
        margin-bottom: 16px;
        margin-top: 40px;
        animation: fadeUp 0.5s ease 0.35s both;
    }
    
    /* Disease Cards */
    .disease-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 40px;
        animation: fadeUp 0.5s ease 0.4s both;
    }
    
    .disease-card {
        position: relative;
        overflow: hidden;
        background: #0D1117;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px 16px;
        cursor: pointer;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
    }
    
    .disease-card.active {
        border-color: var(--card-color);
        box-shadow: 0 0 20px var(--card-glow);
    }
    
    .disease-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--card-color);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .disease-card.active::before {
        opacity: 1;
    }
    
    .disease-icon {
        font-size: 36px;
        margin-bottom: 12px;
        display: block;
    }
    
    .disease-name {
        font-size: 14px;
        font-weight: 600;
        color: #F0F4FF;
    }
    
    /* Form Panel */
    .form-panel {
        background: linear-gradient(135deg, #0D1117, #111827);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 0;
        margin-top: 32px;
        overflow: hidden;
        animation: fadeUp 0.5s ease 0.5s both;
    }
    
    .form-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 24px 28px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .form-header-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .form-header-icon {
        width: 48px;
        height: 48px;
        background: var(--panel-color);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        opacity: 0.9;
    }
    
    .fh-title {
        font-size: 20px;
        font-weight: 700;
        color: #F0F4FF;
        margin-bottom: 4px;
    }
    
    .fh-sub {
        font-size: 13px;
        color: #8892A4;
    }
    
    .fh-fields-count {
        font-size: 12px;
        color: #8892A4;
        background: rgba(255, 255, 255, 0.03);
        padding: 6px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: #0D1117 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #F0F4FF !important;
        padding: 12px !important;
        font-size: 14px !important;
        transition: all 0.2s !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--input-color, #00D9B8) !important;
        box-shadow: 0 0 0 3px var(--input-glow, rgba(0, 217, 184, 0.15)) !important;
        outline: none !important;
    }
    
    /* Labels */
    .stNumberInput > label,
    .stSelectbox > label {
        color: #F0F4FF !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    
    /* Field helper text */
    .field-helper {
        font-size: 11px;
        color: #5A6478;
        margin-top: 6px;
        padding-left: 2px;
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
        box-shadow: 0 8px 24px rgba(0, 217, 184, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Disease selector buttons */
    .disease-btn {
        background: #0D1117 !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: #F0F4FF !important;
        border-radius: 16px !important;
        padding: 20px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .disease-btn:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-4px) !important;
    }
    
    /* Action Bar */
    .action-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 24px 28px;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        background: rgba(255, 255, 255, 0.02);
    }
    
    .action-info {
        font-size: 13px;
        color: #8892A4;
    }
    
    /* Results Card */
    .results-card {
        background: linear-gradient(135deg, #0D1117, #111827);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 32px;
        margin-top: 32px;
        animation: fadeUp 0.5s ease both;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .result-badge {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid var(--metric-color);
    }
    
    .metric-label {
        font-size: 11px;
        color: #8892A4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .metric-sub {
        font-size: 12px;
        color: #8892A4;
    }
    
    /* Risk Bar */
    .risk-bar-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .risk-bar-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    
    .risk-bar-label {
        font-size: 11px;
        color: #8892A4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-bar-percent {
        font-size: 14px;
        font-weight: 600;
    }
    
    .risk-bar {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        overflow: hidden;
        margin: 12px 0;
    }
    
    .risk-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .risk-bar-sub {
        font-size: 11px;
        color: #5A6478;
        margin-top: 12px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #5A6478;
        font-size: 12px;
        margin-top: 60px;
    }
    
    .footer-title {
        margin-bottom: 8px;
    }
    
    .footer-disclaimer {
        margin-top: 12px;
    }
    
    /* Responsive */
    @media (max-width: 1024px) {
        .disease-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .disease-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .main-title {
            font-size: 36px;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0D1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DISEASE CONFIGURATIONS ====================
DISEASE_CONFIG = {
    'diabetes': {
        'name': 'Diabetes',
        'icon': '🩸',
        'color': '#F59E0B',
        'glow': 'rgba(245, 158, 11, 0.15)',
        'fields': [
            {'name': 'Pregnancies', 'min': 0, 'max': 20, 'default': 0, 'help': 'Number of pregnancies', 'range': 'Range: 0–17', 'type': 'number'},
            {'name': 'Glucose', 'min': 0, 'max': 250, 'default': 120, 'help': 'Plasma glucose concentration', 'range': 'Normal: 70–140 mg/dL', 'type': 'number'},
            {'name': 'BloodPressure', 'min': 0, 'max': 150, 'default': 70, 'help': 'Diastolic blood pressure', 'range': 'Normal: 60–90 mm Hg', 'type': 'number'},
            {'name': 'SkinThickness', 'min': 0, 'max': 100, 'default': 20, 'help': 'Triceps skin fold thickness', 'range': 'Normal: 10–40 mm', 'type': 'number'},
            {'name': 'Insulin', 'min': 0, 'max': 900, 'default': 79, 'help': 'Serum insulin level', 'range': 'Normal: 15–276 µU/mL', 'type': 'number'},
            {'name': 'BMI', 'min': 0, 'max': 70, 'default': 24.5, 'step': 0.1, 'help': 'Body mass index', 'range': 'Normal: 18.5–24.9 kg/m²', 'type': 'number'},
            {'name': 'DiabetesPedigreeFunction', 'min': 0.0, 'max': 3.0, 'default': 0.47, 'step': 0.01, 'help': 'Genetic risk score', 'range': 'Range: 0.08–2.42', 'type': 'number'},
            {'name': 'Age', 'min': 21, 'max': 100, 'default': 33, 'help': 'Age in years', 'range': 'Adults 21–81 years', 'type': 'number'},
        ]
    },
    'anemia': {
        'name': 'Anemia',
        'icon': '🔬',
        'color': '#F43F5E',
        'glow': 'rgba(244, 63, 94, 0.15)',
        'fields': [
            {'name': 'Hemoglobin', 'min': 0, 'max': 25, 'default': 13.5, 'step': 0.1, 'help': 'Hemoglobin level', 'range': 'Normal: F 12–16, M 13.5–17.5 g/dL', 'type': 'number'},
            {'name': 'MCH', 'min': 0, 'max': 50, 'default': 30, 'step': 0.1, 'help': 'Mean corpuscular hemoglobin', 'range': 'Normal: 27–33 pg', 'type': 'number'},
            {'name': 'MCHC', 'min': 0, 'max': 50, 'default': 34, 'step': 0.1, 'help': 'Mean corpuscular hemoglobin concentration', 'range': 'Normal: 32–36 g/dL', 'type': 'number'},
            {'name': 'MCV', 'min': 0, 'max': 150, 'default': 90, 'step': 0.1, 'help': 'Mean corpuscular volume', 'range': 'Normal: 80–100 fL', 'type': 'number'},
        ]
    },
    'kidney': {
        'name': 'Kidney Disease',
        'icon': '💜',
        'color': '#A78BFA',
        'glow': 'rgba(167, 139, 250, 0.15)',
        'fields': [
            {'name': 'Age', 'min': 1, 'max': 100, 'default': 48, 'help': 'Age in years', 'range': 'Adults', 'type': 'number'},
            {'name': 'BloodPressure', 'min': 0, 'max': 180, 'default': 80, 'help': 'Blood pressure', 'range': 'Normal: 70–90 mm Hg', 'type': 'number'},
            {'name': 'SpecificGravity', 'min': 1.0, 'max': 1.03, 'default': 1.015, 'step': 0.001, 'help': 'Urine specific gravity', 'range': 'Normal: 1.005–1.025', 'type': 'number'},
            {'name': 'Albumin', 'min': 0, 'max': 5, 'default': 0, 'help': 'Albumin level (0-5 scale)', 'range': '0=None, 5=Highest', 'type': 'number'},
            {'name': 'Sugar', 'min': 0, 'max': 5, 'default': 0, 'help': 'Sugar level (0-5 scale)', 'range': '0=None, 5=Highest', 'type': 'number'},
            {'name': 'BloodGlucoseRandom', 'min': 0, 'max': 500, 'default': 120, 'help': 'Random blood glucose', 'range': 'Normal: 70–140 mg/dL', 'type': 'number'},
            {'name': 'BloodUrea', 'min': 0, 'max': 200, 'default': 15, 'help': 'Blood urea nitrogen', 'range': 'Normal: 7–25 mg/dL', 'type': 'number'},
            {'name': 'SerumCreatinine', 'min': 0, 'max': 15, 'default': 1.0, 'step': 0.1, 'help': 'Serum creatinine', 'range': 'Normal: 0.6–1.2 mg/dL', 'type': 'number'},
            {'name': 'Sodium', 'min': 0, 'max': 200, 'default': 140, 'help': 'Sodium level', 'range': 'Normal: 135–145 mEq/L', 'type': 'number'},
            {'name': 'Potassium', 'min': 0, 'max': 10, 'default': 4.5, 'step': 0.1, 'help': 'Potassium level', 'range': 'Normal: 3.5–5.0 mEq/L', 'type': 'number'},
            {'name': 'Hemoglobin', 'min': 0, 'max': 25, 'default': 15, 'step': 0.1, 'help': 'Hemoglobin level', 'range': 'Normal: 12–17 g/dL', 'type': 'number'},
            {'name': 'RedBloodCells', 'min': 0, 'max': 1, 'default': 0, 'help': 'Red blood cell count status', 'range': '0=Normal, 1=Abnormal', 'type': 'select', 'options': ['Normal', 'Abnormal']},
            {'name': 'PusCells', 'min': 0, 'max': 1, 'default': 0, 'help': 'Pus cell status', 'range': '0=Normal, 1=Abnormal', 'type': 'select', 'options': ['Normal', 'Abnormal']},
            {'name': 'Bacteria', 'min': 0, 'max': 1, 'default': 0, 'help': 'Bacteria presence', 'range': '0=Not Present, 1=Present', 'type': 'select', 'options': ['Not Present', 'Present']},
        ]
    },
    'liver': {
        'name': 'Liver Disease',
        'icon': '🫀',
        'color': '#10B981',
        'glow': 'rgba(16, 185, 129, 0.15)',
        'fields': [
            {'name': 'Age', 'min': 4, 'max': 90, 'default': 45, 'help': 'Age in years', 'range': 'Adults 4–90 years', 'type': 'number'},
            {'name': 'Gender', 'min': 0, 'max': 1, 'default': 1, 'help': 'Gender', 'range': '0=Female, 1=Male', 'type': 'select', 'options': ['Female', 'Male']},
            {'name': 'TotalBilirubin', 'min': 0, 'max': 75, 'default': 0.9, 'step': 0.1, 'help': 'Total bilirubin level', 'range': 'Normal: 0.1–1.2 mg/dL', 'type': 'number'},
            {'name': 'DirectBilirubin', 'min': 0, 'max': 20, 'default': 0.2, 'step': 0.1, 'help': 'Direct bilirubin level', 'range': 'Normal: 0.0–0.3 mg/dL', 'type': 'number'},
            {'name': 'AlkalinePhosphatase', 'min': 0, 'max': 2500, 'default': 90, 'help': 'Alkaline phosphatase level', 'range': 'Normal: 44–147 IU/L', 'type': 'number'},
            {'name': 'AlaninAminotransferase', 'min': 0, 'max': 2500, 'default': 25, 'help': 'ALT / SGPT level', 'range': 'Normal: 7–56 U/L', 'type': 'number'},
            {'name': 'AspartateAminotransferase', 'min': 0, 'max': 4500, 'default': 20, 'help': 'AST / SGOT level', 'range': 'Normal: 10–40 U/L', 'type': 'number'},
            {'name': 'TotalProtiens', 'min': 0, 'max': 15, 'default': 7.0, 'step': 0.1, 'help': 'Total proteins', 'range': 'Normal: 6.3–8.2 g/dL', 'type': 'number'},
            {'name': 'Albumin', 'min': 0, 'max': 10, 'default': 4.2, 'step': 0.1, 'help': 'Albumin level', 'range': 'Normal: 3.5–5.5 g/dL', 'type': 'number'},
            {'name': 'AlbuminGlobulinRatio', 'min': 0, 'max': 5, 'default': 1.3, 'step': 0.1, 'help': 'A/G ratio', 'range': 'Normal: 0.9–2.0', 'type': 'number'},
        ]
    },
    'hypothyroid': {
        'name': 'Hypothyroid',
        'icon': '🦋',
        'color': '#0099FF',
        'glow': 'rgba(0, 153, 255, 0.15)',
        'fields': [
            {'name': 'Age', 'min': 0, 'max': 120, 'default': 40, 'help': 'Age in years', 'range': 'All ages', 'type': 'number'},
            {'name': 'Sex', 'min': 0, 'max': 1, 'default': 0, 'help': 'Sex', 'range': '0=Female, 1=Male', 'type': 'select', 'options': ['Female', 'Male']},
            {'name': 'OnThyroxine', 'min': 0, 'max': 1, 'default': 0, 'help': 'On thyroxine treatment', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'QueryOnThyroxine', 'min': 0, 'max': 1, 'default': 0, 'help': 'Query on thyroxine', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'OnAntithyroidMedication', 'min': 0, 'max': 1, 'default': 0, 'help': 'On antithyroid medication', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Sick', 'min': 0, 'max': 1, 'default': 0, 'help': 'Currently sick', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Pregnant', 'min': 0, 'max': 1, 'default': 0, 'help': 'Currently pregnant', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'ThyroidSurgery', 'min': 0, 'max': 1, 'default': 0, 'help': 'History of thyroid surgery', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'I131Treatment', 'min': 0, 'max': 1, 'default': 0, 'help': 'I131 treatment', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'QueryHypothyroid', 'min': 0, 'max': 1, 'default': 0, 'help': 'Query hypothyroid', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'QueryHyperthyroid', 'min': 0, 'max': 1, 'default': 0, 'help': 'Query hyperthyroid', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Lithium', 'min': 0, 'max': 1, 'default': 0, 'help': 'On lithium', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Goitre', 'min': 0, 'max': 1, 'default': 0, 'help': 'Goitre present', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Tumor', 'min': 0, 'max': 1, 'default': 0, 'help': 'Tumor present', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Hypopituitary', 'min': 0, 'max': 1, 'default': 0, 'help': 'Hypopituitary condition', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'Psych', 'min': 0, 'max': 1, 'default': 0, 'help': 'Psychological condition', 'range': '0=No, 1=Yes', 'type': 'select', 'options': ['No', 'Yes']},
            {'name': 'TSH', 'min': 0, 'max': 500, 'default': 2.5, 'step': 0.1, 'help': 'TSH level', 'range': 'Normal: 0.4–4.0 mU/L', 'type': 'number'},
            {'name': 'T3', 'min': 0, 'max': 20, 'default': 1.5, 'step': 0.1, 'help': 'T3 level', 'range': 'Normal: 0.8–2.0 nmol/L', 'type': 'number'},
            {'name': 'TT4', 'min': 0, 'max': 500, 'default': 110, 'help': 'Total T4 level', 'range': 'Normal: 70–150 nmol/L', 'type': 'number'},
            {'name': 'T4U', 'min': 0, 'max': 3, 'default': 1.0, 'step': 0.01, 'help': 'T4 uptake', 'range': 'Normal: 0.7–1.4', 'type': 'number'},
            {'name': 'FTI', 'min': 0, 'max': 500, 'default': 110, 'help': 'Free thyroxine index', 'range': 'Normal: 64–155', 'type': 'number'},
        ]
    }
}

# ==================== HELPER FUNCTIONS ====================
def load_model_artifacts(disease):
    """Load all required model artifacts for a disease"""
    try:
        current_dir = Path.cwd()
        
        model_path = current_dir / f"{disease}_model.pkl"
        scaler_path = current_dir / f"{disease}_scaler.pkl"
        imputer_path = current_dir / f"{disease}_imputer.pkl"
        features_path = current_dir / f"{disease}_feature_names.pkl"
        threshold_path = current_dir / f"{disease}_threshold.pkl"
        
        # Check if all files exist
        if not all([model_path.exists(), scaler_path.exists(), imputer_path.exists(), 
                   features_path.exists(), threshold_path.exists()]):
            return None
        
        # Load all artifacts
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'feature_names': feature_names,
            'threshold': threshold
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None

def predict_disease(patient_data, artifacts):
    """Make prediction using loaded model artifacts"""
    try:
        # Extract artifacts
        model = artifacts['model']
        scaler = artifacts['scaler']
        imputer = artifacts['imputer']
        feature_names = artifacts['feature_names']
        threshold = artifacts['threshold']
        
        # Create DataFrame with exact feature names
        input_df = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Preprocess: impute then scale
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Get prediction probability
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Make prediction using optimal threshold
        is_positive = probability >= threshold
        
        # Calculate confidence (distance from threshold)
        confidence = abs(probability - threshold) * 100
        confidence = min(max(confidence * 2, 70), 99)  # Scale to 70-99%
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
            risk_color = "#10B981"
        elif probability < 0.7:
            risk_level = "MODERATE"
            risk_color = "#F59E0B"
        else:
            risk_level = "HIGH"
            risk_color = "#EF4444"
        
        return {
            'is_positive': is_positive,
            'probability': probability,
            'confidence': round(confidence, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'threshold': threshold,
            'prediction': 'Elevated Risk Detected' if is_positive else 'Within Normal Bounds'
        }
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ==================== MAIN APPLICATION ====================
def main():
    # Background elements
    st.markdown("""
    <div class="bg-orb bg-orb-1"></div>
    <div class="bg-orb bg-orb-2"></div>
    <div class="bg-grid"></div>
    """, unsafe_allow_html=True)
    
    # Navigation Bar
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-inner">
            <div class="logo">
                <div class="logo-icon">🏥</div>
                <div>
                    Medi<span class="logo-text">Scan</span>
                    <span class="nav-badge">Beta</span>
                </div>
            </div>
            <div class="nav-right">
                <div class="nav-chip">v1.2</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">
            AI-POWERED MEDICAL SCREENING
        </div>
        <h1 class="main-title">
            Advanced Disease<br><span class="gradient-text">Prediction System</span>
        </h1>
        <p class="hero-sub">
            Leverage ensemble machine learning models to screen for multiple diseases using laboratory biomarkers. Built with Random Forest, XGBoost, and LightGBM.
        </p>
        <div class="disclaimer">
            ⚠️ For educational & screening purposes only — not a diagnostic tool
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disease Selection Section
    st.markdown("""
    <div class="section-label">🔍 Select Disease to Screen</div>
    """, unsafe_allow_html=True)
    
    # Disease selector cards
    cols = st.columns(5)
    
    disease_buttons = {
        'diabetes': (cols[0], '#F59E0B', 'rgba(245, 158, 11, 0.15)'),
        'anemia': (cols[1], '#F43F5E', 'rgba(244, 63, 94, 0.15)'),
        'kidney': (cols[2], '#A78BFA', 'rgba(167, 139, 250, 0.15)'),
        'liver': (cols[3], '#10B981', 'rgba(16, 185, 129, 0.15)'),
        'hypothyroid': (cols[4], '#0099FF', 'rgba(0, 153, 255, 0.15)')
    }
    
    for disease_key, (col, color, glow) in disease_buttons.items():
        with col:
            config = DISEASE_CONFIG[disease_key]
            card_class = 'active' if st.session_state.get('selected_disease') == disease_key else ''
            
            st.markdown(f"""
            <style>
                .dc-{disease_key} {{
                    --card-color: {color};
                    --card-glow: {glow};
                }}
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="disease-card dc-{disease_key} {card_class}">
                <span class="disease-icon">{config['icon']}</span>
                <div class="disease-name">{config['name']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select {config['name']}", key=f"btn_{disease_key}", use_container_width=True, type="secondary"):
                st.session_state.selected_disease = disease_key
                st.rerun()
    
    # Check if a disease is selected
    if 'selected_disease' not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👆 Please select a disease from the options above to begin screening")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <div class="footer-title">
                🏥 MediScan AI | Powered by Machine Learning Ensemble Models<br>
                Random Forest • XGBoost • LightGBM
            </div>
            <div class="footer-disclaimer">
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. 
                Always consult qualified healthcare professionals for medical advice and diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    selected_disease = st.session_state.selected_disease
    config = DISEASE_CONFIG[selected_disease]
    
    # CSS variables for the selected disease
    st.markdown(f"""
    <style>
        :root {{
            --panel-color: {config['color']};
            --panel-glow: {config['glow']};
            --input-color: {config['color']};
            --input-glow: {config['glow']};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Form Panel Header
    st.markdown(f"""
    <div class="form-panel">
        <div class="form-header">
            <div class="form-header-left">
                <div class="form-header-icon">{config['icon']}</div>
                <div>
                    <div class="fh-title">{config['name']} Prediction</div>
                    <div class="fh-sub">Enter patient lab report values below</div>
                </div>
            </div>
            <span class="fh-fields-count">{len(config['fields'])} parameters</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
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
                    
                    # Display range info
                    if 'range' in field:
                        st.markdown(f'<div class="field-helper">{field["range"]}</div>', unsafe_allow_html=True)
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
                    
                    # Display range info
                    if 'range' in field:
                        st.markdown(f'<div class="field-helper">{field["range"]}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Bar
    st.markdown("""
    <div class="action-bar">
        <div class="action-info">Model ready — fill in all values to predict</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        else:
            # Make prediction
            with st.spinner("🧠 Analyzing patient data..."):
                result = predict_disease(patient_data, artifacts)
            
            if result is not None:
                # Display Results
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="results-card">
                    <div class="result-title" style="color: {config['color']};">
                        📊 Analysis Results
                        <span class="result-badge" style="color: {config['color']}; border-color: {config['color']};">{config['name']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="--metric-color: {result['risk_color']};">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value" style="color: {result['risk_color']};">
                            {result['risk_level']}
                        </div>
                        <div class="metric-sub">{result['prediction']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="--metric-color: {config['color']};">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="color: {config['color']};">
                            {result['confidence']}%
                        </div>
                        <div class="metric-sub">Model certainty level</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    prediction_text = f"Possible {config['name']}" if result['is_positive'] else f"No {config['name']} Detected"
                    st.markdown(f"""
                    <div class="metric-card" style="--metric-color: {result['risk_color']};">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value" style="color: {result['risk_color']}; font-size: 18px;">
                            {prediction_text}
                        </div>
                        <div class="metric-sub">ML ensemble output</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk Progress Bar
                risk_percentage = int(result['probability'] * 100)
                gradient_color = f"linear-gradient(90deg, #10B981, {result['risk_color']})"
                
                st.markdown(f"""
                <div class="risk-bar-container">
                    <div class="risk-bar-header">
                        <span class="risk-bar-label">Disease Probability</span>
                        <span class="risk-bar-percent" style="color: {result['risk_color']};">{risk_percentage}%</span>
                    </div>
                    <div class="risk-bar">
                        <div class="risk-fill" style="width: {risk_percentage}%; background: {gradient_color};"></div>
                    </div>
                    <div class="risk-bar-sub">
                        Decision threshold: {round(result['threshold'] * 100, 1)}% | Ensemble ML model prediction
                    </div>
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
    st.markdown("""
    <div class="footer">
        <div class="footer-title">
            🏥 MediScan AI | Powered by Machine Learning Ensemble Models<br>
            Random Forest • XGBoost • LightGBM
        </div>
        <div class="footer-disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. 
            Always consult qualified healthcare professionals for medical advice and diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
