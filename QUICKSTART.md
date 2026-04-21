# 🚀 Quick Start Guide - MediScan AI

## ⚡ 3-Step Setup (First Time Only)

### Step 1: Install Requirements
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
```

### Step 2: Train Models (ONE TIME ONLY)
1. Open `UPDATED_Disease_PREDICTION.ipynb` in Jupyter or Google Colab
2. Upload your 5 CSV dataset files when prompted:
   - anemia.csv
   - diabetes.csv
   - hypothyroid.csv
   - kidney_disease.csv
   - Liver Patient Dataset (LPD)_train.csv
3. Run all cells (this will take 5-10 minutes)
4. Wait for training to complete
5. **25 .pkl files** will be created in your directory

### Step 3: Launch the App
```bash
streamlit run disease_prediction_app.py
```

Your browser will open automatically at `http://localhost:8501` 🎉

---

## 🎯 Daily Usage (After Setup)

Just run:
```bash
streamlit run disease_prediction_app.py
```

---

## ✅ Checklist Before First Run

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] 5 CSV dataset files available
- [ ] Jupyter notebook executed successfully
- [ ] 25 .pkl model files created
- [ ] All files in the same directory

---

## 🔍 Verify Installation

Check if model files exist:
```bash
ls *.pkl
```

You should see:
```
anemia_model.pkl          liver_model.pkl
anemia_scaler.pkl         liver_scaler.pkl
anemia_imputer.pkl        liver_imputer.pkl
anemia_feature_names.pkl  liver_feature_names.pkl
anemia_threshold.pkl      liver_threshold.pkl

diabetes_model.pkl        hypothyroid_model.pkl
diabetes_scaler.pkl       hypothyroid_scaler.pkl
diabetes_imputer.pkl      hypothyroid_imputer.pkl
diabetes_feature_names.pkl hypothyroid_feature_names.pkl
diabetes_threshold.pkl    hypothyroid_threshold.pkl

kidney_model.pkl
kidney_scaler.pkl
kidney_imputer.pkl
kidney_feature_names.pkl
kidney_threshold.pkl
```

---

## ❓ Troubleshooting

**"Model files not found" error?**
→ Run the Jupyter notebook first to train models

**Import errors?**
→ Run: `pip install -r requirements.txt`

**Port already in use?**
→ Run: `streamlit run disease_prediction_app.py --server.port 8502`

---

## 🎨 How to Use the App

1. **Select a disease** - Click one of the 5 disease cards
2. **Enter patient data** - Fill in the laboratory values
3. **Click "Run AI Prediction"** - Get instant results
4. **View analysis** - See prediction, confidence, and recommendations

---

## 📊 Sample Test Data (Diabetes)

Try these values for a quick test:

- Pregnancies: 6
- Glucose: 148
- Blood Pressure: 72
- Skin Thickness: 35
- Insulin: 0
- BMI: 33.6
- Diabetes Pedigree Function: 0.627
- Age: 50

Expected Result: **POSITIVE** with ~72% confidence

---

**Need more help?** Read the full `README.md`

**Ready to start?** Run `streamlit run disease_prediction_app.py` 🚀
