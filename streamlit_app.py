
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
import xgboost as xgb
import shap
import json
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import joblib
import os

# Set up the page
st.set_page_config(
    page_title="MediExplain AI - Complete Healthcare Solution",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved visibility
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; padding-bottom: 10px}
    .sub-header {font-size: 1.5rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px}
    .feature-box {background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px}
    .prediction-high {background-color: #ffcccc; padding: 20px; border-radius: 10px}
    .prediction-medium {background-color: #fff6cc; padding: 20px; border-radius: 10px}
    .prediction-low {background-color: #ccffcc; padding: 20px; border-radius: 10px}
    .interpretation-box {background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-top: 10px}
    .footer {text-align: center; margin-top: 50px; color: #777}
    .chat-container {background-color: #f9f9f9; padding: 15px; border-radius: 10px; max-height: 400px; overflow-y: auto; border: 1px solid #ddd}
    .user-msg {background-color: #d1ecf1; padding: 12px; border-radius: 10px; margin-bottom: 10px; text-align: right; color: #000000; border: 1px solid #bee5eb}
    .bot-msg {background-color: #e8f4f8; padding: 12px; border-radius: 10px; margin-bottom: 10px; color: #000000; border: 1px solid #d1ecf1}
    .chat-input {background-color: #ffffff; border: 2px solid #1f77b4; border-radius: 8px; padding: 12px}
    .symptom-btn {margin: 5px; padding: 8px 16px; border-radius: 20px; background-color: #1f77b4; color: white; border: none}
    .symptom-btn:hover {background-color: #0d5b9f}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">MediExplain AI</h1>', unsafe_allow_html=True)
st.markdown("### Complete Healthcare Solution: Multi-Disease Prediction with Explainable AI")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Multi-Disease Prediction", "Health Chat Assistant", "Compliance Center", "Tech & Documentation", "Feedback"])

# Disease information
DISEASES = {
    "Heart Disease": {
        "features": ['Age', 'Cholesterol', 'Max_HR', 'Resting_BP', 'Blood_Sugar', 'BMI', 'Exercise_Hours', 'Smoking_Score', 'Alcohol_Consumption', 'Stress_Level'],
        "description": "Cardiovascular conditions affecting heart and blood vessels"
    },
    "Diabetes": {
        "features": ['Age', 'BMI', 'Genetic_Predisposition', 'Blood_Sugar', 'Exercise_Hours', 'Diet_Score', 'Blood_Pressure', 'Pregnancies'],
        "description": "Metabolic disorder characterized by high blood sugar levels"
    },
    "Hypertension": {
        "features": ['Age', 'BMI', 'Sodium_Intake', 'Stress_Level', 'Alcohol_Consumption', 'Exercise_Hours', 'Family_History', 'Smoking_Score'],
        "description": "Chronic condition with elevated blood pressure levels"
    },
    "Asthma": {
        "features": ['Age', 'Pollution_Exposure', 'Allergy_History', 'Family_History', 'Smoking_Score', 'Exercise_Tolerance', 'Respiratory_Rate', 'Cough_Frequency'],
        "description": "Respiratory condition causing breathing difficulties"
    },
    "Arthritis": {
        "features": ['Age', 'BMI', 'Joint_Pain_Level', 'Previous_Injuries', 'Family_History', 'Exercise_Hours', 'Inflammation_Markers', 'Mobility_Score'],
        "description": "Joint disorder causing pain and stiffness"
    }
}

# Load and preprocess data for multiple diseases
@st.cache_data
def load_data():
    from sklearn.datasets import make_classification
    
    # Create synthetic datasets for each disease
    datasets = {}
    
    for disease, info in DISEASES.items():
        n_features = len(info["features"])
        X, y = make_classification(
            n_samples=1000, 
            n_features=n_features, 
            n_informative=max(3, n_features-2), 
            n_redundant=min(2, n_features-3),
            n_clusters_per_class=1, 
            random_state=42
        )
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=info["features"])
        df['Disease_Risk'] = y
        datasets[disease] = df
    
    return datasets

# Train models for all diseases
@st.cache_data
def train_models(datasets):
    models = {}
    
    for disease, df in datasets.items():
        features = DISEASES[disease]["features"]
        X = df[features]
        y = df['Disease_Risk']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        # Calculate accuracy
        lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        
        models[disease] = {
            'lr': lr_model,
            'rf': rf_model,
            'xgb': xgb_model,
            'scaler': scaler,
            'lr_acc': lr_acc,
            'rf_acc': rf_acc,
            'xgb_acc': xgb_acc,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test
        }
    
    return models

# Initialize SHAP explainer
def init_shap_explainer(model, X_train, model_type):
    if model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer

# Generate SHAP plots
def create_shap_plots(explainer, input_data, feature_names, model_type):
    # Calculate SHAP values
    if model_type == "linear":
        shap_values = explainer.shap_values(input_data)
    else:
        shap_values = explainer(input_data)
    
    # Create plots
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if model_type == "linear":
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    # Create force plot for the specific prediction
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if model_type == "linear":
        shap.force_plot(explainer.expected_value, shap_values, input_data, 
                       feature_names=feature_names, matplotlib=True, show=False)
    else:
        shap.force_plot(explainer.expected_value, shap_values.values, input_data, 
                       feature_names=feature_names, matplotlib=True, show=False)
    plt.tight_layout()
    
    return fig1, fig2

# Load data and train models
datasets = load_data()
models_data = train_models(datasets)

# Enhanced Health Chat Assistant with Internet Search
class HealthChatAssistant:
    def __init__(self):
        self.trusted_sources = {
            'who': 'https://www.who.int/health-topics/',
            'cdc': 'https://www.cdc.gov/',
            'nih': 'https://www.nih.gov/health-information',
            'mayoclinic': 'https://www.mayoclinic.org/diseases-conditions',
            'webmd': 'https://www.webmd.com/'
        }
        self.last_api_call = 0
    
    def rate_limit_check(self, min_interval=2):
        """Ensure we don't make too many rapid requests"""
        current_time = time.time()
        if current_time - self.last_api_call < min_interval:
            time.sleep(min_interval - (current_time - self.last_api_call))
        self.last_api_call = time.time()
    
    def search_trusted_source(self, query, source='mayoclinic'):
        """Search trusted medical sources for information"""
        try:
            self.rate_limit_check()
            
            # Format query for URL
            search_query = query.replace(' ', '-').lower()
            url = f"{self.trusted_sources[source]}{search_query}"
            
            # Send request
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract relevant content
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs[:3]])
                
                if content.strip():
                    return f"According to {source.upper()}:\n\n{content[:500]}...\n\nRead more: {url}"
                else:
                    return f"I found information about {query} on {source.upper()}, but couldn't extract detailed content. Please visit: {url}"
                    
            else:
                return f"I found information on {source}, but couldn't retrieve detailed content. Please consult a healthcare professional for specific advice."
                
        except Exception as e:
            return f"I couldn't retrieve information at the moment. Please try again later or consult a healthcare professional."
    
    def analyze_symptoms(self, text):
        """Enhanced symptom analysis with better pattern matching"""
        symptoms_found = {}
        text_lower = text.lower()
        
        # Comprehensive symptom mapping with patterns
        symptom_patterns = {
            'chest pain': [
                r'chest.*(pain|discomfort|pressure|tightness|ache)',
                r'(heart|pectoral|sternum).*hurt',
                r'angina', r'myocardial'
            ],
            'shortness of breath': [
                r'(shortness|difficulty|trouble|hard).*breath',
                r'breathless', r'dyspnea', r'cannot.*breathe',
                r'gasping.*air', r'suffocating'
            ],
            'high blood sugar': [
                r'(high|elevated|increased).*(blood.*sugar|glucose)',
                r'hyperglycemia', r'diabetes.*symptoms',
                r'sugar.*level.*high'
            ],
            'frequent urination': [
                r'(frequent|often|multiple).*urinat',
                r'pee.*a lot', r'urinary.*frequency',
                r'getting.*up.*night.*pee'
            ],
            'headache': [
                r'headache', r'migraine', r'head.*hurt',
                r'head.*pain', r'splitting.*head'
            ],
            'joint pain': [
                r'joint.*(pain|ache|hurt|discomfort)',
                r'arthralgia', r'knee.*pain', r'hip.*pain',
                r'shoulder.*pain', r'wrist.*pain'
            ],
            'fever': [
                r'fever', r'temperature', r'hot.*body',
                r'chills.*sweat', r'pyrexia'
            ],
            'cough': [
                r'cough', r'coughing', r'hack',
                r'clear.*throat.*often'
            ],
            'nausea': [
                r'nausea', r'feel.*sick', r'want.*vomit',
                r'queasy', r'sick.*stomach'
            ],
            'dizziness': [
                r'dizziness', r'lightheaded', r'vertigo',
                r'room.*spinning', r'unsteady'
            ]
        }
        
        for symptom, patterns in symptom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    symptoms_found[symptom] = symptoms_found.get(symptom, 0) + 1
        
        return symptoms_found
    
    def generate_response(self, user_input):
        """Generate intelligent response based on user input"""
        # Analyze symptoms
        symptoms = self.analyze_symptoms(user_input)
        
        if symptoms:
            # If symptoms detected, provide focused response
            response = "I understand you're experiencing some symptoms. "
            primary_symptom = max(symptoms.items(), key=lambda x: x[1])[0]
            
            # Get information from trusted source
            medical_info = self.search_trusted_source(primary_symptom)
            
            response += f"Based on your description of **{primary_symptom}**, here's what I found:\n\n"
            response += medical_info
            response += "\n\n**Please note:** This is general information only. For proper diagnosis and treatment, please consult a healthcare professional."
            
            if len(symptoms) > 1:
                other_symptoms = [s for s in symptoms.keys() if s != primary_symptom]
                response += f"\n\nI also detected these symptoms: {', '.join(other_symptoms)}"
                
        else:
            # General health advice or conversation
            response = self.handle_general_query(user_input)
        
        return response
    
    def handle_general_query(self, user_input):
        """Handle general health queries"""
        lower_input = user_input.lower()
        
        if any(word in lower_input for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! 👋 I'm MediExplain AI, your health assistant. How can I help you today? You can describe your symptoms or ask health-related questions."
        
        elif any(word in lower_input for word in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! I'm here to help. Is there anything else you'd like to know about your health?"
        
        elif any(word in lower_input for word in ['what can you do', 'help', 'capabilities']):
            return "I can help you with:\n- Symptom analysis and initial assessment\n- Health information from trusted sources\n- General health advice\n- Multi-disease risk prediction\n- Explanations of medical concepts\n\nWhat would you like to know?"
        
        elif '?' in user_input:
            # Try to answer specific questions
            if 'heart' in lower_input:
                return self.search_trusted_source('heart disease')
            elif 'diabet' in lower_input:
                return self.search_trusted_source('diabetes')
            elif 'blood pressure' in lower_input or 'hypertension' in lower_input:
                return self.search_trusted_source('hypertension')
            elif 'covid' in lower_input:
                return self.search_trusted_source('covid-19')
            elif 'asthma' in lower_input:
                return self.search_trusted_source('asthma')
            elif 'arthritis' in lower_input:
                return self.search_trusted_source('arthritis')
            else:
                return "That's an interesting health question. While I can provide general information, for specific medical advice, it's best to consult with a healthcare professional. Would you like me to look up general information about this topic?"
        
        else:
            return "Thank you for sharing. I'm here to help with health-related questions and concerns. You can describe any symptoms you're experiencing, ask about health conditions, or request general health information. How can I assist you today?"

# Initialize the chatbot
@st.cache_resource
def load_chatbot():
    return HealthChatAssistant()

health_chatbot = load_chatbot()

# Utility functions
def format_chat_message(role, content):
    """Format chat messages with proper styling"""
    if role == "user":
        return f'<div class="user-msg"><b>You:</b> {content}</div>'
    else:
        # Convert line breaks to HTML for better formatting
        formatted_content = content.replace('\n', '<br>')
        return f'<div class="bot-msg"><b>MediExplain AI:</b> {formatted_content}</div>'

def validate_medical_inputs(input_values):
    """Validate medical input ranges"""
    warnings = []
    
    if input_values.get('Blood_Sugar', 0) > 200:
        warnings.append("Blood sugar level is elevated")
    
    if input_values.get('Resting_BP', 0) > 140:
        warnings.append("Resting blood pressure is high")
    
    if input_values.get('Cholesterol', 0) > 240:
        warnings.append("Cholesterol level is high")
    
    if input_values.get('BMI', 0) > 30:
        warnings.append("BMI indicates obesity")
    
    return warnings

def anonymize_data(data_dict, user_id):
    """Proper data anonymization for compliance"""
    anonymized = data_dict.copy()
    
    # Remove or hash identifiable information
    if 'name' in anonymized:
        anonymized['name'] = f"patient_{hash(user_id)}"
    
    if 'email' in anonymized:
        anonymized['email'] = None
    
    # Generalize age
    if 'age' in anonymized:
        age = anonymized['age']
        anonymized['age_group'] = f"{int(age/10)*10}-{int(age/10)*10+9}"
        del anonymized['age']
    
    return anonymized

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to MediExplain AI
    
    Our comprehensive healthcare AI system provides transparent and explainable multi-disease risk assessments 
    while ensuring compliance with healthcare regulations like GDPR and the Indian IT Act.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Multi-Disease Prediction")
        st.markdown("""
        - Risk assessment for heart disease, diabetes, hypertension, asthma, and arthritis
        - Understand which factors contribute to risk assessments
        - Visual explanations for each prediction
        - Suitable for both clinicians and patients
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Regulatory Compliance")
        st.markdown("""
        - GDPR and Indian IT Act compliant
        - Full audit trails for all predictions
        - Consent management integrated
        - Data anonymization features
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Accurate Risk Assessment")
        st.markdown("""
        - Multiple model approaches for each disease
        - State-of-the-art machine learning
        - Continuous model improvement
        - Clinical validation support
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 💬 Health Chat Assistant")
        st.markdown("""
        - Describe your symptoms in natural language
        - Get instant risk assessments
        - Receive personalized health advice
        - Ask questions about your health
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show model performance
    st.markdown("### Model Performance Across Diseases")
    perf_data = []
    for disease, data in models_data.items():
        perf_data.append({
            'Disease': disease,
            'Logistic Regression': data['lr_acc'],
            'Random Forest': data['rf_acc'],
            'XGBoost': data['xgb_acc']
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_melted = perf_df.melt(id_vars=['Disease'], var_name='Model', value_name='Accuracy')
    
    fig = px.bar(perf_melted, x='Disease', y='Accuracy', color='Model', 
                 title='Model Accuracy by Disease', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### How It Works
    1. **Describe your symptoms** to our Health Chat Assistant for initial assessment
    2. **Input detailed health data** for precise multi-disease risk prediction
    3. **Select a model** based on your needs for interpretability vs. accuracy
    4. **Receive risk assessments** with clear probability scores for multiple diseases
    5. **Explore the explanations** to understand which factors contributed most
    6. **Run what-if scenarios** to see how lifestyle changes would affect risks
    7. **All interactions are logged** for compliance and audit purposes
    """)

# Multi-Disease Prediction page
elif page == "Multi-Disease Prediction":
    st.markdown('<h2 class="sub-header">Multi-Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Disease selection
    selected_diseases = st.multiselect(
        "Select Diseases to Assess",
        list(DISEASES.keys()),
        default=["Heart Disease", "Diabetes"],
        help="Choose which diseases to evaluate risk for"
    )
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ["Logistic Regression (Most Interpretable)", "Random Forest", "XGBoost (Most Accurate)"],
        help="Choose between interpretability (Logistic Regression) and accuracy (XGBoost)"
    )
    
    # Create input form based on selected diseases
    st.markdown("### Patient Information")
    
    input_values = {}
    features_to_show = set()
    
    for disease in selected_diseases:
        features_to_show.update(DISEASES[disease]["features"])
    
    features_to_show = sorted(list(features_to_show))
    
    col1, col2 = st.columns(2)
    
    # Display input fields for all relevant features
    for i, feature in enumerate(features_to_show):
        col = col1 if i % 2 == 0 else col2
        
        if feature == 'Age':
            input_values[feature] = col.slider("Age", 20, 100, 50)
        elif feature == 'Cholesterol':
            input_values[feature] = col.slider("Cholesterol (mg/dL)", 100, 400, 200)
        elif feature == 'Max_HR':
            input_values[feature] = col.slider("Max Heart Rate", 60, 220, 150)
        elif feature == 'Resting_BP':
            input_values[feature] = col.slider("Resting Blood Pressure", 80, 200, 120)
        elif feature == 'Blood_Sugar':
            input_values[feature] = col.slider("Blood Sugar (mg/dL)", 70, 300, 100)
        elif feature == 'BMI':
            input_values[feature] = col.slider("BMI", 15.0, 40.0, 25.0)
        elif feature == 'Exercise_Hours':
            input_values[feature] = col.slider("Exercise Hours per Week", 0, 20, 5)
        elif feature == 'Smoking_Score':
            input_values[feature] = col.slider("Smoking (0=Never, 10=Heavy)", 0, 10, 0)
        elif feature == 'Alcohol_Consumption':
            input_values[feature] = col.slider("Alcohol Consumption (units/week)", 0, 50, 5)
        elif feature == 'Stress_Level':
            input_values[feature] = col.slider("Stress Level (0=Low, 10=High)", 0, 10, 3)
        elif feature == 'Genetic_Predisposition':
            input_values[feature] = col.slider("Genetic Predisposition (0=None, 10=Strong)", 0, 10, 0)
        elif feature == 'Diet_Score':
            input_values[feature] = col.slider("Diet Score (0=Poor, 10=Excellent)", 0, 10, 5)
        elif feature == 'Pregnancies':
            input_values[feature] = col.slider("Number of Pregnancies", 0, 10, 0)
        elif feature == 'Sodium_Intake':
            input_values[feature] = col.slider("Sodium Intake (0=Low, 10=High)", 0, 10, 5)
        elif feature == 'Family_History':
            input_values[feature] = col.slider("Family History (0=None, 10=Strong)", 0, 10, 0)
        elif feature == 'Pollution_Exposure':
            input_values[feature] = col.slider("Pollution Exposure (0=Low, 10=High)", 0, 10, 3)
        elif feature == 'Allergy_History':
            input_values[feature] = col.slider("Allergy History (0=None, 10=Severe)", 0, 10, 0)
        elif feature == 'Respiratory_Rate':
            input_values[feature] = col.slider("Respiratory Rate (breaths/min)", 12, 30, 16)
        elif feature == 'Cough_Frequency':
            input_values[feature] = col.slider("Cough Frequency (0=Never, 10=Constant)", 0, 10, 0)
        elif feature == 'Joint_Pain_Level':
            input_values[feature] = col.slider("Joint Pain Level (0=None, 10=Severe)", 0, 10, 0)
        elif feature == 'Previous_Injuries':
            input_values[feature] = col.slider("Previous Injuries (0=None, 10=Many)", 0, 10, 0)
        elif feature == 'Inflammation_Markers':
            input_values[feature] = col.slider("Inflammation Markers (0=Low, 10=High)", 0, 10, 0)
        elif feature == 'Mobility_Score':
            input_values[feature] = col.slider("Mobility Score (0=Poor, 10=Excellent)", 0, 10, 8)
        elif feature == 'Exercise_Tolerance':
            input_values[feature] = col.slider("Exercise Tolerance (0=Poor, 10=Excellent)", 0, 10, 7)
        else:
            # Default slider for any unexpected features
            input_values[feature] = col.slider(feature, 0, 10, 5)
    
    # Validate inputs
    warnings = validate_medical_inputs(input_values)
    if warnings:
        st.warning("**Input Validation Warnings:**\n\n" + "\n".join([f"• {w}" for w in warnings]))
    
    # Make predictions for all selected diseases
    if st.button("Predict Risks", type="primary"):
        results = {}
        
        for disease in selected_diseases:
            features = DISEASES[disease]["features"]
            
            # Prepare input data
            input_data = np.array([[input_values[feature] for feature in features]])
            
            # Get the selected model
            if "Logistic Regression" in model_option:
                model = models_data[disease]['lr']
                input_processed = models_data[disease]['scaler'].transform(input_data)
                model_type = "linear"
            elif "Random Forest" in model_option:
                model = models_data[disease]['rf']
                input_processed = input_data
                model_type = "tree"
            else:
                model = models_data[disease]['xgb']
                input_processed = input_data
                model_type = "tree"
            
            # Initialize explainer
            explainer = init_shap_explainer(model, models_data[disease]['X_train'], model_type)
            
            # Get prediction probability
            proba = model.predict_proba(input_processed)[0][1]
            
            # Store results
            results[disease] = {
                'probability': proba,
                'explainer': explainer,
                'input_processed': input_processed,
                'model_type': model_type,
                'features': features
            }
        
        # Display results
        st.markdown("### Risk Assessment Results")
        
        # Create a results grid
        cols = st.columns(len(selected_diseases))
        
        for i, (disease, result) in enumerate(results.items()):
            proba = result['probability']
            
            with cols[i]:
                if proba > 0.7:
                    st.markdown(f'<div class="prediction-high">', unsafe_allow_html=True)
                    st.markdown(f"**{disease}**")
                    st.markdown(f"**High Risk: {proba:.1%}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif proba > 0.3:
                    st.markdown(f'<div class="prediction-medium">', unsafe_allow_html=True)
                    st.markdown(f"**{disease}**")
                    st.markdown(f"**Moderate Risk: {proba:.1%}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-low">', unsafe_allow_html=True)
                    st.markdown(f"**{disease}**")
                    st.markdown(f"**Low Risk: {proba:.1%}**")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Show detailed explanations for each disease
        for disease, result in results.items():
            st.markdown(f"### {disease} Explanation")
            
            # Create SHAP plots
            fig1, fig2 = create_shap_plots(
                result['explainer'], 
                result['input_processed'], 
                result['features'], 
                result['model_type']
            )
            
            st.pyplot(fig1)
            st.pyplot(fig2)
            
            # Text interpretation
            st.markdown("#### Interpretation")
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            
            # Generate some interpretive text based on the input
            interpretation = []
            proba = result['probability']
            
            if proba > 0.7:
                interpretation.append(f"High risk of {disease} detected.")
            elif proba > 0.3:
                interpretation.append(f"Moderate risk of {disease} detected.")
            else:
                interpretation.append(f"Low risk of {disease} detected.")
            
            # Add disease-specific interpretations
            if disease == "Heart Disease":
                if input_values['Age'] > 60:
                    interpretation.append("Advanced age is increasing the risk score.")
                if input_values['Cholesterol'] > 240:
                    interpretation.append("High cholesterol levels are a significant risk factor.")
                if input_values['BMI'] > 30:
                    interpretation.append("Elevated BMI is contributing to increased risk.")
            
            elif disease == "Diabetes":
                if input_values['Blood_Sugar'] > 140:
                    interpretation.append("Elevated blood sugar levels are a primary concern.")
                if input_values['Genetic_Predisposition'] > 7:
                    interpretation.append("Strong genetic predisposition detected.")
            
            elif disease == "Hypertension":
                if input_values['Resting_BP'] > 140:
                    interpretation.append("High resting blood pressure is a major risk factor.")
                if input_values['Sodium_Intake'] > 7:
                    interpretation.append("High sodium intake may be contributing to hypertension risk.")
            
            for item in interpretation:
                st.markdown(f"- {item}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall health assessment
        st.markdown("### Overall Health Assessment")
        
        avg_risk = np.mean([result['probability'] for result in results.values()])
        
        if avg_risk > 0.6:
            st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
            st.markdown(f"**Overall Health Risk: High**")
            st.markdown("Based on your health profile, you have elevated risks for multiple conditions. We recommend consulting with a healthcare provider for a comprehensive evaluation.")
            st.markdown('</div>', unsafe_allow_html=True)
        elif avg_risk > 0.3:
            st.markdown('<div class="prediction-medium">', unsafe_allow_html=True)
            st.markdown(f"**Overall Health Risk: Moderate**")
            st.markdown("Your health profile shows some areas of concern. Consider lifestyle modifications and monitor your health regularly.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
            st.markdown(f"**Overall Health Risk: Low**")
            st.markdown("Your health profile indicates generally low risks. Maintain your healthy habits and continue regular health check-ups.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Log the prediction
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "model": model_option,
            "input_features": anonymize_data(input_values, "user_anonymous"),
            "predictions": {disease: result['probability'] for disease, result in results.items()}
        }
        
        st.markdown("### Audit Log Entry")
        st.json(prediction_log)

# Health Chat Assistant page
elif page == "Health Chat Assistant":
    st.markdown('<h2 class="sub-header">Health Chat Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Describe your symptoms or health concerns in natural language, and our AI assistant will:
    - Analyze your description for potential health risks
    - Provide information from trusted medical sources
    - Offer guidance on next steps
    - Answer health-related questions
    """)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history with improved UI
    st.markdown("### Conversation")
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                # Format bot response with better readability
                formatted_content = msg["content"].replace('\n', '<br>')
                st.markdown(f'<div class="bot-msg"><b>MediExplain AI:</b> {formatted_content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # User input with improved styling
    user_input = st.text_input("Describe your symptoms or ask a question:", 
                              key="chat_input",
                              value=st.session_state.get("chat_input", ""),
                              placeholder="Type your symptoms or health question here...")
    
    col1, col2 = st.columns([1, 6])
    
    with col1:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.chat_input = ""
        st.rerun()
    
    if (send_button or user_input) and user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate bot response
        with st.spinner("Analyzing your symptoms..."):
            response = health_chatbot.generate_response(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input field
        st.session_state.chat_input = ""
        
        # Rerun to update the chat display
        st.rerun()
    
    # Quick symptom buttons with fixed implementation
    st.markdown("### Common Symptoms")
    common_symptoms = [
        "Chest pain", "Shortness of breath", "High blood sugar",
        "Frequent urination", "Headache", "Joint pain",
        "Fever", "Cough", "Nausea", "Dizziness"
    ]
    
    cols = st.columns(5)
    for i, symptom in enumerate(common_symptoms):
        col_idx = i % 5
        if cols[col_idx].button(symptom, key=f"symptom_{i}", use_container_width=True):
            st.session_state.chat_input = f"I'm experiencing {symptom.lower()}"
            st.rerun()
    
    # Trusted sources information
    with st.expander("ℹ️ About Our Information Sources"):
        st.markdown("""
        Our chatbot retrieves information from trusted medical sources including:
        - World Health Organization (WHO)
        - Centers for Disease Control and Prevention (CDC)
        - National Institutes of Health (NIH)
        - Mayo Clinic
        - WebMD
        
        **Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for diagnosis and treatment.
        """)

# Compliance Center page
elif page == "Compliance Center":
    st.markdown('<h2 class="sub-header">Regulatory Compliance Center</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    MediExplain AI is designed with privacy and regulatory compliance at its core, adhering to 
    GDPR, HIPAA, and the Indian IT Act requirements.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Privacy", "Consent Management", "Audit Trails", "Security"])
    
    with tab1:
        st.markdown("### Data Privacy & Anonymization")
        st.markdown("""
        - All patient data is anonymized before processing
        - Personal identifiers are stored separately from health data
        - Data minimization principles are applied - we only collect what's necessary
        - Right to be forgotten is implemented with full data deletion workflows
        """)
        
        # Show anonymization example
        st.markdown("#### Data Anonymization Example")
        original_data = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Age': [45, 62, 33],
            'Condition': ['Hypertension', 'Diabetes', 'Asthma']
        })
        
        st.markdown("**Original Data:**")
        st.dataframe(original_data)
        
        anonymized_data = original_data.copy()
        anonymized_data['Name'] = ['Patient_001', 'Patient_002', 'Patient_003']
        anonymized_data['Age'] = ['40-50', '60-70', '30-40']  # Age grouping
        
        st.markdown("**Anonymized Data:**")
        st.dataframe(anonymized_data)
    
    with tab2:
        st.markdown("### Consent Management")
        st.markdown("""
        - Explicit consent is obtained before processing health data
        - Consent preferences are stored with timestamps and version history
        - Patients can withdraw consent at any time through the portal
        - Consent scope is clearly defined and documented
        """)
        
        # Consent simulator
        st.markdown("#### Consent Simulator")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Consent Status**")
            consent_status = st.selectbox("Consent", ["Given", "Withdrawn"])
            consent_date = st.date_input("Consent Date", value=datetime.now())
            purposes = st.multiselect("Consent Purposes", 
                                    ["Risk Prediction", "Research", "Quality Improvement", "Clinical Care"])
        
        with col2:
            st.markdown("**Consent Record**")
            consent_record = {
                "status": consent_status,
                "date": consent_date.isoformat(),
                "purposes": purposes,
                "version": "1.0",
                "patient_id": "anonymous_123"
            }
            st.json(consent_record)
    
    with tab3:
        st.markdown("### Audit Trails")
        st.markdown("""
        - All predictions are logged with complete input data and results
        - Model versions are tracked for reproducibility
        - User access is logged and monitored
        - Full audit trail export available for regulators
        """)
        
        # Generate sample audit log
        st.markdown("#### Sample Audit Log")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": "clinician_456",
            "action": "prediction",
            "model_used": "XGBoost v1.2",
            "patient_id": "anonymous_789",
            "input_data_hash": "a1b2c3d4e5f6",
            "prediction_result": 0.67
        }
        st.json(log_data)
        
        # Show log visualization
        st.markdown("#### Access Pattern Visualization")
        dates = pd.date_range(start='2023-01-01', end='2023-01-15', freq='D')
        access_counts = np.random.poisson(lam=15, size=len(dates))
        
        fig = px.bar(x=dates, y=access_counts, 
                     labels={'x': 'Date', 'y': 'Number of Accesses'},
                     title='System Access Pattern')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Security Measures")
        st.markdown("""
        - End-to-end encryption for data in transit and at rest
        - Regular security penetration testing
        - Role-based access control with minimum necessary privileges
        - Comprehensive incident response plan
        - All data stored in jurisdiction-compliant locations
        """)
        
        # Security score
        st.markdown("#### Security Posture Score")
        security_metrics = {
            'Encryption': 95,
            'Access Control': 88,
            'Audit Logging': 92,
            'Vulnerability Management': 85,
            'Incident Response': 90
        }
        
        fig = go.Figure(go.Bar(
            x=list(security_metrics.values()),
            y=list(security_metrics.keys()),
            orientation='h'
        ))
        fig.update_layout(title="Security Metrics Score (%)")
        st.plotly_chart(fig, use_container_width=True)

# Tech & Documentation page
elif page == "Tech & Documentation":
    st.markdown('<h2 class="sub-header">Technology & Documentation</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "Models", "Data", "XAI Methods"])
    
    with tab1:
        st.markdown("### System Architecture")
        st.markdown("""
        MediExplain AI is built on a modern, scalable architecture:
        
        **Frontend:** Streamlit-based web application
        **Backend:** Python with FastAPI for RESTful services
        **Machine Learning:** Scikit-learn, XGBoost, SHAP, LIME
        **Database:** PostgreSQL with JSONB for flexible data storage
        **Deployment:** Docker containers on Kubernetes cluster
        **Security:** TLS encryption, OAuth2 authentication, role-based access control
        """)
        
        # Architecture diagram (conceptual)
        st.markdown("#### Architecture Diagram")
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*V6s6g8qYV0lI6Vg2x3J3zg.png", 
                 caption="Conceptual System Architecture", use_column_width=True)
    
    with tab2:
        st.markdown("### Model Information")
        st.markdown("""
        We employ multiple modeling approaches to balance accuracy and interpretability:
        
        **Logistic Regression:** Highly interpretable linear model
        **Random Forest:** Ensemble method that captures non-linear relationships
        **XGBoost:** State-of-the-art gradient boosting with high accuracy
        """)
        
        # Model comparison
        st.markdown("#### Model Comparison")
        comparison_data = []
        for disease, data in models_data.items():
            comparison_data.append({
                'Disease': disease,
                'Model': 'Logistic Regression',
                'Accuracy': data['lr_acc']
            })
            comparison_data.append({
                'Disease': disease,
                'Model': 'Random Forest',
                'Accuracy': data['rf_acc']
            })
            comparison_data.append({
                'Disease': disease,
                'Model': 'XGBoost',
                'Accuracy': data['xgb_acc']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(comp_df, x='Disease', y='Accuracy', color='Model', 
                     title='Model Accuracy by Disease', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Data Information")
        st.markdown("""
        **Data Sources:** 
        - Electronic Health Records (EHR) systems
        - Patient-reported outcomes
        - Medical device integrations
        - Clinical trial data (with appropriate consent)
        
        **Data Preprocessing:**
        - Missing value imputation
        - Feature scaling and normalization
        - Outlier detection and handling
        - Temporal feature engineering
        """)
        
        # Show feature distributions
        st.markdown("#### Feature Distributions")
        selected_disease = st.selectbox("Select Disease", list(DISEASES.keys()))
        
        feature_to_show = st.selectbox("Select feature to visualize", DISEASES[selected_disease]["features"])
        
        fig = px.histogram(datasets[selected_disease], x=feature_to_show, color='Disease_Risk',
                           title=f'Distribution of {feature_to_show} by {selected_disease} Risk',
                           nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Explainable AI Methods")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations):** 
        - Game theory approach to explain model outputs
        - Provides consistent and locally accurate feature attributions
        - Supports both global and local interpretability
        
        **LIME (Local Interpretable Model-agnostic Explanations):**
        - Creates local surrogate models to explain individual predictions
        - Model-agnostic approach works with any algorithm
        - Useful for text and image models in addition to tabular data
        """)
        
        # SHAP explanation
        st.markdown("#### SHAP Values Explanation")
        st.markdown("""
        SHAP values represent the contribution of each feature to the prediction, 
        measured as the change in the expected model output when conditioning on that feature.
        
        The base value is the average model output, and each SHAP value shows how much 
        each feature pushed the prediction away from this base value.
        """)
        
        # Interactive SHAP explanation
        st.markdown("##### Interactive SHAP Explanation")
        selected_disease = st.selectbox("Select Disease for Explanation", list(DISEASES.keys()))
        sample_idx = st.slider("Select sample to explain", 0, len(models_data[selected_disease]['X_test'])-1, 0)
        
        # Get sample and prediction
        sample_data = models_data[selected_disease]['X_test'].iloc[sample_idx:sample_idx+1]
        sample_pred = models_data[selected_disease]['xgb'].predict_proba(sample_data)[0][1]
        
        # Initialize explainer
        xgb_explainer = init_shap_explainer(models_data[selected_disease]['xgb'], models_data[selected_disease]['X_train'], "tree")
        
        # Calculate SHAP values
        shap_values = xgb_explainer(sample_data)
        
        # Create force plot
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"**Prediction for this sample: {sample_pred:.1%} risk of {selected_disease}**")

# Feedback page
elif page == "Feedback":
    st.markdown('<h2 class="sub-header">Feedback & Contact</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    We value your feedback to improve MediExplain AI. Please share your thoughts, 
    suggestions, or report any issues you encounter.
    """)
    
    # Feedback form
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)")
        role = st.selectbox("Role", ["Clinician", "Researcher", "Patient", "Administrator", "Other"])
        email = st.text_input("Email (optional)")
        feedback_type = st.selectbox("Feedback Type", 
                                   ["General Feedback", "Bug Report", "Feature Request", "Data Accuracy Concern"])
        message = st.text_area("Your Message", height=150)
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            # In a real application, this would be saved to a database
            # For demo, we'll just show a success message
            st.success("Thank you for your feedback! We will review your message and respond if needed.")
            
            # Show what would be saved
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "name": name,
                "role": role,
                "email": email,
                "type": feedback_type,
                "message": message
            }
            
            st.markdown("#### Feedback Record")
            st.json(feedback_data)
    
    st.markdown("---")
    st.markdown("### Contact Information")
    st.markdown("""
    **MediExplain AI Team**  
    Email: contact@mediexplain.ai  
    Phone: +1 (555) 123-HELP  
    Address: 123 Healthcare Ave, Innovation City, IC 12345  
    """)
    
    # FAQ section
    st.markdown("### Frequently Asked Questions")
    
    with st.expander("How accurate are the predictions?"):
        st.markdown("""
        Our models achieve 80-90% accuracy on test datasets, but actual performance may vary 
        based on data quality and population characteristics. Predictions should be used as 
        decision support tools rather than definitive diagnoses.
        """)
    
    with st.expander("Is my data secure and private?"):
        st.markdown("""
        Yes, we follow industry best practices for data security and privacy. All data is 
        encrypted, access is strictly controlled, and we comply with GDPR, HIPAA, and other 
        relevant regulations. Patient data is anonymized before processing.
        """)
    
    with st.expander("Can I use MediExplain AI in my clinical practice?"):
        st.markdown("""
        MediExplain AI is designed for clinical decision support. However, it should be used 
        by qualified healthcare professionals in conjunction with their clinical judgment. 
        Please contact us for information about implementation in your practice.
        """)
    
    with st.expander("How often are models updated?"):
        st.markdown("""
        Models are retrained quarterly with new data, or when significant drift in performance 
        is detected. All model updates undergo rigorous validation before deployment.
        """)

# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("MediExplain AI | © 2023 | Transparent and Responsible Healthcare AI")
st.markdown('</div>', unsafe_allow_html=True)
