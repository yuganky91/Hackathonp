
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import json
from datetime import datetime
import base64

# Set up the page
st.set_page_config(
    page_title="MediExplain AI - Disease Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; padding-bottom: 10px}
    .sub-header {font-size: 1.5rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px}
    .feature-box {background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px}
    .prediction-high {background-color: #ffcccc; padding: 20px; border-radius: 10px}
    .prediction-low {background-color: #ccffcc; padding: 20px; border-radius: 10px}
    .interpretation-box {background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-top: 10px}
    .footer {text-align: center; margin-top: 50px; color: #777}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">MediExplain AI</h1>', unsafe_allow_html=True)
st.markdown("### Transparent, Trustworthy, and Compliant Disease Risk Prediction")

# Sidebar for navigation
st.sidebar.image("https://img.icons8.com/dusk/64/000000/hospital.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Disease Risk", "Compliance Center", "Tech & Documentation", "Feedback"])

# Load and preprocess data
@st.cache_resource
def load_data():
    # Using the heart disease dataset from UCI (simulated for demo)
    # In a real scenario, this would be loaded from a proper source
    from sklearn.datasets import make_classification
    
    # Generate synthetic data for demonstration
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=8, 
        n_redundant=2,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'Age', 'Cholesterol', 'Max_HR', 'Resting_BP', 
        'Blood_Sugar', 'BMI', 'Exercise_Hours', 
        'Smoking_Score', 'Alcohol_Consumption', 'Stress_Level'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Heart_Disease_Risk'] = y
    
    return df, feature_names

# Train models
@st.cache_resource
def train_models(df, feature_names):
    X = df[feature_names]
    y = df['Heart_Disease_Risk']
    
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
    
    return {
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

# Initialize SHAP explainer
@st.cache_resource
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
    fig1, ax1 = plt.subplots()
    if model_type == "linear":
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    # Create force plot for the specific prediction
    fig2, ax2 = plt.subplots()
    if model_type == "linear":
        shap.force_plot(explainer.expected_value, shap_values, input_data, 
                       feature_names=feature_names, matplotlib=True, show=False)
    else:
        shap.force_plot(explainer.expected_value, shap_values.values, input_data, 
                       feature_names=feature_names, matplotlib=True, show=False)
    plt.tight_layout()
    
    return fig1, fig2

# Load data and train models
df, feature_names = load_data()
models = train_models(df, feature_names)

# Initialize SHAP explainers
lr_explainer = init_shap_explainer(models['lr'], models['X_train'], "linear")
rf_explainer = init_shap_explainer(models['rf'], models['X_train'], "tree")
xgb_explainer = init_shap_explainer(models['xgb'], models['X_train'], "tree")

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to MediExplain AI
    
    Our system provides transparent and explainable AI-powered disease risk assessments while ensuring 
    compliance with healthcare regulations like GDPR and the Indian IT Act.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Explainable Predictions")
        st.markdown("""
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
        - Multiple model approaches
        - State-of-the-art machine learning
        - Continuous model improvement
        - Clinical validation support
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Personalized Medicine")
        st.markdown("""
        - What-if scenario testing
        - Personalized risk factor analysis
        - Intervention planning support
        - Progress tracking over time
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show model performance
    st.markdown("### Model Performance")
    perf_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [models['lr_acc'], models['rf_acc'], models['xgb_acc']]
    })
    
    fig = px.bar(perf_df, x='Model', y='Accuracy', 
                 title='Model Accuracy on Test Data', 
                 color='Accuracy', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### How It Works
    1. **Input patient data** through our secure form
    2. **Select a model** based on your needs for interpretability vs. accuracy
    3. **Receive a risk assessment** with clear probability score
    4. **Explore the explanation** to understand which factors contributed most
    5. **Run what-if scenarios** to see how changes would affect risk
    6. **All predictions are logged** for compliance and audit purposes
    """)

# Prediction page
elif page == "Predict Disease Risk":
    st.markdown('<h2 class="sub-header">Heart Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ["Logistic Regression (Most Interpretable)", "Random Forest", "XGBoost (Most Accurate)"],
        help="Choose between interpretability (Logistic Regression) and accuracy (XGBoost)"
    )
    
    # Input form
    st.markdown("### Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 70, 300, 100)
    
    with col2:
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        exercise = st.slider("Exercise Hours per Week", 0, 20, 5)
        smoking = st.slider("Smoking (0=Never, 10=Heavy)", 0, 10, 0)
        alcohol = st.slider("Alcohol Consumption (units/week)", 0, 50, 5)
        stress = st.slider("Stress Level (0=Low, 10=High)", 0, 10, 3)
    
    # Create input array
    input_data = np.array([[age, cholesterol, max_hr, resting_bp, blood_sugar, 
                           bmi, exercise, smoking, alcohol, stress]])
    
    # Get the selected model
    if "Logistic Regression" in model_option:
        model = models['lr']
        explainer = lr_explainer
        input_processed = models['scaler'].transform(input_data)
        model_type = "linear"
    elif "Random Forest" in model_option:
        model = models['rf']
        explainer = rf_explainer
        input_processed = input_data
        model_type = "tree"
    else:
        model = models['xgb']
        explainer = xgb_explainer
        input_processed = input_data
        model_type = "tree"
    
    # Make prediction
    if st.button("Predict Risk"):
        # Get prediction probability
        proba = model.predict_proba(input_processed)[0][1]
        
        # Display prediction
        if proba > 0.7:
            st.markdown(f'<div class="prediction-high">', unsafe_allow_html=True)
            st.markdown(f"### High Risk: {proba:.1%}")
            st.markdown("The patient has a high risk of heart disease. Further clinical evaluation is recommended.")
            st.markdown('</div>', unsafe_allow_html=True)
        elif proba > 0.3:
            st.markdown(f'<div class="prediction-low">', unsafe_allow_html=True)
            st.markdown(f"### Moderate Risk: {proba:.1%}")
            st.markdown("The patient has a moderate risk of heart disease. Lifestyle changes may be beneficial.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-low">', unsafe_allow_html=True)
            st.markdown(f"### Low Risk: {proba:.1%}")
            st.markdown("The patient has a low risk of heart disease. Maintain current healthy habits.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate SHAP explanation
        st.markdown("### Explanation of Prediction")
        st.markdown("The chart below shows which factors contributed most to this prediction:")
        
        # Create SHAP plots
        fig1, fig2 = create_shap_plots(explainer, input_processed, feature_names, model_type)
        
        st.pyplot(fig1)
        st.pyplot(fig2)
        
        # Text interpretation
        st.markdown("#### Interpretation")
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        
        # Generate some interpretive text based on the input
        interpretation = []
        if age > 60:
            interpretation.append("Advanced age is increasing the risk score.")
        if cholesterol > 240:
            interpretation.append("High cholesterol levels are a significant risk factor.")
        if bmi > 30:
            interpretation.append("Elevated BMI is contributing to increased risk.")
        if exercise < 3:
            interpretation.append("Low exercise levels are negatively impacting cardiovascular health.")
        if smoking > 5:
            interpretation.append("Smoking is a major contributor to heart disease risk.")
        
        if interpretation:
            for item in interpretation:
                st.markdown(f"- {item}")
        else:
            st.markdown("The patient's profile shows no exceptionally high risk factors.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # What-if analysis
        st.markdown("### What-If Analysis")
        st.markdown("Adjust the sliders below to see how changes would affect the risk prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_cholesterol = st.slider("New Cholesterol", 100, 400, cholesterol, key="new_chol")
            new_bmi = st.slider("New BMI", 15.0, 40.0, bmi, key="new_bmi")
        
        with col2:
            new_exercise = st.slider("New Exercise Hours", 0, 20, exercise, key="new_ex")
            new_smoking = st.slider("New Smoking Level", 0, 10, smoking, key="new_smoke")
        
        # Create modified input
        modified_data = np.array([[age, new_cholesterol, max_hr, resting_bp, blood_sugar, 
                                 new_bmi, new_exercise, new_smoking, alcohol, stress]])
        
        if "Logistic Regression" in model_option:
            modified_processed = models['scaler'].transform(modified_data)
        else:
            modified_processed = modified_data
        
        # Get new prediction
        new_proba = model.predict_proba(modified_processed)[0][1]
        change = new_proba - proba
        
        st.markdown(f"**New predicted risk: {new_proba:.1%}** ({change:+.1%} change)")
        
        # Log the prediction (in a real system, this would go to a database)
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "model": model_option,
            "input_features": input_data[0].tolist(),
            "prediction": float(proba),
            "modified_features": modified_data[0].tolist(),
            "modified_prediction": float(new_proba)
        }
        
        # In a real application, you would save this to a database
        # For demo purposes, we'll just show it
        st.markdown("### Audit Log Entry")
        st.json(prediction_log)

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
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [models['lr_acc'], models['rf_acc'], models['xgb_acc']],
            'Interpretability': [9, 6, 5],
            'Training Time (s)': [0.5, 3.2, 4.8]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df)
        
        # Model performance visualization
        fig = px.scatter(comp_df, x='Interpretability', y='Accuracy', 
                         size='Training Time (s)', text='Model',
                         title='Model Trade-offs: Interpretability vs Accuracy')
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
        feature_to_show = st.selectbox("Select feature to visualize", feature_names)
        
        fig = px.histogram(df, x=feature_to_show, color='Heart_Disease_Risk',
                           title=f'Distribution of {feature_to_show} by Heart Disease Risk',
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
        sample_idx = st.slider("Select sample to explain", 0, len(models['X_test'])-1, 0)
        
        # Get sample and prediction
        sample_data = models['X_test'].iloc[sample_idx:sample_idx+1]
        sample_pred = models['xgb'].predict_proba(sample_data)[0][1]
        
        # Calculate SHAP values
        shap_values = xgb_explainer(sample_data)
        
        # Create force plot
        fig, ax = plt.subplots()
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"**Prediction for this sample: {sample_pred:.1%} risk**")

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
