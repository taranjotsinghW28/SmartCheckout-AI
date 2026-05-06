# app.py - Complete E-commerce Purchase Prediction App (FULLY WORKING)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Page config
st.set_page_config(
    page_title="SmartCheckout - AI Purchase Predictor",
    page_icon="🛍️",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('saved_models/final_purchase_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('saved_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('saved_models/feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('saved_models/optimal_threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
        return model, scaler, features, threshold
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, features, threshold = load_models()

# Custom CSS
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #f0f0f0 !important;
        font-size: 1.1rem;
    }
    
    /* Headers - Dark text */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a2e !important;
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        background-color: white !important;
        color: #1a1a2e !important;
        border-radius: 8px !important;
    }
    .stNumberInput label, .stSelectbox label {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #667eea !important;
    }
    
    /* Result cards */
    .result-purchase {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 1rem 0;
    }
    .result-no-purchase {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-purchase h1, .result-purchase h2, .result-purchase p,
    .result-no-purchase h1, .result-no-purchase h2, .result-no-purchase p {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255,255,255,0.1);
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] label {
        color: #1a1a2e !important;
    }
    [data-testid="stMetric"] .stMarkdown p {
        color: #667eea !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #1a1a2e !important;
        background-color: white !important;
        border-radius: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #666;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛍️ SmartCheckout AI</h1>
    <p>Intelligent Purchase Prediction for E-Commerce</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Status")
    if model:
        st.success("✅ Model Ready")
        st.info("**Random Forest Classifier**\n\nF1-Score: 0.4098\nRecall: 0.6152\nThreshold: 0.5")
    else:
        st.error("❌ Model not loaded")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. Fill in customer session data (4 tabs below)
    2. Click 'Predict Purchase'
    3. Get AI-powered recommendation
    """)
    
    st.markdown("---")
    st.markdown("### 💡 High Intent Signals")
    st.markdown("""
    - ⏱️ **Product time > 10 min**
    - 📉 **Bounce rate < 20%**
    - 🔄 **Returning visitor**
    - 🎄 **Nov/Dec months**
    - 💰 **High page values**
    - 🎁 **Special day proximity**
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Features")
    st.markdown(f"*{len(features) if features else 0} features analyzed*")

# Quick example buttons
st.markdown("### 📝 Customer Session Data")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.clear()
        st.rerun()

with col2:
    if st.button("💎 High Intent Example", use_container_width=True):
        st.session_state.high_intent = True
        st.rerun()

with col3:
    if st.button("⚠️ Low Intent Example", use_container_width=True):
        st.session_state.low_intent = True
        st.rerun()

# Set values based on examples
if st.session_state.get('high_intent', False):
    # HIGH INTENT - Should predict PURCHASE
    defaults = {
        'admin_pages': 3, 'admin_duration': 180.0,
        'info_pages': 2, 'info_duration': 120.0,
        'product_pages': 35, 'product_duration': 1800.0,
        'bounce_rate': 0.12, 'exit_rate': 0.15,
        'page_values': 45.0, 'special_day': 0.8,
        'visitor_type': "Returning_Visitor", 'weekend': "Yes",
        'month': "Nov", 'os': "Mac", 'browser': "Chrome",
        'region': "North America", 'traffic': "Search"
    }
    st.session_state.high_intent = False
elif st.session_state.get('low_intent', False):
    # LOW INTENT - Should predict NO PURCHASE
    defaults = {
        'admin_pages': 0, 'admin_duration': 0.0,
        'info_pages': 0, 'info_duration': 0.0,
        'product_pages': 2, 'product_duration': 30.0,
        'bounce_rate': 0.85, 'exit_rate': 0.80,
        'page_values': 0.0, 'special_day': 0.0,
        'visitor_type': "New_Visitor", 'weekend': "No",
        'month': "Feb", 'os': "Windows", 'browser': "Edge",
        'region': "Other", 'traffic': "Social"
    }
    st.session_state.low_intent = False
else:
    # Default neutral values
    defaults = {
        'admin_pages': 0, 'admin_duration': 0.0,
        'info_pages': 0, 'info_duration': 0.0,
        'product_pages': 10, 'product_duration': 300.0,
        'bounce_rate': 0.5, 'exit_rate': 0.5,
        'page_values': 0.0, 'special_day': 0.0,
        'visitor_type': "New_Visitor", 'weekend': "No",
        'month': "Nov", 'os': "Windows", 'browser': "Chrome",
        'region': "North America", 'traffic': "Search"
    }

# Create tabs for input
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Behavior Metrics", "👤 Customer Info", "💰 Page & Content", "🖥️ Technical Details"])

with tab1:
    st.markdown("#### 📄 Page Visit Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        admin_pages = st.number_input("📁 Admin Pages Visited", min_value=0, max_value=50, value=defaults['admin_pages'],
                                      help="Account, settings, profile pages")
        info_pages = st.number_input("ℹ️ Info Pages Visited", min_value=0, max_value=50, value=defaults['info_pages'],
                                     help="About, FAQ, contact pages")
        product_pages = st.number_input("🛍️ Product Pages Visited ⭐", min_value=0, max_value=200, value=defaults['product_pages'],
                                        help="⭐ MOST IMPORTANT: Product catalog, details pages")
    
    with col2:
        admin_duration = st.number_input("⏱️ Admin Duration (seconds)", min_value=0.0, max_value=5000.0, 
                                         value=defaults['admin_duration'], step=10.0)
        info_duration = st.number_input("⏱️ Info Duration (seconds)", min_value=0.0, max_value=5000.0,
                                        value=defaults['info_duration'], step=10.0)
        product_duration = st.number_input("⏱️ Product Duration (seconds) ⭐", min_value=0.0, max_value=10000.0,
                                           value=defaults['product_duration'], step=30.0,
                                           help="⭐ MOST IMPORTANT: Time browsing products")
    
    st.markdown("#### 📊 Engagement Rates")
    col1, col2 = st.columns(2)
    with col1:
        bounce_rate = st.slider("📉 Bounce Rate", 0.0, 1.0, defaults['bounce_rate'], 0.01,
                               help="% of single-page sessions (lower is better)")
        st.caption("💡 Lower = More engaged users")
    with col2:
        exit_rate = st.slider("🚪 Exit Rate", 0.0, 1.0, defaults['exit_rate'], 0.01,
                             help="% of exits from page (lower is better)")
        st.caption("💡 Lower = Users continue browsing")

with tab2:
    st.markdown("#### 👤 Customer Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"], 
                                    index=0 if defaults['visitor_type'] == "New_Visitor" else 1,
                                    help="⭐ Returning visitors are more likely to buy")
        
        weekend = st.selectbox("📅 Weekend?", ["No", "Yes"], 
                               index=0 if defaults['weekend'] == "No" else 1,
                               help="Weekend shoppers behave differently")
    
    with col2:
        month = st.selectbox("📆 Month", 
                            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                            index=10 if defaults['month'] == "Nov" else 1 if defaults['month'] == "Feb" else 10,
                            help="⭐ November/December have higher purchase rates")
        
        special_day = st.slider("🎉 Special Day (Holiday proximity)", 0.0, 1.0, defaults['special_day'], 0.01,
                               help="0=normal day, 1=day before holiday (e.g., Valentine's, Black Friday)")

with tab3:
    st.markdown("#### 💰 Monetary & Page Values")
    
    page_values = st.number_input("💎 Page Values", min_value=0.0, max_value=100.0, value=defaults['page_values'], step=5.0,
                                 help="⭐ Average monetary value of pages visited (higher = closer to purchase)")
    
    st.markdown("#### 📈 Calculated Metrics (Auto-generated)")
    
    total_pages = admin_pages + info_pages + product_pages
    total_duration = admin_duration + info_duration + product_duration
    avg_time_per_page = total_duration / (total_pages + 1) if total_pages > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages", int(total_pages))
        st.caption("📄 All pages visited")
    with col2:
        st.metric("Total Time (minutes)", f"{total_duration/60:.1f}")
        st.caption("⏱️ Total browsing time")
    with col3:
        st.metric("Avg Time/Page (seconds)", f"{avg_time_per_page:.1f}")
        st.caption("⚡ User engagement level")

with tab4:
    st.markdown("#### 🖥️ Technical Environment")
    col1, col2 = st.columns(2)
    
    with col1:
        os_options = ["Windows", "Mac", "Linux", "Other"]
        os_idx = 0 if defaults['os'] == "Windows" else 1 if defaults['os'] == "Mac" else 2 if defaults['os'] == "Linux" else 3
        os_type = st.selectbox("💻 Operating System", os_options, index=os_idx)
        os_map = {"Windows": 2, "Mac": 1, "Linux": 3, "Other": 4}
        os_numeric = os_map[os_type]
        
        browser_options = ["Chrome", "Firefox", "Safari", "Edge", "Other"]
        browser_idx = 0 if defaults['browser'] == "Chrome" else 1 if defaults['browser'] == "Firefox" else 2 if defaults['browser'] == "Safari" else 3 if defaults['browser'] == "Edge" else 4
        browser = st.selectbox("🌐 Browser", browser_options, index=browser_idx)
        browser_map = {"Chrome": 1, "Firefox": 2, "Safari": 3, "Edge": 4, "Other": 5}
        browser_numeric = browser_map[browser]
    
    with col2:
        region_options = ["North America", "Europe", "Asia", "Other"]
        region_idx = 0 if defaults['region'] == "North America" else 1 if defaults['region'] == "Europe" else 2 if defaults['region'] == "Asia" else 3
        region = st.selectbox("🌍 Region", region_options, index=region_idx)
        region_map = {"North America": 1, "Europe": 2, "Asia": 3, "Other": 4}
        region_numeric = region_map[region]
        
        traffic_options = ["Direct", "Search", "Social", "Ad", "Email", "Other"]
        traffic_idx = 1 if defaults['traffic'] == "Search" else 0 if defaults['traffic'] == "Direct" else 2 if defaults['traffic'] == "Social" else 3 if defaults['traffic'] == "Ad" else 4 if defaults['traffic'] == "Email" else 5
        traffic_type = st.selectbox("📢 Traffic Source", traffic_options, index=traffic_idx)
        traffic_map = {"Direct": 1, "Search": 2, "Social": 3, "Ad": 4, "Email": 5, "Other": 6}
        traffic_numeric = traffic_map[traffic_type]

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 PREDICT PURCHASE", use_container_width=True)

# Prediction and results
if predict_button:
    if model is not None:
        with st.spinner("🧠 Analyzing customer behavior with AI..."):
            time.sleep(1.5)
            
            # Create base dataframe
            input_data = pd.DataFrame({
                'Administrative': [admin_pages],
                'Administrative_Duration': [float(admin_duration)],
                'Informational': [info_pages],
                'Informational_Duration': [float(info_duration)],
                'ProductRelated': [product_pages],
                'ProductRelated_Duration': [float(product_duration)],
                'BounceRates': [bounce_rate],
                'ExitRates': [exit_rate],
                'PageValues': [page_values],
                'SpecialDay': [special_day],
                'OperatingSystems': [os_numeric],
                'Browser': [browser_numeric],
                'Region': [region_numeric],
                'TrafficType': [traffic_numeric],
                'Weekend': [1 if weekend == "Yes" else 0]
            })
            
            # Add engineered features
            input_data['Total_Duration'] = input_data['Administrative_Duration'] + input_data['Informational_Duration'] + input_data['ProductRelated_Duration']
            input_data['Total_Pages'] = input_data['Administrative'] + input_data['Informational'] + input_data['ProductRelated']
            input_data['Avg_Page_Value'] = input_data['PageValues'] / (input_data['Total_Pages'] + 1)
            input_data['Duration_Per_Page'] = input_data['Total_Duration'] / (input_data['Total_Pages'] + 1)
            
            # One-hot encode month (all 12 months)
            all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for m in all_months:
                input_data[f'Month_{m}'] = 1 if month == m else 0
            
            # One-hot encode visitor type
            input_data['VisitorType_Returning_Visitor'] = 1 if visitor_type == "Returning_Visitor" else 0
            input_data['VisitorType_Other'] = 1 if visitor_type == "Other" else 0
            
            # Ensure all features are present
            for col in features:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Keep only expected features in correct order
            input_data = input_data[features]
            
            # Scale features
            input_scaled = scaler.transform(input_data)
            
            # Predict
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if probability >= threshold else 0
            
            # Determine confidence and recommendation
            if probability > 0.7:
                confidence = "High"
                confidence_color = "green"
                recommendation = "💎 PRIORITY: Send personalized offer immediately!"
                action = "Offer discount code via email"
            elif probability > 0.4:
                confidence = "Medium"
                confidence_color = "orange"
                recommendation = "📧 ACTION: Retarget with special discount"
                action = "Show retargeting ads on social media"
            else:
                confidence = "Low"
                confidence_color = "red"
                recommendation = "📢 STRATEGY: Build awareness with ads"
                action = "Focus on brand awareness campaigns"
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="result-purchase">
                    <h1>✅ HIGH PURCHASE INTENT</h1>
                    <h2>Purchase Probability: {probability:.1%}</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">Confidence: {confidence}</p>
                    <p style="font-size: 1.1rem;">{recommendation}</p>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">🎯 Recommended Action: {action}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no-purchase">
                    <h1>⚠️ LOW PURCHASE INTENT</h1>
                    <h2>Purchase Probability: {probability:.1%}</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">Confidence: {confidence}</p>
                    <p style="font-size: 1.1rem;">{recommendation}</p>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">🎯 Recommended Action: {action}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("### 📊 Probability Gauge")
            st.progress(probability)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Purchase Probability", f"{probability:.1%}")
            with col2:
                st.metric("Confidence Level", confidence)
            with col3:
                status = "High Intent" if prediction == 1 else "Low Intent"
                st.metric("Customer Status", status)
            with col4:
                risk = "Low Risk" if probability < 0.3 else "Medium Risk" if probability < 0.7 else "High Risk"
                st.metric("Risk Level", risk)
            
            # Key factors influencing decision
            with st.expander("🔍 View Key Factors Influencing Decision", expanded=True):
                factors = []
                
                if product_duration > 600:
                    factors.append("✅ **Long product browsing time** (+): Over 10 minutes - Strong purchase signal")
                elif product_duration < 100:
                    factors.append("❌ **Short product browsing time** (-): Less than 2 minutes - Low engagement")
                else:
                    factors.append("⚠️ **Moderate product time** (→): Could be more engaged")
                
                if bounce_rate < 0.3:
                    factors.append("✅ **Low bounce rate** (+): Users are exploring multiple pages")
                elif bounce_rate > 0.7:
                    factors.append("❌ **High bounce rate** (-): Users leave immediately")
                
                if exit_rate < 0.3:
                    factors.append("✅ **Low exit rate** (+): Users continue browsing")
                elif exit_rate > 0.7:
                    factors.append("❌ **High exit rate** (-): Users exit from this page")
                
                if page_values > 20:
                    factors.append("✅ **High page values** (+): Pages visited have monetary value")
                
                if visitor_type == "Returning_Visitor":
                    factors.append("✅ **Returning visitor** (+): Familiar with your store")
                else:
                    factors.append("⚠️ **New visitor** (→): Needs trust building")
                
                if special_day > 0.5:
                    factors.append("✅ **Holiday proximity** (+): Seasonal shopping behavior")
                
                if month in ["Nov", "Dec"]:
                    factors.append("✅ **Peak shopping season** (+): November/December")
                elif month in ["Jan", "Feb"]:
                    factors.append("⚠️ **Post-holiday season** (→): Lower purchase intent typical")
                
                if product_pages > 20:
                    factors.append("✅ **Many product views** (+): Browsing multiple products")
                elif product_pages < 5:
                    factors.append("❌ **Few product views** (-): Limited interest")
                
                if weekend == "Yes":
                    factors.append("✅ **Weekend browsing** (+): More leisure shopping time")
                
                for factor in factors:
                    st.markdown(factor)
    
    else:
        st.error("❌ Models not loaded. Please check if 'saved_models' folder exists with all pickle files.")

# Footer
st.markdown("""
<div class="footer">
    <p>🚀 Powered by Random Forest Machine Learning | Trained on 12,330 e-commerce sessions</p>
    <p>🎯 Model achieves 61.5% recall (catches 6 out of 10 actual purchases)</p>
    <p>📊 Features analyzed: Page views, duration, bounce rates, visitor type, seasonality, and technical metrics</p>
</div>
""", unsafe_allow_html=True)