import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Marketing Analytics AI", layout="wide", page_icon="📈")

# --- Load Models ---
@st.cache_resource
def load_regression_model():
    return joblib.load('regression_model.pkl')

@st.cache_resource
def load_classification_models():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --- Load Data for EDA ---
@st.cache_data
def load_merged_data():
    nykaa = pd.read_csv("nykaa_campaign_data.csv")
    nykaa["Brand"] = "Nykaa"
    purplle = pd.read_csv("purplle_campaign_data.csv")
    purplle["Brand"] = "Purplle"
    tira = pd.read_csv("tira_campaign_data.csv")
    tira["Brand"] = "Tira"
    return pd.concat([nykaa, purplle, tira], ignore_index=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a Machine Learning Project:")
page = st.sidebar.radio("", ["1. Conversions Predictor (Regression)", "2. Success Classifier (Classification)"])

st.sidebar.markdown("---")
st.sidebar.info("This dashboard uses Machine Learning to analyze and predict marketing campaign performances.")

# ==========================================
# PAGE 1: REGRESSION (Merged Data)
# ==========================================
if page == "1. Conversions Predictor (Regression)":
    st.title("📈 Campaign Conversions Predictor")
    st.markdown("Predict the expected number of conversions based on campaign metrics and audience data across Nykaa, Purplle, and Tira.")
    
    try:
        reg_model = load_regression_model()
        
        with st.form("regression_form"):
            st.subheader("Campaign Performance Metrics")
            col1, col2, col3 = st.columns(3)
            impressions = col1.number_input("Impressions", min_value=0, value=50000)
            clicks = col2.number_input("Clicks", min_value=0, value=5000)
            leads = col3.number_input("Leads", min_value=0, value=1500)
            
            st.subheader("Audience & Details")
            col4, col5, col6 = st.columns(3)
            brand = col4.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
            campaign_type = col5.selectbox("Campaign Type", ["Influencer", "Email", "Paid Ads", "SEO", "Social Media"])
            language = col6.selectbox("Language", ["Hindi", "Tamil", "Bengali", "English"])
            
            col7, col8, col9 = st.columns(3)
            target_audience = col7.selectbox("Target Audience", ["Premium Shoppers", "Working Women", "Youth", "Tier 2 City Customers", "College Students"])
            customer_segment = col8.selectbox("Customer Segment", ["Youth", "College Students", "Working Women", "Tier 2 City Customers", "Premium Shoppers"])
            duration = col9.number_input("Duration (Days)", min_value=1, value=15)
            
            col10, col11 = st.columns(2)
            year = col10.selectbox("Year", [2024, 2025])
            month = col11.slider("Month", 1, 12, 6)
            
            st.subheader("Channels Used (Check all that apply)")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            fb = c1.checkbox("Facebook", value=True)
            wa = c2.checkbox("WhatsApp")
            ggl = c3.checkbox("Google", value=True)
            yt = c4.checkbox("YouTube")
            ig = c5.checkbox("Instagram")
            eml = c6.checkbox("Email")
            
            submit_reg = st.form_submit_button("Predict Conversions")
            
        if submit_reg:
            # Create DataFrame with exact column names expected by the pipeline
            input_df = pd.DataFrame({
                "Impressions": [impressions], "Clicks": [clicks], "Leads": [leads],
                "Brand": [brand], "Campaign_Type": [campaign_type], 
                "Target_Audience": [target_audience], "Language": [language], 
                "Customer_Segment": [customer_segment], "Duration": [duration], 
                "Year": [year], "Month": [month],
                "Facebook": [int(fb)], "WhatsApp": [int(wa)], "Google": [int(ggl)], 
                "YouTube": [int(yt)], "Instagram": [int(ig)], "Email": [int(eml)]
            })
            
            # Predict (The pipeline handles encoding)
            pred_log = reg_model.predict(input_df)[0]
            # Reverse the log1p transformation
            pred_real = np.expm1(pred_log)
            
            st.success(f"🎯 Estimated Conversions: **{int(pred_real):,}** conversions")
            
    except FileNotFoundError:
        st.error("Missing model file. Please ensure 'regression_model.pkl' is uploaded.")

# ==========================================
# PAGE 2: CLASSIFICATION
# ==========================================
elif page == "2. Success Classifier (Classification)":
    st.title("🎯 Campaign Success Classifier")
    st.markdown("Enter customer details to classify whether they will successfully convert (1) or not (0).")
    
    try:
        clf_model, scaler = load_classification_models()
        
        with st.form("classification_form"):
            st.subheader("Customer Profile")
            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=18, max_value=100, value=30)
            gender = col2.selectbox("Gender", ["Female", "Male"])
            income = col3.number_input("Income ($)", min_value=10000, value=50000)
            
            loyalty_points = col1.number_input("Loyalty Points", min_value=0, value=1000)
            prev_purchases = col2.number_input("Previous Purchases", min_value=0, value=2)
            
            st.subheader("Engagement Metrics")
            c1, c2, c3, c4 = st.columns(4)
            web_visits = c1.number_input("Website Visits", min_value=0, value=10)
            pages_visit = c2.number_input("Pages Per Visit", min_value=1.0, value=3.0)
            time_site = c3.number_input("Time On Site (min)", min_value=0.0, value=5.0)
            social_shares = c4.number_input("Social Shares", min_value=0, value=10)
            
            c5, c6 = st.columns(2)
            email_opens = c5.number_input("Email Opens", min_value=0, value=5)
            email_clicks = c6.number_input("Email Clicks", min_value=0, value=2)

            st.subheader("Campaign Details")
            col7, col8, col9 = st.columns(3)
            ad_spend = col7.number_input("Ad Spend ($)", min_value=0.0, value=1500.0)
            ctr = col8.number_input("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.1)
            conv_rate = col9.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)
            
            col10, col11 = st.columns(2)
            channel = col10.selectbox("Campaign Channel", ["Email", "PPC", "Referral", "SEO", "Social Media"])
            camp_type = col11.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
            
            submit_clf = st.form_submit_button("Predict Conversion")

        if submit_clf:
            # Map Categorical Variables manually for classification model
            input_dict = {
                'Age': [age],
                'Gender': [1 if gender == "Male" else 0],
                'Income': [income],
                'AdSpend': [ad_spend],
                'ClickThroughRate': [ctr],
                'ConversionRate': [conv_rate],
                'WebsiteVisits': [web_visits],
                'PagesPerVisit': [pages_visit],
                'TimeOnSite': [time_site],
                'SocialShares': [social_shares],
                'EmailOpens': [email_opens],
                'EmailClicks': [email_clicks],
                'PreviousPurchases': [prev_purchases],
                'LoyaltyPoints': [loyalty_points],
                'CampaignChannel_PPC': [1 if channel == "PPC" else 0],
                'CampaignChannel_Referral': [1 if channel == "Referral" else 0],
                'CampaignChannel_SEO': [1 if channel == "SEO" else 0],
                'CampaignChannel_Social Media': [1 if channel == "Social Media" else 0],
                'CampaignType_Consideration': [1 if camp_type == "Consideration" else 0],
                'CampaignType_Conversion': [1 if camp_type == "Conversion" else 0],
                'CampaignType_Retention': [1 if camp_type == "Retention" else 0]
            }
            
            input_df = pd.DataFrame(input_dict)
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = clf_model.predict(input_scaled)[0]
            probability = clf_model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.success(f"✅ **Prediction: CONVERTED** (Probability: {probability:.1%})")
            else:
                st.error(f"❌ **Prediction: NOT CONVERTED** (Probability: {(1-probability):.1%})")

    except FileNotFoundError:
        st.error("Missing model files. Please ensure 'best_model.pkl' and 'scaler.pkl' are uploaded.")
