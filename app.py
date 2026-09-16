import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# إعدادات الصفحة
st.set_page_config(page_title="Marketing AI Predictor", layout="wide")

# دالة لتحميل الموديلات من جوجل درايف لو مش موجودة
@st.cache_resource
def load_models():
    # حطي الـ IDs بتاعة الملفات من لينكات جوجل درايف بتاعتك
    # (هقولك إزاي تجيبي الـ ID تحت الكود)
    files_to_download = {
        'regression_model.pkl': 'هنا_حطي_الـ_ID_بتاع_الموديل_الأول',
        'best_model.pkl': 'هنا_حطي_الـ_ID_بتاع_الموديل_التاني',
        'scaler.pkl': 'هنا_حطي_الـ_ID_بتاع_الـscaler'
    }
    
    # تحميل الملفات لو مش موجودة
    for filename, file_id in files_to_download.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
            
    # قراءة الموديلات بعد التحميل
    reg_model = joblib.load('regression_model.pkl')
    clf_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    return reg_model, clf_model, scaler

try:
    with st.spinner("Downloading models from Drive... Please wait ⏳"):
        reg_model, clf_model, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")


# القائمة الجانبية (Sidebar)
st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.radio("Choose a Model:", 
                            ["Campaign Conversions Predictor", "User Conversion Classifier"])

# =========================================================================
# Model 1: Regression (Campaign Conversions)
# =========================================================================
if app_mode == "Campaign Conversions Predictor":
    st.title("📈 Campaign Conversions Predictor")
    st.markdown("Enter the campaign details below to predict the number of conversions.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Campaign Type", ["Influencer", "Email", "Paid Ads", "SEO", "Social Media"])
        target_audience = st.selectbox("Target Audience", ["Premium Shoppers", "Working Women", "Youth", "Tier 2 City Customers", "College Students"])
        customer_segment = st.selectbox("Customer Segment", ["Youth", "College Students", "Working Women", "Tier 2 City Customers", "Premium Shoppers"])
        language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Bengali"])
        
    with col2:
        impressions = st.number_input("Impressions", min_value=0, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=5000)
        leads = st.number_input("Leads", min_value=0, value=1000)
        duration = st.number_input("Duration (Days)", min_value=1, value=15)
        year = st.number_input("Year", min_value=2000, value=2024, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        
    with col3:
        st.write("Channels Used (Check all that apply):")
        facebook = st.checkbox("Facebook")
        whatsapp = st.checkbox("WhatsApp")
        google = st.checkbox("Google")
        youtube = st.checkbox("YouTube")
        instagram = st.checkbox("Instagram")
        email = st.checkbox("Email")
        
    if st.button("Predict Conversions 🚀"):
        # تجميع المدخلات في DataFrame بنفس أسماء الأعمدة في d1.py
        input_data = pd.DataFrame({
            "Impressions": [impressions],
            "Clicks": [clicks],
            "Leads": [leads],
            "Brand": [brand],
            "Campaign_Type": [campaign_type],
            "Target_Audience": [target_audience],
            "Language": [language],
            "Customer_Segment": [customer_segment],
            "Duration": [duration],
            "Year": [year],
            "Month": [month],
            "Facebook": [int(facebook)],
            "WhatsApp": [int(whatsapp)],
            "Google": [int(google)],
            "YouTube": [int(youtube)],
            "Instagram": [int(instagram)],
            "Email": [int(email)],
        })
        
        # التوقع (نستخدم expm1 عشان نرجع الرقم لأصله)
        pred_log = reg_model.predict(input_data)[0]
        prediction = np.expm1(pred_log)
        
        st.success(f"### Predicted Conversions: {int(prediction):,}")


# =========================================================================
# Model 2: Classification (User Conversion)
# =========================================================================
elif app_mode == "User Conversion Classifier":
    st.title("🎯 User Conversion Classifier")
    st.markdown("Enter user and campaign metrics to predict if they will convert (1) or not (0).")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Income ($)", min_value=0, value=50000)
        ad_spend = st.number_input("Ad Spend ($)", min_value=0.0, value=1000.0)
        ctr = st.number_input("Click-Through Rate", min_value=0.0, max_value=1.0, value=0.1)
        cvr = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)
        
    with col2:
        visits = st.number_input("Website Visits", min_value=0, value=5)
        pages = st.number_input("Pages Per Visit", min_value=0.0, value=3.0)
        time_site = st.number_input("Time On Site (mins)", min_value=0.0, value=5.0)
        shares = st.number_input("Social Shares", min_value=0, value=10)
        email_opens = st.number_input("Email Opens", min_value=0, value=2)
        email_clicks = st.number_input("Email Clicks", min_value=0, value=1)
        
    with col3:
        prev_purchases = st.number_input("Previous Purchases", min_value=0, value=1)
        loyalty = st.number_input("Loyalty Points", min_value=0, value=500)
        campaign_channel = st.selectbox("Campaign Channel", ["Email", "PPC", "Referral", "SEO", "Social Media"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
        
    if st.button("Predict Conversion Status 🔍"):
        
        # بناء الداتا فريم بـ 21 عمود بالظبط زي ما الـ Scaler والموديل متوقعين
        clf_input = {
            'Age': age,
            'Gender': 1 if gender == 'Male' else 0,
            'Income': income,
            'AdSpend': ad_spend,
            'ClickThroughRate': ctr,
            'ConversionRate': cvr,
            'WebsiteVisits': visits,
            'PagesPerVisit': pages,
            'TimeOnSite': time_site,
            'SocialShares': shares,
            'EmailOpens': email_opens,
            'EmailClicks': email_clicks,
            'PreviousPurchases': prev_purchases,
            'LoyaltyPoints': loyalty,
            
            # Dummy variables for CampaignChannel (drop_first -> Email is dropped)
            'CampaignChannel_PPC': 1 if campaign_channel == 'PPC' else 0,
            'CampaignChannel_Referral': 1 if campaign_channel == 'Referral' else 0,
            'CampaignChannel_SEO': 1 if campaign_channel == 'SEO' else 0,
            'CampaignChannel_Social Media': 1 if campaign_channel == 'Social Media' else 0,
            
            # Dummy variables for CampaignType (drop_first -> Awareness is dropped)
            'CampaignType_Consideration': 1 if campaign_type == 'Consideration' else 0,
            'CampaignType_Conversion': 1 if campaign_type == 'Conversion' else 0,
            'CampaignType_Retention': 1 if campaign_type == 'Retention' else 0,
        }
        
        df_clf = pd.DataFrame([clf_input])
        
        # Scaling
        scaled_data = scaler.transform(df_clf)
        
        # Prediction
        prediction = clf_model.predict(scaled_data)[0]
        
        if hasattr(clf_model, "predict_proba"):
            prob = clf_model.predict_proba(scaled_data)[0][1]
            prob_text = f"(Probability: {prob:.2%})"
        else:
            prob_text = ""
            
        if prediction == 1:
            st.success(f"### Result: Converted (Yes) ✅ {prob_text}")
        else:
            st.error(f"### Result: Not Converted (No) ❌ {prob_text}")
