import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعدادات الصفحة
st.set_page_config(page_title="Digital Marketing Prediction App", layout="wide")

# دالة لتحميل الموديلات عشان متعملش Load مع كل تغيير في الصفحة
@st.cache_resource
def load_models():
    reg_model = joblib.load('regression_model.pkl')
    clf_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl') # لازم ترفع الفايل ده كمان مع الداتا
    return reg_model, clf_model, scaler

try:
    reg_model, clf_model, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure regression_model.pkl, best_model.pkl, and scaler.pkl are in the same folder.")

# القائمة الجانبية للتنقل
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Choose a Model:", ["📈 Campaign Conversions Predictor (Regression)", "👥 Customer Conversion Classifier (Classification)"])

# ---------------------------------------------------------
# الموديل الأول: Regression
# ---------------------------------------------------------
if page == "📈 Campaign Conversions Predictor (Regression)":
    st.title("📈 Campaign Conversions Predictor")
    st.write("Enter the campaign details below to predict the number of conversions.")

    # تقسيم الشاشة لأعمدة عشان الشكل يكون منظم
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
        target_audience = st.selectbox("Target Audience", ["Youth", "College Students", "Working Women", "Tier 2 City Customers", "Premium Shoppers"])
        language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Bengali"])
        customer_segment = st.selectbox("Customer Segment", ["Youth", "College Students", "Working Women", "Tier 2 City Customers", "Premium Shoppers"])

    with col2:
        impressions = st.number_input("Impressions", min_value=0, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=5000)
        leads = st.number_input("Leads", min_value=0, value=1000)
        duration = st.number_input("Duration (Days)", min_value=1, value=15)
        year = st.selectbox("Year", [2024, 2025, 2026])
        month = st.slider("Month", 1, 12, 6)

    with col3:
        st.write("Channels Used (Check all that apply):")
        facebook = 1 if st.checkbox("Facebook") else 0
        whatsapp = 1 if st.checkbox("WhatsApp") else 0
        google = 1 if st.checkbox("Google") else 0
        youtube = 1 if st.checkbox("YouTube") else 0
        instagram = 1 if st.checkbox("Instagram") else 0
        email = 1 if st.checkbox("Email") else 0

    if st.button("Predict Conversions 🚀"):
        # تجميع المدخلات في DataFrame
        input_data = pd.DataFrame({
            "Impressions": [impressions], "Clicks": [clicks], "Leads": [leads],
            "Brand": [brand], "Campaign_Type": [campaign_type], "Target_Audience": [target_audience], 
            "Language": [language], "Customer_Segment": [customer_segment],
            "Duration": [duration], "Year": [year], "Month": [month],
            "Facebook": [facebook], "WhatsApp": [whatsapp], "Google": [google], 
            "YouTube": [youtube], "Instagram": [instagram], "Email": [email]
        })

        # التوقع (لاحظ إني استخدمت expm1 عشان أرجع الرقم لأصله زي ما عملت في الكود بتاعك)
        log_pred = reg_model.predict(input_data)[0]
        actual_pred = np.expm1(log_pred)
        
        st.success(f"🎉 Predicted Conversions: **{int(actual_pred)}**")

# ---------------------------------------------------------
# الموديل التاني: Classification
# ---------------------------------------------------------
else:
    st.title("👥 Customer Conversion Classifier")
    st.write("Enter customer and campaign details to predict if they will convert or not.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Income", min_value=0, value=50000)
        loyalty_points = st.number_input("Loyalty Points", min_value=0, value=1000)
        previous_purchases = st.number_input("Previous Purchases", min_value=0, value=2)

    with col2:
        campaign_channel = st.selectbox("Campaign Channel", ["Email", "PPC", "Referral", "SEO", "Social Media"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
        ad_spend = st.number_input("Ad Spend", min_value=0.0, value=1000.0)
        ctr = st.number_input("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.1)
        conv_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)

    with col3:
        website_visits = st.number_input("Website Visits", min_value=0, value=10)
        pages_per_visit = st.number_input("Pages Per Visit", min_value=1.0, value=3.0)
        time_on_site = st.number_input("Time On Site (mins)", min_value=0.0, value=5.0)
        social_shares = st.number_input("Social Shares", min_value=0, value=5)
        email_opens = st.number_input("Email Opens", min_value=0, value=2)
        email_clicks = st.number_input("Email Clicks", min_value=0, value=1)

    if st.button("Predict Conversion 🎯"):
        # تحويل الداتا عشان تناسب التدريب
        gender_map = 1 if gender == "Male" else 0
        
        # إنشاء الأعمدة الوهمية (Dummies) بنفس ترتيب الكود بتاعك
        # القنوات: PPC, Referral, SEO, Social Media (Email is base/0)
        ch_ppc = 1 if campaign_channel == "PPC" else 0
        ch_ref = 1 if campaign_channel == "Referral" else 0
        ch_seo = 1 if campaign_channel == "SEO" else 0
        ch_sm = 1 if campaign_channel == "Social Media" else 0
        
        # الأنواع: Consideration, Conversion, Retention (Awareness is base/0)
        type_cons = 1 if campaign_type == "Consideration" else 0
        type_conv = 1 if campaign_type == "Conversion" else 0
        type_ret = 1 if campaign_type == "Retention" else 0

        # ترتيب الأعمدة زي ما دخلت للـ Scaler والموديل بالظبط
        input_array = np.array([[
            age, gender_map, income, ad_spend, ctr, conv_rate, website_visits, 
            pages_per_visit, time_on_site, social_shares, email_opens, email_clicks, 
            previous_purchases, loyalty_points, 
            ch_ppc, ch_ref, ch_seo, ch_sm, 
            type_cons, type_conv, type_ret
        ]])

        # عمل Scaling
        input_scaled = scaler.transform(input_array)
        
        # التوقع
        prediction = clf_model.predict(input_scaled)[0]
        probability = clf_model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ The customer is likely to **Convert**. (Confidence: {probability:.2%})")
        else:
            st.error(f"❌ The customer is **NOT** likely to convert. (Confidence: {1 - probability:.2%})")
