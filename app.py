import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digital Marketing Dashboard", layout="wide")

st.sidebar.title("لوحة التحكم")
page = st.sidebar.radio("اختر المشروع:", ["تحليل الحملات (Regression)", "توقع التحويلات (Classification)"])

@st.cache_resource
def load_classification_models():
    # تحميل النموذج وأداة التسوية
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

if page == "تحليل الحملات (Regression)":
    st.title("تحليل الحملات المدمجة (Nykaa, Purplle, Tira)")
    st.info("قم بإضافة كود استكشاف البيانات والنمذجة الخاص بالانحدار هنا لاحقاً.")

elif page == "توقع التحويلات (Classification)":
    st.title("توقع نجاح الحملات التسويقية (Conversion Prediction)")
    st.write("أدخل بيانات العميل وتفاصيل الحملة لمعرفة ما إذا كان سيقوم بالتحويل (Conversion) أم لا.")
    
    try:
        model_clf, scaler = load_classification_models()
        
        # إنشاء نموذج إدخال البيانات
        with st.form("prediction_form"):
            st.subheader("بيانات العميل")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("العمر (Age)", min_value=18, max_value=100, value=30)
                gender = st.selectbox("الجنس (Gender)", ["Female", "Male"])
            with col2:
                income = st.number_input("الدخل (Income)", min_value=10000, max_value=200000, value=50000)
                loyalty_points = st.number_input("نقاط الولاء (Loyalty Points)", min_value=0, value=1000)
            with col3:
                previous_purchases = st.number_input("المشتريات السابقة", min_value=0, value=2)
            
            st.subheader("تفاعل العميل مع الموقع والبريد")
            col4, col5, col6 = st.columns(3)
            with col4:
                website_visits = st.number_input("زيارات الموقع (Website Visits)", min_value=0, value=10)
                pages_per_visit = st.number_input("الصفحات لكل زيارة", min_value=1.0, value=3.0)
            with col5:
                time_on_site = st.number_input("الوقت على الموقع بالدقائق", min_value=0.0, value=5.0)
                social_shares = st.number_input("المشاركات الاجتماعية", min_value=0, value=10)
            with col6:
                email_opens = st.number_input("مرات فتح البريد", min_value=0, value=5)
                email_clicks = st.number_input("النقرات داخل البريد", min_value=0, value=2)

            st.subheader("بيانات الحملة التسويقية")
            col7, col8, col9 = st.columns(3)
            with col7:
                ad_spend = st.number_input("حجم الإنفاق الإعلاني (Ad Spend)", min_value=0.0, value=1500.0)
                click_through_rate = st.number_input("نسبة النقر للظهور (CTR)", min_value=0.0, max_value=1.0, value=0.1)
            with col8:
                conversion_rate = st.number_input("معدل التحويل (Conversion Rate)", min_value=0.0, max_value=1.0, value=0.05)
                campaign_channel = st.selectbox("قناة الحملة (Channel)", ["Email", "PPC", "Referral", "SEO", "Social Media"])
            with col9:
                campaign_type = st.selectbox("نوع الحملة (Type)", ["Awareness", "Consideration", "Conversion", "Retention"])
            
            submit_button = st.form_submit_button("توقع النتيجة")

        if submit_button:
            # 1. تحويل المدخلات إلى DataFrame كما يتوقعها النموذج بالضبط
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [1 if gender == "Male" else 0], # تحويل الجنس إلى 0 و 1 بناءً على كودك
                'Income': [income],
                'AdSpend': [ad_spend],
                'ClickThroughRate': [click_through_rate],
                'ConversionRate': [conversion_rate],
                'WebsiteVisits': [website_visits],
                'PagesPerVisit': [pages_per_visit],
                'TimeOnSite': [time_on_site],
                'SocialShares': [social_shares],
                'EmailOpens': [email_opens],
                'EmailClicks': [email_clicks],
                'PreviousPurchases': [previous_purchases],
                'LoyaltyPoints': [loyalty_points],
                # One-Hot Encoding للأعمدة المتبقية تماماً كما تم في Jupyter
                'CampaignChannel_PPC': [1 if campaign_channel == "PPC" else 0],
                'CampaignChannel_Referral': [1 if campaign_channel == "Referral" else 0],
                'CampaignChannel_SEO': [1 if campaign_channel == "SEO" else 0],
                'CampaignChannel_Social Media': [1 if campaign_channel == "Social Media" else 0],
                'CampaignType_Consideration': [1 if campaign_type == "Consideration" else 0],
                'CampaignType_Conversion': [1 if campaign_type == "Conversion" else 0],
                'CampaignType_Retention': [1 if campaign_type == "Retention" else 0]
            })

            # 2. تسوية البيانات (Scaling)
            input_scaled = scaler.transform(input_data)
            
            # 3. التوقع
            prediction = model_clf.predict(input_scaled)
            probability = model_clf.predict_proba(input_scaled)[0][1]

            # 4. عرض النتائج
            st.markdown("---")
            if prediction[0] == 1:
                st.success(f"🎉 نتيجة التوقع: العميل سيقوم بالتحويل (Converted)! بنسبة احتمال {probability:.1%}")
            else:
                st.error(f"⚠️ نتيجة التوقع: العميل لن يقوم بالتحويل. بنسبة احتمال {(1-probability):.1%}")

    except FileNotFoundError:
        st.warning("يرجى التأكد من رفع ملفات `best_model.pkl` و `scaler.pkl` إلى GitHub.")
