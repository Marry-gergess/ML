import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Digital Marketing Dashboard", layout="wide")

st.sidebar.title("لوحة التحكم")
page = st.sidebar.radio("اختر المشروع:", ["تحليل الحملات المدمجة (Regression)", "تصنيف التحويلات (Classification)"])

# دالة لتحميل البيانات بفعالية
@st.cache_data
def load_merged_data():
    nykaa = pd.read_csv("nykaa_campaign_data.csv")
    nykaa["Brand"] = "Nykaa"
    purplle = pd.read_csv("purplle_campaign_data.csv")
    purplle["Brand"] = "Purplle"
    tira = pd.read_csv("tira_campaign_data.csv")
    tira["Brand"] = "Tira"
    return pd.concat([nykaa, purplle, tira], ignore_index=True)

@st.cache_data
def load_classification_data():
    return pd.read_csv("digital_marketing_campaign_dataset.csv")

if page == "تحليل الحملات المدمجة (Regression)":
    st.title("تحليل الحملات المدمجة (Nykaa, Purplle, Tira)")
    df_merged = load_merged_data()
    
    st.write("عينة من البيانات المدمجة:", df_merged.head())
    
    st.subheader("إحصائيات أساسية")
    st.write(df_merged.describe())
    
    # يمكنك إضافة واجهة تحميل النموذج هنا
    # model_reg = joblib.load('regression_model.pkl')
    st.info("لإضافة التوقع، تأكد من تصدير نموذج الـ Regression بصيغة pkl.")

elif page == "تصنيف التحويلات (Classification)":
    st.title("تصنيف نجاح الحملات التسويقية")
    df_class = load_classification_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("توزيع التحويلات (Conversions)")
        fig, ax = plt.subplots()
        df_class['Conversion'].value_counts(normalize=True).mul(100).plot(kind="bar", ax=ax, color=['salmon', 'lightblue'])
        st.pyplot(fig)
        
    with col2:
        st.subheader("عينة البيانات")
        st.write(df_class.head())
    
    st.subheader("تجربة النموذج (Prediction)")
    try:
        model_clf = joblib.load('best_model.pkl')
        st.success("تم تحميل النموذج بنجاح! يمكنك إضافة حقول الإدخال (Inputs) هنا ليتفاعل معها المستخدم.")
    except FileNotFoundError:
        st.warning("يرجى التأكد من رفع ملف best_model.pkl إلى GitHub.")
