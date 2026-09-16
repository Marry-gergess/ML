import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Digital Marketing Dashboard", layout="wide")

# تحميل البيانات والنموذج
@st.cache_data
def load_data():
    return pd.read_csv("digital_marketing_campaign_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

df = load_data()
model = load_model()

# القائمة الجانبية
st.sidebar.title("القائمة")
page = st.sidebar.radio("اختر الصفحة:", ["استكشاف البيانات (EDA)", "التوقع (Prediction)"])

if page == "استكشاف البيانات (EDA)":
    st.title("تحليل بيانات الحملات التسويقية")
    st.write(df.head())
    
    st.subheader("مصفوفة الارتباط (Correlation Matrix)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "التوقع (Prediction)":
    st.title("توقع التحويلات (Conversions)")
    st.write("أدخل بيانات العميل والحملة لمعرفة احتمالية التحويل.")
    
    # مثال لواجهة إدخال البيانات (يجب إكمال باقي المتغيرات بنفس الطريقة)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("العمر", min_value=18, max_value=100, value=30)
        income = st.number_input("الدخل", min_value=0, value=50000)
    with col2:
        ad_spend = st.number_input("الإنفاق الإعلاني", min_value=0.0, value=1000.0)
        clicks = st.number_input("النقرات", min_value=0, value=100)
    
    # زر التوقع
    if st.button("توقع النتيجة"):
        # هنا ستقوم بجمع المتغيرات في DataFrame جديد وتمريرها للنموذج
        # prediction = model.predict(input_df)
        # st.success(f"النتيجة: {prediction[0]}")
        st.info("يجب تمرير البيانات بنفس شكل الأعمدة المستخدمة أثناء تدريب النموذج.")
