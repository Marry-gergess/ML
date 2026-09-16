import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Digital Marketing Dashboard", layout="wide")

st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Select Project:", ["Campaign Analysis (Regression)", "Conversion Prediction (Classification)"])

@st.cache_data
def load_merged_data():
    nykaa = pd.read_csv("nykaa_campaign_data.csv")
    nykaa["Brand"] = "Nykaa"
    purplle = pd.read_csv("purplle_campaign_data.csv")
    purplle["Brand"] = "Purplle"
    tira = pd.read_csv("tira_campaign_data.csv")
    tira["Brand"] = "Tira"
    return pd.concat([nykaa, purplle, tira], ignore_index=True)

@st.cache_resource
def load_classification_models():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

if page == "Campaign Analysis (Regression)":
    st.markdown("<h1 style='text-align: left;'>Merged Campaigns Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: gray;'>Nykaa, Purplle, Tira</h3>", unsafe_allow_html=True)
    
    try:
        df_merged = load_merged_data()
        
        st.markdown("### Merged Data Sample")
        st.dataframe(df_merged.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Campaigns per Brand")
            st.dataframe(df_merged["Brand"].value_counts())
            
        with col2:
            st.markdown("### Conversions Statistics")
            st.dataframe(df_merged["Conversions"].describe())
            
        st.markdown("---")
        st.markdown("### ROI Comparison by Brand")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_merged, x="Brand", y="ROI", palette="viridis", ax=ax)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Please make sure to upload nykaa_campaign_data.csv, purplle_campaign_data.csv, and tira_campaign_data.csv to the repository.")

elif page == "Conversion Prediction (Classification)":
    st.title("Marketing Campaign Conversion Prediction")
    st.write("Enter customer and campaign details to predict the conversion probability.")
    
    try:
        model_clf, scaler = load_classification_models()
        
        with st.form("prediction_form"):
            st.subheader("Customer Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", ["Female", "Male"])
            with col2:
                income = st.number_input("Income", min_value=10000, max_value=200000, value=50000)
                loyalty_points = st.number_input("Loyalty Points", min_value=0, value=1000)
            with col3:
                previous_purchases = st.number_input("Previous Purchases", min_value=0, value=2)
            
            st.subheader("Engagement Metrics")
            col4, col5, col6 = st.columns(3)
            with col4:
                website_visits = st.number_input("Website Visits", min_value=0, value=10)
                pages_per_visit = st.number_input("Pages Per Visit", min_value=1.0, value=3.0)
            with col5:
                time_on_site = st.number_input("Time On Site (minutes)", min_value=0.0, value=5.0)
                social_shares = st.number_input("Social Shares", min_value=0, value=10)
            with col6:
                email_opens = st.number_input("Email Opens", min_value=0, value=5)
                email_clicks = st.number_input("Email Clicks", min_value=0, value=2)

            st.subheader("Campaign Metrics")
            col7, col8, col9 = st.columns(3)
            with col7:
                ad_spend = st.number_input("Ad Spend", min_value=0.0, value=1500.0)
                click_through_rate = st.number_input("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.1)
            with col8:
                conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)
                campaign_channel = st.selectbox("Campaign Channel", ["Email", "PPC", "Referral", "SEO", "Social Media"])
            with col9:
                campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
            
            submit_button = st.form_submit_button("Predict Conversion")

        if submit_button:
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [1 if gender == "Male" else 0],
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
                'CampaignChannel_PPC': [1 if campaign_channel == "PPC" else 0],
                'CampaignChannel_Referral': [1 if campaign_channel == "Referral" else 0],
                'CampaignChannel_SEO': [1 if campaign_channel == "SEO" else 0],
                'CampaignChannel_Social Media': [1 if campaign_channel == "Social Media" else 0],
                'CampaignType_Consideration': [1 if campaign_type == "Consideration" else 0],
                'CampaignType_Conversion': [1 if campaign_type == "Conversion" else 0],
                'CampaignType_Retention': [1 if campaign_type == "Retention" else 0]
            })

            input_scaled = scaler.transform(input_data)
            
            prediction = model_clf.predict(input_scaled)
            probability = model_clf.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction[0] == 1:
                st.success(f"Prediction: The customer is likely to CONVERT. (Probability: {probability:.1%})")
            else:
                st.error(f"Prediction: The customer is NOT likely to convert. (Probability: {(1-probability):.1%})")

    except FileNotFoundError:
        st.warning("Please ensure that both 'best_model.pkl' and 'scaler.pkl' are uploaded to the repository.")
