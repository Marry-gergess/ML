if page == "تحليل الحملات (Regression)":
    # حل مشكلة النص المقلوب باستخدام HTML لضبط الاتجاه
    st.markdown("<h1 style='text-align: right; direction: rtl;'>تحليل الحملات المدمجة</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: right; color: gray;'>Nykaa, Purplle, Tira</h3>", unsafe_allow_html=True)
    
    # دالة لدمج وقراءة البيانات
    @st.cache_data
    def load_merged_data():
        nykaa = pd.read_csv("nykaa_campaign_data.csv")
        nykaa["Brand"] = "Nykaa"
        purplle = pd.read_csv("purplle_campaign_data.csv")
        purplle["Brand"] = "Purplle"
        tira = pd.read_csv("tira_campaign_data.csv")
        tira["Brand"] = "Tira"
        return pd.concat([nykaa, purplle, tira], ignore_index=True)
        
    try:
        df_merged = load_merged_data()
        
        st.markdown("### عينة من البيانات المدمجة")
        st.dataframe(df_merged.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### توزيع الحملات حسب المنصة")
            st.dataframe(df_merged["Brand"].value_counts())
            
        with col2:
            st.markdown("### إحصائيات (Conversions)")
            st.dataframe(df_merged["Conversions"].describe())
            
        st.markdown("---")
        st.markdown("### مقارنة العائد على الاستثمار (ROI) حسب المنصة")
        
        # رسم بياني باستخدام Seaborn
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_merged, x="Brand", y="ROI", palette="viridis", ax=ax)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("يرجى التأكد من رفع ملفات البيانات (nykaa, purplle, tira) إلى المستودع.")
