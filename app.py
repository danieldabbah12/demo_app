# --- עמוד 2: ניתוח נתונים וסטטיסטיקה ---
elif page == "ניתוח נתונים וסטטיסטיקה":
    st.title("📊 ניתוח נתונים וסטטיסטיקה פדגוגית")
    st.write("בעמוד זה נבחן את 300 הרשומות שנוצרו עבור המודל כדי להבין את הקשרים בין המשתנים השונים.")
    
    # הצגת טבלה בסיסית
    st.subheader("👀 הצצה למאגר הנתונים (5 שורות ראשונות)")
    st.dataframe(df.head(), use_container_width=True)
    
    # מדדים מרכזיים בכרטיסיות
    st.subheader("📈 מדדים כלליים")
    col1, col2, col3 = st.columns(3)
    col1.metric("מחיר ממוצע", f"₪{df['מחיר'].mean():,.0f}")
    col2.metric("שטח דירה ממוצע", f"{df['שטח_דירה'].mean():.1f} מ״ר")
    col3.metric("מרחק ממוצע מהים", f"{df['מרחק_מהים'].mean():.2f} ק״מ")
    
    st.markdown("---")
    
    # גרפים באמצעות Streamlit בלבד
    st.subheader("📉 גרפים ומגמות בדאטה")
    
    # תיקון גרף 1 (שימוש בגרש בודד בחוץ)
    st.markdown('**הקשר בין שטח הדירה (מ"ר) למחיר השוק (₪):**')
    st.scatter_chart(data=df, x='שטח_דירה', y='מחיר', color='שכונה', use_container_width=True)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**מחיר ממוצע לפי שכונה (₪):**")
        neighborhood_prices = df.groupby('שכונה')['מחיר'].mean()
        st.bar_chart(neighborhood_prices, use_container_width=True)
        
    with col_g2:
        # תיקון גרף 3 (שימוש בגרש בודד בחוץ)
        st.markdown('**הקשר בין המרחק מהים (ק"מ) למחיר הדירה (₪):**')
        st.scatter_chart(data=df, x='מרחק_מהים', y='מחיר', use_container_width=True)
