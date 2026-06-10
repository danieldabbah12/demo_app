import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. הגדרות עמוד ועיצוב תל-אביבי (כחול ולבן)
st.set_page_config(page_title="נדל״ן תל אביב - חיזוי מחירי דירות", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #005A9C; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button {
        background-color: #005A9C;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #004477;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #F0F4F8;
    }
    </style>
""", unsafe_allow_html=True)

# 2. פונקציה לייצור הדאטה (300 שורות) - פדגוגי ומבוסס חוקיות הגיונית
@st.cache_data
def generate_tel_aviv_data():
    np.random.seed(42)
    n_samples = 300
    
    # יצירת המשתנים עצמם
    שטח_דירה = np.random.normal(80, 25, n_samples).clip(30, 200).astype(int)
    מרחק_מהים = np.random.uniform(0.1, 3.0, n_samples).round(2) # בקילומטרים
    שנת_בנייה = np.random.randint(1950, 2026, n_samples)
    מספר_דירות = np.random.randint(6, 60, n_samples)
    כמה_חדרים = np.round(שטח_דירה / 25 + np.random.normal(0, 0.5, n_samples)).clip(1, 6).astype(int)
    האם_יש_מרפסת = np.random.choice(['כן', 'לא'], n_samples, p=[0.5, 0.5])
    שכונות = ['לב העיר', 'צפון ישן', 'פלורנטין', 'יפו', 'בבלי']
    שכונה = np.random.choice(שכונות, n_samples)
    
    # חישוב מחיר דירה מבוסס נוסחה עם רעש סטטיסטי (בשביל המודל)
    מחיר_בסיס = 2_000_000
    מחיר = (
        מחיר_בסיס + 
        (שטח_דירה * 45_000) - 
        (מרחק_מהים * 600_000) + 
        ((שנת_בנייה - 1950) * 12_000) + 
        (כמה_חדרים * 150_000)
    )
    
    # התאמות לפי שכונה ומרפסת
    for i in range(n_samples):
        if האם_יש_מרפסת[i] == 'כן': מחיר[i] += 350_000
        if שכונה[i] == 'לב העיר': מחיר[i] += 800_000
        elif שכונה[i] == 'צפון ישן': מחיר[i] += 600_000
        elif שכונה[i] == 'בבלי': מחיר[i] += 400_000
        elif שכונה[i] == 'פלורנטין': מחיר[i] -= 300_000
        elif שכונה[i] == 'יפו': מחיר[i] -= 200_000
        
    # הוספת רעש אקראי נורמלי
    מחיר += np.random.normal(0, 150_000, n_samples)
    מחיר = np.round(מחיר / 10_000) * 10_000 # עיגול למספרים יפים
    
    df = pd.DataFrame({
        'שטח_דירה': שטח_דירה,
        'מרחק_מהים': מרחק_מהים,
        'שנת_בנייה': שנת_בנייה,
        'מספר_דירות_בבנין': מספר_דירות,
        'כמה_חדרים': כמה_חדרים,
        'האם_יש_מרפסת': האם_יש_מרפסת,
        'שכונה': שכונה,
        'מחיר': מחיר
    })
    return df

df = generate_tel_aviv_data()

# 3. תפריט ניווט צידי
st.sidebar.title("🏙️ נדל״ן תל אביב")
page = st.sidebar.radio("ניווט בין עמודים:", ["עמוד כניסה", "ניתוח נתונים וסטטיסטיקה", "חיזוי מחיר דירה"])

# --- עמוד 1: עמוד כניסה ---
if page == "עמוד כניסה":
    st.title("🏙️ ברוכים הבאים לאפליקציית חיזוי מחירי הדירות בתל אביב")
    st.subheader("כלי פדגוגי להבנת עקרונות המודל הליניארי ב-Machine Learning")
    
    st.markdown("""
    אפליקציה זו נבנתה במטרה להדגים כיצד ניתן לקחת נתוני נדל"ן, לנתח אותם ויזואלית, ולבנות מודל בינה מלאכותית בסיסי (Linear Regression) שיודע לחזות מחירי דירות על סמך מאפיינים שונים.
    
    ### 📊 מה תמצאו באפליקציה?
    * **ניתוח נתונים וסטטיסטיקה:** הצצה למאגר הנתונים הסינתטי שיצרנו (300 דירות בתל אביב) וגרפים מובנים שמראים את הקשרים בין המשתנים.
    * **חיזוי מחיר דירה:** סימולטור אינטראקטיבי שבו תוכלו להזין מאפייני דירה משלכם ולקבל הערכת מחיר מיידית מהמודל.
    
    *הדאטה באפליקציה זו נוצר בצורה אקראית מבוססת חוקיות הגיונית לצרכים לימודיים בלבד.*
    """)
    st.info("💡 השתמשו בתפריט הצידי כדי לעבור בין העמודים השונים.")

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
    
    # גרף 1: קשר בין שטח למחיר (Scatter Chart)
    st.markdown("**הקשר בין שטח הדירה (מ"ר) למחיר השוק (₪):**")
    st.scatter_chart(data=df, x='שטח_דירה', y='מחיר', color='שכונה', use_container_width=True)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # גרף 2: מחיר ממוצע לפי שכונה (Bar Chart)
        st.markdown("**מחיר ממוצע לפי שכונה (₪):**")
        neighborhood_prices = df.groupby('שכונה')['מחיר'].mean()
        st.bar_chart(neighborhood_prices, use_container_width=True)
        
    with col_g2:
        # גרף 3: השפעת המרחק מהים על המחיר
        st.markdown("**הקשר בין המרחק מהים (ק"מ) למחיר הדירה (₪):**")
        st.scatter_chart(data=df, x='מרחק_מהים', y='מחיר', use_container_width=True)

# --- עמוד 3: חיזוי מחיר דירה ---
elif page == "חיזוי מחיר דירה":
    st.title("🔮 סימולטור לחיזוי מחיר דירה")
    st.write("הזינו את מאפייני הדירה שלכם כדי לראות כיצד מודל הרגרסיה הליניארית מתרגם את הנתונים למחיר שוק מוערך.")
    
    # יצירת טפסים לקבלת קלט מהמשתמש
    st.subheader("🛠️ מאפייני הדירה המבוקשת")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        שכונה_נבחרת = st.selectbox("בחר שכונה בתל אביב:", df['שכונה'].unique())
        שטח = st.number_input("שטח הדירה (במ״ר):", min_value=20, max_value=300, value=80, step=5)
        חדרים = st.slider("מספר חדרים:", min_value=1, max_value=6, value=3)
        מרפסת = st.radio("האם יש מרפסת לדירה?", ['כן', 'לא'])
        
    with col_in2:
        מרחק = st.slider("מרחק מהים (בקילומטרים):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        שנה = st.number_input("שנת בנייה:", min_value=1900, max_value=2026, value=2000, step=1)
        דירות_בבנין = st.number_input("מספר דירות כולל בבניין:", min_value=1, max_value=150, value=24, step=1)

    # 4. אימון המודל מאחורי הקלעים בצורה רובסטית (One-Hot Encoding מובנה לפדגוגיה)
    # אנו מייצרים שורת קלט חדשה ומאחדים אותה עם הדאטה המקורי כדי שה-Encoding יעבוד בצורה מושלמת
    user_input_df = pd.DataFrame([{
        'שטח_דירה': שטח,
        'מרחק_מהים': מרחק,
        'שנת_בנייה': שנה,
        'מספר_דירות_בבנין': דירות_בבנין,
        'כמה_חדרים': חדרים,
        'האם_יש_מרפסת': מרפסת,
        'שכונה': שכונה_נבחרת
    }])
    
    # הפרדה של המטרה (X ו-y)
    X_raw = df.drop(columns=['מחיר'])
    y = df['מחיר']
    
    # שרשור של שורת המשתמש כדי לבצע One-Hot Encoding אחיד
    X_combined = pd.concat([X_raw, user_input_df], ignore_index=True)
    X_encoded = pd.get_dummies(X_combined, columns=['שכונה', 'האם_יש_מרפסת'], drop_first=True)
    
    # פיצול מחדש לנתוני אימון ונתון החיזוי של המשתמש
    X_train = X_encoded.iloc[:-1]
    X_user = X_encoded.iloc[[-1]]
    
    # אימון מודל ליניארי מהיר
    model = LinearRegression()
    model.fit(X_train, y)
    
    st.markdown("---")
    
    # כפתור הפעלת החיזוי
    if st.button("🚀 חשב מחיר דירה משוער"):
        prediction = model.predict(X_user)[0]
        
        # הצגת התוצאה בצורה בולטת
        st.success(f"### 🎯 המחיר המוערך על ידי המודל הינו: ₪{prediction:,.0f}")
        
        # הסבר פדגוגי קצר על המודל
        st.markdown("""
        ### 🧠 איך המודל הגיע לתוצאה הזו?
        המודל השתמש ב**רגרסיה ליניארית (Linear Regression)**. הוא למד מתוך 300 הדירות בדאטה את 'המשקל' (Coefficient) של כל תכונה:
        * ככל ש**השטח** גדול יותר ותרם מחיר חיובי.
        * ככל ש**המרחק מהים** קטן, המחיר עלה בהתאמה.
        * השכונה הנבחרת משמשת כקבוע (Bias משתנה) שמעלה או מוריד את מחיר הבסיס של הנכס.
        """)
