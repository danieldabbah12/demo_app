import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- הגדרות עמוד ועיצוב (כחול ולבן - תל אביב) ---
st.set_page_config(page_title="חיזוי מחירי דירות - תל אביב", layout="wide", page_icon="🏙️")

st.markdown("""
    <style>
    /* עיצוב רקע האפליקציה ללבן וטקסט כחול/שחור קריא */
    .stApp {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #005EB8 !important; /* כחול תל אביבי */
    }
    .stButton>button {
        background-color: #005EB8;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00468C;
        color: white;
    }
    div[data-baseweb="select"] > div {
        border-color: #005EB8;
    }
    </style>
""", unsafe_allow_html=True)

# --- פונקציה ליצירת נתונים סינתטיים (נטענת פעם אחת) ---
@st.cache_data
def generate_data():
    np.random.seed(42) # לשמירה על עקביות הנתונים בכל הרצה
    n = 300
    
    neighborhoods = ["פלורנטין", "הצפון הישן", "רמת אביב", "יפו", "מרכז העיר"]
    
    # הגרלת מאפייני הדירות
    area = np.random.randint(40, 150, n)
    distance_to_sea = np.random.randint(100, 3500, n)
    year_built = np.random.randint(1950, 2024, n)
    total_apts = np.random.randint(4, 40, n)
    rooms = np.round(area / 25).astype(int) # קשר הגיוני בין שטח למספר חדרים
    rooms = np.clip(rooms, 2, 6)
    balcony = np.random.choice([True, False], n, p=[0.7, 0.3])
    neighborhood = np.random.choice(neighborhoods, n)
    
    # יצירת מחירון בסיס (בשקלים) עם לוגיקה מסוימת + רעש סטטיסטי כדי שייראה אמיתי
    base_price = 1_500_000
    price = base_price + (area * 35_000) - (distance_to_sea * 150) + ((year_built - 1950) * 8_000) + (balcony * 200_000)
    
    # תוספת מחיר לפי שכונה
    neighborhood_premium = {
        "הצפון הישן": 1_200_000,
        "רמת אביב": 800_000,
        "מרכז העיר": 600_000,
        "פלורנטין": 200_000,
        "יפו": 50_000
    }
    
    for i in range(n):
        price[i] += neighborhood_premium[neighborhood[i]]
        # הוספת רעש רנדומלי למחיר (כדי שהמודל לא יהיה מושלם ב-100%)
        price[i] += np.random.randint(-150_000, 150_000)
        
    df = pd.DataFrame({
        'שטח (מ"ר)': area,
        'מרחק מהים (מטרים)': distance_to_sea,
        'שנת בנייה': year_built,
        'דירות בבניין': total_apts,
        'מספר חדרים': rooms,
        'יש מרפסת': balcony,
        'שכונה': neighborhood,
        'מחיר (ש"ח)': price
    })
    
    return df

# טעינת הנתונים
df = generate_data()

# --- פונקציה לאימון המודל ---
@st.cache_resource
def train_model(data):
    # המרת משתנים קטגוריאליים (שכונה) למספרים כדי שהמודל יוכל ללמוד
    df_encoded = pd.get_dummies(data, columns=['שכונה'], drop_first=True)
    X = df_encoded.drop('מחיר (ש"ח)', axis=1)
    y = df_encoded['מחיר (ש"ח)']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns

model, model_features = train_model(df)

# --- תפריט ניווט צדדי ---
st.sidebar.title("ניווט")
page = st.sidebar.radio("בחר עמוד:", ["🏠 דף הבית", "📊 חקר נתונים (EDA)", "🤖 מודל חיזוי מחירים"])

# --- עמוד 1: דף הבית ---
if page == "🏠 דף הבית":
    st.title("ברוכים הבאים למערכת חיזוי מחירי הדירות של תל אביב 🏙️")
    st.markdown("""
    אפליקציה זו מדגימה את העקרונות הבסיסיים של בניית מודל **למידת מכונה (Machine Learning)**. 
    
    **מה יש באפליקציה?**
    * **נתונים:** מאגר מידע סינתטי (שנוצר במיוחד לצורך ההדגמה) המכיל 300 עסקאות נדל"ן בתל אביב.
    * **חקר נתונים (EDA):** עמוד המציג סטטיסטיקות וקשרים מעניינים בין מאפייני הדירה למחיר שלה, באמצעות גרפים. הויזואליזציה עוזרת לנו "להבין" את הנתונים לפני שמפעילים עליהם מתמטיקה.
    * **מודל חיזוי:** אלגוריתם רגרסיה ליניארית
