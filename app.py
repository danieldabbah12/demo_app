import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==================================================
# הגדרות עמוד
# ==================================================

st.set_page_config(
    page_title="נדל״ן תל אביב AI",
    page_icon="🏙️",
    layout="wide"
)

# ==================================================
# עיצוב
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f9ff;
}

h1, h2, h3 {
    color: #0057B7;
}

section[data-testid="stSidebar"] {
    background-color: #eaf4ff;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #dbeafe;
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
}

.stButton>button {
    background-color: #0057B7;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# יצירת דאטה סינתטי
# ==================================================

@st.cache_data
def create_data():

    np.random.seed(42)

    neighborhoods = {
        "לב תל אביב": 1.35,
        "רמת אביב": 1.45,
        "פלורנטין": 1.10,
        "יפו": 1.00,
        "הצפון הישן": 1.55,
        "הצפון החדש": 1.40,
        "נאות אפקה": 1.20,
        "קריית שלום": 0.90,
    }

    data = []

    for _ in range(300):

        neighborhood = np.random.choice(list(neighborhoods.keys()))
        factor = neighborhoods[neighborhood]

        size = np.random.randint(45, 180)

        distance_from_sea = np.random.randint(100, 5000)

        year = np.random.randint(1960, 2025)

        apartments_in_building = np.random.randint(4, 80)

        rooms = np.random.randint(2, 7)

        balcony = np.random.choice(
            ["כן", "לא"],
            p=[0.7, 0.3]
        )

        # =====================================
        # נוסחת מחיר ריאליסטית יותר
        # =====================================

        price = (
            size * 25000
            + (5000 - distance_from_sea) * 180
            + (year - 1960) * 7000
            + rooms * 120000
            + (250000 if balcony == "כן" else 0)
        )

        price = price * factor

        # רעש אקראי
        price += np.random.normal(0, 180000)

        # גבולות ריאליסטיים
        price = max(price, 1800000)
        price = min(price, 8500000)

        data.append({
            "שטח הדירה": int(size),
            "מרחק מהים": int(distance_from_sea),
            "שנת בנייה": int(year),
            "מספר דירות בבניין": int(apartments_in_building),
            "מספר חדרים": int(rooms),
            "מרפסת": balcony,
            "שכונה": neighborhood,
            "מחיר הדירה": int(price)
        })

    return pd.DataFrame(data)

df = create_data()

# ==================================================
# הכנת מודל
# ==================================================

model_df = df.copy()

model_df["מרפסת"] = model_df["מרפסת"].map({
    "כן": 1,
    "לא": 0
})

model_df = pd.get_dummies(
    model_df,
    columns=["שכונה"]
)

X = model_df.drop("מחיר הדירה", axis=1)
y = model_df["מחיר הדירה"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=120,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🏙️ תל אביב נדל״ן AI")

page = st.sidebar.radio(
    "ניווט",
    [
        "דף הבית",
        "אנליטיקות",
        "חיזוי מחיר",
        "איך המודל עובד?"
    ]
)

st.sidebar.info(
    "האפליקציה נבנתה לצורכי לימוד והדגמת Machine Learning."
)

# ==================================================
# דף הבית
# ==================================================

if page == "דף הבית":

    st.title("🏙️ חיזוי מחירי דירות בתל אביב")

    st.markdown("""
    ### ברוכים הבאים למערכת נדל״ן AI

    האפליקציה מדגימה כיצד ניתן:

    - ליצור דאטה סינתטי
    - לנתח נתונים
    - להציג גרפים
    - לאמן מודל Machine Learning
    - לחזות מחירי דירות בזמן אמת

    המערכת מבוססת על מודל Random Forest.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "מספר דירות",
        len(df)
    )

    col2.metric(
        "מחיר ממוצע",
        f"{int(df['מחיר הדירה'].mean()):,} ₪"
    )

    col3.metric(
        "דיוק המודל (R²)",
        f"{r2:.2f}"
    )

    st.divider()

    st.subheader("📌 תובנות מרכזיות")

    st.success(
        "דירות קרובות לים יקרות משמעותית."
    )

    st.info(
        "שכונות הצפון הישן ורמת אביב הן היקרות ביותר."
    )

    st.warning(
        "מרפסת מעלה את מחיר הדירה באופן משמעותי."
    )

    st.divider()

    st.subheader("🏘️ דוגמה מהדאטה")

    st.dataframe(df.sample(10))

# ==================================================
# אנליטיקות
# ==================================================

elif page == "אנליטיקות":

    st.title("📊 אנליטיקות וגרפים")

    tab1, tab2, tab3, tab4 = st.tabs([
        "מחירים",
        "שכונות",
        "פיצ'רים",
        "התפלגויות"
    ])

    # ==================================================
    # TAB 1
    # ==================================================

    with tab1:

        st.subheader("מחיר מול שטח דירה")

        st.scatter_chart(
            df,
            x="שטח הדירה",
            y="מחיר הדירה"
        )

        st.subheader("מחיר לפי מספר חדרים")

        room_prices = (
            df.groupby("מספר חדרים")["מחיר הדירה"]
            .mean()
        )

        st.line_chart(room_prices)

    # ==================================================
    # TAB 2
    # ==================================================

    with tab2:

        st.subheader("מחיר ממוצע לפי שכונה")

        avg_prices = (
            df.groupby("שכונה")["מחיר הדירה"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(avg_prices)

        st.subheader("כמה דירות יש בכל שכונה")

        neighborhood_counts = (
            df["שכונה"]
            .value_counts()
        )

        st.bar_chart(neighborhood_counts)

    # ==================================================
    # TAB 3
    # ==================================================

    with tab3:

        st.subheader("Feature Importance")

        importance_df = pd.DataFrame({
            "פיצ'ר": X.columns,
            "חשיבות": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="חשיבות",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index("פיצ'ר")
        )

        st.dataframe(importance_df)

    # ==================================================
    # TAB 4
    # ==================================================

    with tab4:

        st.subheader("התפלגות מחירי הדירות")

        hist_data = pd.DataFrame({
            "מחיר": df["מחיר הדירה"]
        })

        st.bar_chart(
            hist_data["מחיר"].value_counts(
                bins=20
            )
        )

        st.subheader("מרפסות")

        balcony_counts = (
            df["מרפסת"]
            .value_counts()
        )

        st.bar_chart(balcony_counts)

# ==================================================
# חיזוי
# ==================================================

elif page == "חיזוי מחיר":

    st.title("🤖 חיזוי מחיר דירה")

    st.write(
        "הזן נתוני דירה והמודל יחזה את המחיר."
    )

    col1, col2 = st.columns(2)

    with col1:

        size = st.slider(
            "שטח הדירה",
            45,
            200,
            90
        )

        distance = st.slider(
            "מרחק מהים",
            100,
            5000,
            1500
        )

        year = st.slider(
            "שנת בנייה",
            1960,
            2025,
            2005
        )

        rooms = st.slider(
            "מספר חדרים",
            2,
            7,
            4
        )

    with col2:

        apartments = st.slider(
            "מספר דירות בבניין",
            4,
            100,
            20
        )

        balcony = st.selectbox(
            "מרפסת",
            ["כן", "לא"]
        )

        neighborhood = st.selectbox(
            "שכונה",
            sorted(df["שכונה"].unique())
        )

    input_data = {
        "שטח הדירה": size,
        "מרחק מהים": distance,
        "שנת בנייה": year,
        "מספר דירות בבניין": apartments,
        "מספר חדרים": rooms,
        "מרפסת": 1 if balcony == "כן" else 0,
    }

    for col in X.columns:
        if col.startswith("שכונה_"):
            input_data[col] = 0

    neighborhood_col = f"שכונה_{neighborhood}"

    if neighborhood_col in input_data:
        input_data[neighborhood_col] = 1

    input_df = pd.DataFrame([input_data])

    if st.button("חזוי מחיר"):

        prediction = int(
            model.predict(input_df)[0]
        )

        st.success(
            f"💰 מחיר הדירה המשוער: {prediction:,} ₪"
        )

        if prediction > 6500000:

            st.info(
                "זו דירה יוקרתית במיוחד 🏆"
            )

        elif prediction > 4000000:

            st.info(
                "זו דירה ברמת מחיר גבוהה."
            )

        else:

            st.info(
                "זו דירה ברמת מחיר ממוצעת."
            )

# ==================================================
# איך המודל עובד
# ==================================================

elif page == "איך המודל עובד?":

    st.title("🧠 איך המודל עובד?")

    st.markdown("""
    ## שלב 1 — איסוף דאטה

    יצרנו דאטה של 300 דירות בתל אביב.

    לכל דירה יש:
    - שטח
    - מרחק מהים
    - מספר חדרים
    - שכונה
    - שנת בנייה
    - מרפסת

    ---

    ## שלב 2 — אימון מודל

    המודל לומד קשרים בין מאפייני הדירה למחיר שלה.

    השתמשנו במודל:
    ### Random Forest

    זהו מודל מבוסס עצי החלטה.

    ---

    ## שלב 3 — חיזוי

    המשתמש מזין נתונים חדשים,
    והמודל חוזה מחיר משוער.

    ---

    ## מדדי הצלחה

    ### MAE
    שגיאה ממוצעת:
    """)

    st.metric(
        "MAE",
        f"{int(mae):,} ₪"
    )

    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.success(
        "ככל ש-R² קרוב יותר ל-1, המודל טוב יותר."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "נבנה באמצעות Streamlit + Scikit-learn | הדאטה סינתטי לצורכי לימוד"
)
