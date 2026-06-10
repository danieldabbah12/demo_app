import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =========================
# הגדרות עמוד ועיצוב
# =========================

st.set_page_config(
    page_title="חיזוי מחירי דירות בתל אביב",
    page_icon="🏙️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fbff;
    }

    h1, h2, h3 {
        color: #0057b8;
    }

    .stButton>button {
        background-color: #0057b8;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }

    section[data-testid="stSidebar"] {
        background-color: #eef6ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# יצירת דאטה סינתטי
# =========================

@st.cache_data
def create_data():

    np.random.seed(42)

    neighborhoods = [
        ("לב תל אביב", 1.35),
        ("פלורנטין", 1.05),
        ("נאות אפקה", 1.15),
        ("רמת אביב", 1.45),
        ("יפו", 1.00),
        ("הצפון הישן", 1.50),
        ("הצפון החדש", 1.40),
        ("קריית שלום", 0.90),
    ]

    rows = []

    for _ in range(300):

        neighborhood, factor = neighborhoods[
            np.random.randint(len(neighborhoods))
        ]

        size = np.random.randint(40, 180)
        distance = np.random.randint(100, 5000)
        year = np.random.randint(1960, 2025)
        apartments = np.random.randint(4, 80)
        rooms = np.random.randint(2, 7)

        balcony = np.random.choice(
            [0, 1],
            p=[0.35, 0.65]
        )

        base_price = (
            size * 42000
            + (5000 - distance) * 250
            + (year - 1960) * 12000
            + rooms * 180000
            + balcony * 250000
        )

        noise = np.random.normal(0, 250000)

        price = int(base_price * factor + noise)

        price = max(price, 1200000)

        rows.append(
            {
                "שטח הדירה": size,
                "מרחק מהים": distance,
                "שנת בנייה": year,
                "מספר דירות בבניין": apartments,
                "מספר חדרים": rooms,
                "מרפסת": "כן" if balcony else "לא",
                "שכונה": neighborhood,
                "מחיר הדירה": price,
            }
        )

    return pd.DataFrame(rows)


df = create_data()

# =========================
# הכנת מודל
# =========================

model_df = df.copy()

model_df["מרפסת"] = model_df["מרפסת"].map(
    {
        "כן": 1,
        "לא": 0
    }
)

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
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# ניווט
# =========================

page = st.sidebar.radio(
    "ניווט",
    [
        "דף הבית",
        "סטטיסטיקות וגרפים",
        "חיזוי מחיר דירה"
    ]
)

# =========================
# דף הבית
# =========================

if page == "דף הבית":

    st.title("🏙️ חיזוי מחירי דירות בתל אביב")

    st.write(
        """
        ברוכים הבאים לאפליקציית הדגמה לבניית מודל Machine Learning בסיסי באמצעות Streamlit.

        האפליקציה מדגימה:
        - יצירת דאטה
        - ניתוח נתונים
        - ויזואליזציה
        - אימון מודל
        - חיזוי בזמן אמת

        כל הדאטה נוצר בצורה סינתטית לצורכי לימוד.
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1544979590-37e9b47eb705?q=80&w=1200&auto=format&fit=crop",
        use_container_width=True
    )

    st.subheader("📌 מה משפיע על מחיר הדירה?")

    st.markdown(
        """
        - שטח הדירה
        - קרבה לים
        - מספר חדרים
        - שכונה
        - שנת בנייה
        - מרפסת
        """
    )

# =========================
# סטטיסטיקות וגרפים
# =========================

elif page == "סטטיסטיקות וגרפים":

    st.title("📊 סטטיסטיקות וגרפים")

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
        "שטח ממוצע",
        f"{int(df['שטח הדירה'].mean())} מ״ר"
    )

    st.divider()

    st.subheader("מחיר ממוצע לפי שכונה")

    avg_prices = (
        df.groupby("שכונה")["מחיר הדירה"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(avg_prices)

    st.divider()

    st.subheader("השפעת שטח הדירה על המחיר")

    st.scatter_chart(
        df,
        x="שטח הדירה",
        y="מחיר הדירה"
    )

    st.divider()

    st.subheader("מחיר ממוצע לפי מספר חדרים")

    room_prices = (
        df.groupby("מספר חדרים")["מחיר הדירה"]
        .mean()
    )

    st.line_chart(room_prices)

    st.divider()

    st.subheader("כמה דירות כוללות מרפסת?")

    balcony_counts = (
        df["מרפסת"]
        .value_counts()
    )

    st.bar_chart(balcony_counts)

    st.divider()

    st.subheader("הצצה לדאטה")

    st.dataframe(df.head(20))

# =========================
# חיזוי מחיר
# =========================

elif page == "חיזוי מחיר דירה":

    st.title("🤖 חיזוי מחיר דירה")

    st.write(
        "הזן נתונים של דירה וקבל תחזית מחיר מהמודל."
    )

    size = st.slider(
        "שטח הדירה",
        40,
        200,
        90
    )

    distance = st.slider(
        "מרחק מהים (במטרים)",
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

    apartments = st.slider(
        "מספר דירות בבניין",
        4,
        100,
        20
    )

    rooms = st.slider(
        "מספר חדרים",
        2,
        7,
        3
    )

    balcony = st.selectbox(
        "האם יש מרפסת?",
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
            f"💰 מחיר הדירה המשוער הוא: {prediction:,} ₪"
        )

        st.info(
            "המודל מבוסס על דאטה סינתטי לצורכי לימוד."
        )
