import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Tel Aviv Real Estate AI",
    page_icon="🏙️",
    layout="wide"
)

# ==================================================
# DESIGN
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

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 14px;
    padding: 15px;
    border: 1px solid #dbeafe;
}

.stButton>button {
    background-color: #0057B7;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #004799;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# CREATE DATA
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

        neighborhood = np.random.choice(
            list(neighborhoods.keys())
        )

        factor = neighborhoods[neighborhood]

        size = np.random.randint(45, 180)

        distance_from_sea = np.random.randint(
            100,
            5000
        )

        year = np.random.randint(
            1960,
            2025
        )

        apartments_in_building = np.random.randint(
            4,
            80
        )

        rooms = np.random.randint(
            2,
            7
        )

        balcony = np.random.choice(
            ["כן", "לא"],
            p=[0.7, 0.3]
        )

        # ======================================
        # REALISTIC PRICE FORMULA
        # ======================================

        price = (
            size * 25000
            + (5000 - distance_from_sea) * 180
            + (year - 1960) * 7000
            + rooms * 120000
            + (250000 if balcony == "כן" else 0)
        )

        price = price * factor

        # RANDOM NOISE
        price += np.random.normal(
            0,
            180000
        )

        # REALISTIC LIMITS
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
# MODEL
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

X = model_df.drop(
    "מחיר הדירה",
    axis=1
)

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

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🏙️ Tel Aviv Real Estate AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Analytics",
        "Price Prediction",
        "How The Model Works"
    ]
)

# ==================================================
# HOME
# ==================================================

if page == "Home":

    st.title("🏙️ Tel Aviv Real Estate AI")

    st.markdown("""
    ## Smart Real Estate Price Prediction Platform

    This system demonstrates how Machine Learning
    can analyze apartment features and predict
    real estate prices in Tel Aviv.

    The platform includes:
    - Interactive analytics
    - Visual insights
    - Real-time predictions
    - Feature importance analysis
    - Machine Learning metrics
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Apartments",
        len(df)
    )

    col2.metric(
        "Average Price",
        f"{int(df['מחיר הדירה'].mean()):,} ₪"
    )

    col3.metric(
        "Model Accuracy",
        f"{r2:.2f}"
    )

    st.divider()

    st.subheader("📌 Key Insights")

    st.success(
        "Apartments closer to the sea are significantly more expensive."
    )

    st.info(
        "North Tel Aviv neighborhoods dominate the luxury market."
    )

    st.warning(
        "Balconies have a strong impact on apartment prices."
    )

    st.divider()

    st.subheader("🏘️ Sample Data")

    st.dataframe(
        df.sample(10),
        use_container_width=True
    )

# ==================================================
# ANALYTICS
# ==================================================

elif page == "Analytics":

    st.title("📊 Real Estate Analytics")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prices",
        "Neighborhoods",
        "Model Insights",
        "Distributions"
    ])

    # ==================================================
    # TAB 1
    # ==================================================

    with tab1:

        st.subheader("Apartment Size vs Price")

        st.scatter_chart(
            df,
            x="שטח הדירה",
            y="מחיר הדירה"
        )

        st.subheader("Average Price by Number of Rooms")

        room_prices = (
            df.groupby("מספר חדרים")["מחיר הדירה"]
            .mean()
        )

        st.line_chart(room_prices)

    # ==================================================
    # TAB 2
    # ==================================================

    with tab2:

        st.subheader("Average Price by Neighborhood")

        avg_prices = (
            df.groupby("שכונה")["מחיר הדירה"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(avg_prices)

        st.subheader("Number of Apartments by Neighborhood")

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
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        col1.metric(
            "MAE",
            f"{int(mae):,} ₪"
        )

        col2.metric(
            "R² Score",
            f"{r2:.2f}"
        )

    # ==================================================
    # TAB 4
    # ==================================================

    with tab4:

        st.subheader("Price Distribution")

        st.bar_chart(
            df["מחיר הדירה"].value_counts(
                bins=20
            )
        )

        st.subheader("Balcony Distribution")

        balcony_counts = (
            df["מרפסת"]
            .value_counts()
        )

        st.bar_chart(
            balcony_counts
        )

# ==================================================
# PREDICTION
# ==================================================

elif page == "Price Prediction":

    st.title("🤖 Apartment Price Prediction")

    st.write(
        "Enter apartment details and receive a real-time AI prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        size = st.slider(
            "Apartment Size",
            45,
            200,
            90
        )

        distance = st.slider(
            "Distance From Sea",
            100,
            5000,
            1500
        )

        year = st.slider(
            "Year Built",
            1960,
            2025,
            2005
        )

        rooms = st.slider(
            "Rooms",
            2,
            7,
            4
        )

    with col2:

        apartments = st.slider(
            "Apartments In Building",
            4,
            100,
            20
        )

        balcony = st.selectbox(
            "Balcony",
            ["כן", "לא"]
        )

        neighborhood = st.selectbox(
            "Neighborhood",
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

    if st.button("Predict Price"):

        prediction = int(
            model.predict(input_df)[0]
        )

        st.success(
            f"💰 Estimated Price: {prediction:,} ₪"
        )

        if prediction > 6500000:

            st.info(
                "Luxury Apartment 🏆"
            )

        elif prediction > 4000000:

            st.info(
                "High-End Apartment"
            )

        else:

            st.info(
                "Mid-Range Apartment"
            )

# ==================================================
# HOW IT WORKS
# ==================================================

elif page == "How The Model Works":

    st.title("🧠 How The AI Model Works")

    st.markdown("""
    ## Step 1 — Data Collection

    The system analyzes apartment attributes such as:
    - Apartment size
    - Distance from the sea
    - Number of rooms
    - Neighborhood
    - Building age
    - Balcony availability

    ---

    ## Step 2 — Machine Learning

    The AI model learns relationships between
    apartment features and market prices.

    The model used:
    ### Random Forest Regressor

    This algorithm combines multiple decision trees
    to improve prediction accuracy.

    ---

    ## Step 3 — Prediction

    Users enter apartment details,
    and the model generates an estimated market value.
    """)

    st.divider()

    st.subheader("Model Metrics")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{int(mae):,} ₪"
    )

    col2.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.success(
        "Higher R² values indicate better prediction quality."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Powered by Streamlit & Machine Learning"
)
