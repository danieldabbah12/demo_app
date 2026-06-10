import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="חיזוי מחירי דירות | תל אביב",
    page_icon="🏙️",
    layout="wide",
)

# ── Tel Aviv color palette ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #f0f6ff; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003d99 0%, #0055cc 60%, #0077ff 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }

    /* Main header */
    h1 { color: #003d99; font-family: 'Segoe UI', sans-serif; }
    h2, h3 { color: #0055cc; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #cce0ff;
        border-left: 5px solid #0077ff;
        border-radius: 10px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #003d99 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #0055cc !important; }

    /* Info / success boxes */
    .stAlert { border-radius: 10px; }

    /* Buttons */
    .stButton > button {
        background-color: #0055cc;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #003d99; color: white; }

    /* Divider */
    hr { border-color: #cce0ff; }

    /* RTL text helpers */
    .rtl { direction: rtl; text-align: right; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #cce0ff;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 300

    neighborhoods = {
        "צפון תל אביב": {"base": 6_500_000, "dist_sea_range": (2, 8)},
        "רמת אביב":      {"base": 5_800_000, "dist_sea_range": (4, 10)},
        "הצפון הישן":    {"base": 5_200_000, "dist_sea_range": (1, 5)},
        "לב תל אביב":    {"base": 4_800_000, "dist_sea_range": (1, 4)},
        "פלורנטין":      {"base": 3_900_000, "dist_sea_range": (2, 6)},
        "נווה צדק":      {"base": 5_500_000, "dist_sea_range": (0.5, 3)},
        "יפו":           {"base": 3_400_000, "dist_sea_range": (0.5, 4)},
        "הדר יוסף":      {"base": 4_100_000, "dist_sea_range": (5, 12)},
    }

    rows = []
    neigh_list = list(neighborhoods.keys())
    weights = [0.15, 0.12, 0.14, 0.16, 0.10, 0.13, 0.10, 0.10]

    for _ in range(n):
        neigh = np.random.choice(neigh_list, p=weights)
        info  = neighborhoods[neigh]

        area          = int(np.random.normal(80, 30).clip(30, 220))
        dist_sea      = round(np.random.uniform(*info["dist_sea_range"]), 1)
        year_built    = int(np.random.choice(
            range(1950, 2024),
            p=np.array([1/(2024-1950)] * (2024-1950))
        ))
        num_floors    = int(np.random.randint(2, 25))
        rooms         = int(np.random.choice([2, 2.5, 3, 3.5, 4, 4.5, 5, 6],
                                              p=[0.08,0.12,0.20,0.18,0.18,0.12,0.08,0.04]))
        has_balcony   = int(np.random.choice([0, 1], p=[0.35, 0.65]))

        # Price model
        price = info["base"]
        price += area     * np.random.uniform(28_000, 38_000)
        price -= dist_sea * np.random.uniform(60_000, 100_000)
        price += (year_built - 1960) * np.random.uniform(15_000, 25_000)
        price += rooms    * np.random.uniform(80_000, 150_000)
        if has_balcony:
            price += np.random.uniform(100_000, 250_000)
        price += np.random.normal(0, 200_000)          # noise
        price = max(price, 800_000)

        rows.append({
            "שכונה":              neigh,
            "שטח (מ\"ר)":          area,
            "מרחק מהים (ק\"מ)":    dist_sea,
            "שנת בנייה":           year_built,
            "מספר קומות בבניין":   num_floors,
            "מספר חדרים":          rooms,
            "מרפסת":               has_balcony,
            "מחיר (₪)":            int(round(price, -3)),
        })

    return pd.DataFrame(rows)


# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df2 = df.copy()
    df2["שכונה_enc"] = le.fit_transform(df2["שכונה"])

    feature_cols = [
        "שטח (מ\"ר)", "מרחק מהים (ק\"מ)", "שנת בנייה",
        "מספר קומות בבניין", "מספר חדרים", "מרפסת", "שכונה_enc"
    ]
    X = df2[feature_cols]
    y = df2["מחיר (₪)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    return model, le, feature_cols, mae, r2, X_test, y_test, y_pred


# ── Load data & model ─────────────────────────────────────────────────────────
df    = generate_data()
model, le, feature_cols, mae, r2, X_test, y_test, y_pred = train_model(df)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏙️ תל אביב")
st.sidebar.markdown("### ניווט")
page = st.sidebar.radio(
    "",
    ["🏠 דף הבית", "📊 סטטיסטיקות וגרפים", "🔮 חיזוי מחיר"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**על האפליקציה**  
מודל Random Forest  
שאומן על 300 דירות  
מדומות בתל אביב.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 דף הבית":
    st.markdown("# 🏙️ חיזוי מחירי דירות בתל אביב")
    st.markdown("### מודל Machine Learning פדגוגי | לימוד עקרונות בנייה")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class='card rtl'>
        <h3>🎯 מה עושה האפליקציה הזו?</h3>
        <p>האפליקציה ממחישה את <strong>עקרונות בניית מודל ML</strong> לחיזוי מחירי נדל"ן —
        צעד אחר צעד, מהדאטה ועד לתחזית.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card rtl'>
        <h3>📋 שלבי הבנייה שמוצגים כאן</h3>
        <ol>
          <li><strong>יצירת נתונים</strong> — 300 דירות עם מאפיינים ריאליסטיים</li>
          <li><strong>EDA</strong> — ניתוח חקרני של הנתונים</li>
          <li><strong>Feature Engineering</strong> — בחירת ועיבוד משתנים</li>
          <li><strong>אימון מודל</strong> — Random Forest עם 200 עצים</li>
          <li><strong>הערכת ביצועים</strong> — MAE, R²</li>
          <li><strong>חיזוי</strong> — הזן דירה וקבל תמחור</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📈 ביצועי המודל")
        st.metric("R² Score", f"{r2:.3f}", help="כמה מהשונות המודל מסביר (1 = מושלם)")
        st.metric("MAE", f"₪{mae:,.0f}", help="שגיאה ממוצעת בחיזוי המחיר")
        st.metric("גודל הדאטה", "300 דירות")
        st.metric("מספר features", "7 משתנים")

        st.markdown("### 🏘️ שכונות בדאטה")
        neigh_counts = df["שכונה"].value_counts()
        for n, c in neigh_counts.items():
            pct = int(c / len(df) * 100)
            st.progress(pct / 100, text=f"{n}: {c} דירות")

    st.markdown("---")
    st.info("💡 נווט בין הדפים בסרגל השמאלי כדי לחקור את הדאטה ולחזות מחירים.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – STATISTICS & CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 סטטיסטיקות וגרפים":
    st.markdown("# 📊 ניתוח נתונים – EDA")
    st.markdown("הנתונים נוצרו באופן סינתטי כך שישקפו את שוק הנדל\"ן בתל אביב.")
    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("מחיר ממוצע",   f"₪{df['מחיר (₪)'].mean():,.0f}")
    c2.metric("מחיר חציוני",  f"₪{df['מחיר (₪)'].median():,.0f}")
    c3.metric("שטח ממוצע",    f"{df['שטח (מ\"ר)'].mean():.0f} מ\"ר")
    c4.metric("% דירות עם מרפסת", f"{df['מרפסת'].mean()*100:.0f}%")

    st.markdown("---")

    # ── Chart 1: avg price by neighborhood ───────────────────────────────────
    st.subheader("💰 מחיר ממוצע לפי שכונה")
    avg_price = (
        df.groupby("שכונה")["מחיר (₪)"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    avg_price.columns = ["שכונה", "מחיר_ממוצע"]
    avg_price["מחיר_ממוצע_מיליון"] = avg_price["מחיר_ממוצע"] / 1_000_000
    st.bar_chart(avg_price.set_index("שכונה")["מחיר_ממוצע_מיליון"])
    st.caption("מיליוני ₪ | שכונות צפוניות מובילות במחיר")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # ── Chart 2: price vs area scatter ───────────────────────────────────
        st.subheader("📐 מחיר לעומת שטח")
        scatter_df = df[["שטח (מ\"ר)", "מחיר (₪)"]].copy()
        scatter_df["מחיר_מיליון"] = scatter_df["מחיר (₪)"] / 1_000_000
        st.scatter_chart(
            scatter_df,
            x="שטח (מ\"ר)",
            y="מחיר_מיליון",
            color="#0055cc",
        )
        st.caption("קורלציה חיובית ברורה: יותר מ\"ר = מחיר גבוה יותר")

    with col2:
        # ── Chart 3: price vs distance from sea ──────────────────────────────
        st.subheader("🌊 מחיר לעומת מרחק מהים")
        sea_df = df[["מרחק מהים (ק\"מ)", "מחיר (₪)"]].copy()
        sea_df["מחיר_מיליון"] = sea_df["מחיר (₪)"] / 1_000_000
        st.scatter_chart(
            sea_df,
            x="מרחק מהים (ק\"מ)",
            y="מחיר_מיליון",
            color="#ff6600",
        )
        st.caption("ככל שרחוק יותר מהים — המחיר נמוך יותר")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        # ── Chart 4: rooms distribution ──────────────────────────────────────
        st.subheader("🛏️ התפלגות מספר חדרים")
        rooms_dist = df["מספר חדרים"].value_counts().sort_index()
        st.bar_chart(rooms_dist)
        st.caption("3–4 חדרים הם הנפוצים ביותר בדאטה")

    with col4:
        # ── Chart 5: avg price by year built (decade) ─────────────────────────
        st.subheader("🏗️ מחיר ממוצע לפי עשור בנייה")
        df_decade = df.copy()
        df_decade["עשור"] = (df_decade["שנת בנייה"] // 10 * 10).astype(str) + "s"
        decade_price = (
            df_decade.groupby("עשור")["מחיר (₪)"]
                     .mean()
                     .sort_index()
                     / 1_000_000
        )
        st.line_chart(decade_price)
        st.caption("בנייה חדשה מביאה מחירים גבוהים יותר")

    st.markdown("---")

    # ── Chart 6: feature importance ───────────────────────────────────────────
    st.subheader("🧠 חשיבות משתנים במודל (Feature Importance)")
    feature_names_heb = {
        "שטח (מ\"ר)":          "שטח (מ\"ר)",
        "מרחק מהים (ק\"מ)":    "מרחק מהים",
        "שנת בנייה":           "שנת בנייה",
        "מספר קומות בבניין":   "קומות בבניין",
        "מספר חדרים":          "מספר חדרים",
        "מרפסת":               "מרפסת",
        "שכונה_enc":           "שכונה",
    }
    importances = pd.Series(
        model.feature_importances_,
        index=[feature_names_heb.get(c, c) for c in feature_cols]
    ).sort_values(ascending=False)
    st.bar_chart(importances)
    st.caption("השטח והשכונה הם המשתנים החשובים ביותר למודל")

    st.markdown("---")

    # ── Model performance: actual vs predicted ────────────────────────────────
    st.subheader("✅ ביצועי המודל: חיזוי מול מציאות")
    perf_df = pd.DataFrame({
        "מחיר אמיתי (מיליון ₪)":  np.array(y_test) / 1_000_000,
        "מחיר חזוי (מיליון ₪)":   y_pred / 1_000_000,
    }).reset_index(drop=True)
    st.scatter_chart(
        perf_df,
        x="מחיר אמיתי (מיליון ₪)",
        y="מחיר חזוי (מיליון ₪)",
        color="#00aa44",
    )
    st.caption(f"R² = {r2:.3f} | MAE = ₪{mae:,.0f} — קו אלכסוני = חיזוי מושלם")

    # Raw data toggle
    st.markdown("---")
    if st.checkbox("📋 הצג את הדאטה הגולמי"):
        st.dataframe(df, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 חיזוי מחיר":
    st.markdown("# 🔮 חיזוי מחיר דירה")
    st.markdown("הזן את פרטי הדירה — המודל יחזה את המחיר הצפוי.")
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏘️ פרטי מיקום")
            neighborhood = st.selectbox(
                "שכונה",
                options=sorted(df["שכונה"].unique()),
            )
            dist_sea = st.slider(
                "מרחק מהים (ק\"מ)",
                min_value=0.5, max_value=15.0, value=3.0, step=0.5
            )

            st.markdown("#### 🏗️ פרטי הבניין")
            year_built = st.slider(
                "שנת בנייה",
                min_value=1950, max_value=2024, value=2005, step=1
            )
            num_floors = st.slider(
                "מספר קומות בבניין",
                min_value=2, max_value=30, value=10, step=1
            )

        with col2:
            st.markdown("#### 🛋️ פרטי הדירה")
            area = st.slider(
                "שטח הדירה (מ\"ר)",
                min_value=30, max_value=250, value=80, step=5
            )
            rooms = st.select_slider(
                "מספר חדרים",
                options=[2, 2.5, 3, 3.5, 4, 4.5, 5, 6],
                value=3.0
            )
            has_balcony = st.radio(
                "מרפסת?",
                options=["כן ✅", "לא ❌"],
                horizontal=True
            )
            balcony_val = 1 if has_balcony == "כן ✅" else 0

        submitted = st.form_submit_button("🔮 חשב מחיר", use_container_width=True)

    if submitted:
        neigh_enc = le.transform([neighborhood])[0]
        X_input = np.array([[area, dist_sea, year_built, num_floors, rooms, balcony_val, neigh_enc]])
        predicted_price = model.predict(X_input)[0]

        st.markdown("---")
        st.markdown("### 🏆 תוצאת החיזוי")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("💰 מחיר חזוי", f"₪{predicted_price:,.0f}")
        res_col2.metric("💵 מחיר למ\"ר", f"₪{predicted_price/area:,.0f}")
        res_col3.metric("📍 שכונה", neighborhood)

        # Comparison to neighborhood average
        neigh_avg = df[df["שכונה"] == neighborhood]["מחיר (₪)"].mean()
        diff_pct  = (predicted_price - neigh_avg) / neigh_avg * 100
        sign      = "▲" if diff_pct >= 0 else "▼"
        color     = "🟢" if diff_pct >= 0 else "🔴"

        st.info(
            f"{color} הדירה שהזנת **{sign}{abs(diff_pct):.1f}%** "
            f"{'מעל' if diff_pct >= 0 else 'מתחת'} לממוצע שכונת {neighborhood} "
            f"(ממוצע: ₪{neigh_avg:,.0f})"
        )

        # Feature summary
        st.markdown("#### 📋 סיכום הנתונים שהוזנו")
        summary = pd.DataFrame({
            "משתנה": ["שכונה", "שטח", "מרחק מהים", "שנת בנייה", "קומות בבניין", "חדרים", "מרפסת"],
            "ערך":   [neighborhood, f"{area} מ\"ר", f"{dist_sea} ק\"מ",
                      str(year_built), str(num_floors), str(rooms),
                      "כן" if balcony_val else "לא"]
        })
        st.table(summary)

        st.success("✅ המחיר חושב באמצעות מודל Random Forest שאומן על 300 דירות בתל אביב.")
