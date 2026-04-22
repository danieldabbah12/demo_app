import streamlit as st

st.title("האפליקציה שלי")

# כותרת משנה
st.subheader("ברוכים הבאים!")

# פסקת טקסט
st.write("שלום עולם!")

st.write("טקסט רגיל")
st.write("תומך **bold** ו-*italic* בסגנון Markdown")
st.write(42)          # מספרים
st.write([1, 2, 3])   # רשימות
st.write({"a": 1})    # מילונים
