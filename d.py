import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- Page Config --------------------
st.set_page_config(page_title="Gym Dashboard", layout="wide")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Gym.csv")
    return df

df = load_data()

st.title("🏋️‍♂️ Gym Management Dashboard")

# -------------------- Filters --------------------
st.sidebar.header("🔎 Filters")

membership_filter = st.sidebar.selectbox(
    "Membership Type",
    options=["All"] + list(df["membership_type"].unique())
)

gender_filter = st.sidebar.selectbox(
    "Gender",
    options=["All"] + list(df["gender"].unique())
)

trainer_filter = st.sidebar.selectbox(
    "Trainer",
    options=["All"] + list(df["trainer_name"].unique())
)

difficulty_filter = st.sidebar.selectbox(
    "Class Difficulty",
    options=["All"] + list(df["difficulty"].unique())
)

# -------------------- Apply Filters --------------------
filtered_df = df.copy()

if membership_filter != "All":
    filtered_df = filtered_df[filtered_df["membership_type"] == membership_filter]

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

if trainer_filter != "All":
    filtered_df = filtered_df[filtered_df["trainer_name"] == trainer_filter]

if difficulty_filter != "All":
    filtered_df = filtered_df[filtered_df["difficulty"] == difficulty_filter]

# -------------------- Quick Overview --------------------
st.subheader("📊 Quick Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Members", filtered_df["member_id"].nunique())
col2.metric("Total Classes", filtered_df["class_id"].nunique())
col3.metric("Total Trainers", filtered_df["trainer_id"].nunique())
col4.metric("Total Payments", f"${filtered_df['amount'].sum():,.0f}")

# -------------------- Members Analysis --------------------
st.subheader("🧑‍🤝‍🧑 Members Analysis")

col1, col2 = st.columns(2)

with col1:
    gender_fig = px.pie(filtered_df, names="gender", title="Members by Gender")
    st.plotly_chart(gender_fig, use_container_width=True)

with col2:
    membership_fig = px.histogram(
        filtered_df, 
        x="membership_type", 
        color="membership_type",
        title="Membership Type Distribution"
    )
    st.plotly_chart(membership_fig, use_container_width=True)

# -------------------- Attendance Analysis --------------------
st.subheader("Attendance Analysis")

attendance_fig = px.histogram(
    filtered_df, 
    x="class_name", 
    color="difficulty",
    title="Class Attendance by Difficulty"
)
attendance_fig.update_xaxes(categoryorder="total descending")
st.plotly_chart(attendance_fig, use_container_width=True)

# -------------------- Trainers Analysis --------------------
st.subheader("🏋️ Trainers Analysis")

trainer_data = (
    filtered_df.groupby("trainer_name")["attendance_id"]
    .count()
    .reset_index()
    .sort_values(by="attendance_id", ascending=False)
)

trainer_fig = px.bar(
    trainer_data,
    x="trainer_name", 
    y="attendance_id", 
    color="trainer_name",
    title="Top Trainers by Attendance"
)
st.plotly_chart(trainer_fig, use_container_width=True)

# -------------------- Payments Analysis --------------------
st.subheader(" Payments Analysis")

col1, col2 = st.columns(2)

with col1:
    payment_fig = px.pie(filtered_df, names="method", title="Payment Methods")
    st.plotly_chart(payment_fig, use_container_width=True)

with col2:
    revenue_fig = px.histogram(filtered_df, x="amount", nbins=10, title="Revenue Distribution")
    st.plotly_chart(revenue_fig, use_container_width=True)

st.success(" Dashboard Loaded with Single Selection Filters!")
