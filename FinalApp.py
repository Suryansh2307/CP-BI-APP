import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Client Dashboard", layout="wide")

# ---------------- DATABASE CONNECTION ----------------
@st.cache_resource
def init_engine():
    creds = st.secrets["postgres"]
    engine = create_engine(
        f'postgresql+psycopg2://{creds["user"]}:{creds["password"]}@{creds["host"]}:{creds["port"]}/{creds["dbname"]}'
    )
    return engine

@st.cache_data
def run_query(query, params=None):
    engine = init_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

# ---------------- GET MIN & MAX DATE ----------------
df_dates = run_query(
    'SELECT MIN("Date") as min_date, MAX("Date") as max_date '
    'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY";'
)

min_date = pd.to_datetime(df_dates["min_date"].iloc[0]).date()
max_date = pd.to_datetime(df_dates["max_date"].iloc[0]).date()

# ---------------- CSS STYLING ----------------
st.markdown("""
    <style>
        .header {
            background-color: #e74c3c;
            padding: 12px;
            text-align: center;
            color: white;
            font-size: 22px;
            font-weight: bold;
            border-radius: 5px;
        }
        .sub-header {
            background-color: #0077c8;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 5px;
            margin-top: 10px;
        }
        .sample-box {
            background-color: #a0a0a0;
            padding: 8px;
            text-align: center;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- UI SECTION ----------------
st.markdown('<div class="header">PRIMARY INTELLIGENCE AND DISCOVERY INTERFACE CATI</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 6, 2])

with col1:
    date_range = st.date_input(
        "Select date range",
        [min_date, max_date],   # default range = full range from DB
        min_value=min_date,
        max_value=max_date
    )
    start_date, end_date = None, None
    if len(date_range) == 2:
        start_date, end_date = date_range

with col2:
    st.markdown('<div class="sub-header">STATE LEVEL</div>', unsafe_allow_html=True)

with col3:
    if start_date and end_date:
        df_const = run_query(
            'SELECT DISTINCT "What is the name of your constituency?" '
            'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
            'WHERE "Date" BETWEEN :start_date AND :end_date '
            'ORDER BY "What is the name of your constituency?";',
            params={"start_date": start_date, "end_date": end_date}
        )
    else:
        df_const = run_query(
            'SELECT DISTINCT "What is the name of your constituency?" '
            'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
            'ORDER BY "What is the name of your constituency?";'
        )

    if not df_const.empty:
        constituency = st.selectbox(
            "Constituency",
            df_const['What is the name of your constituency?'].tolist()
        )
    else:
        st.warning("No constituencies found in database.")
        constituency = None

# ---------------- SAMPLE SIZE ----------------
if start_date and end_date:
    df_sample = run_query(
        'SELECT COUNT(*) as sample_size '
        'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date;',
        params={"start_date": start_date, "end_date": end_date}
    )
else:
    df_sample = run_query(
        'SELECT COUNT(*) as sample_size '
        'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY";'
    )

sample_size = int(df_sample["sample_size"].iloc[0])
st.markdown(f'<div class="sample-box">SAMPLE SIZE: {sample_size:,}</div>', unsafe_allow_html=True)

