import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Client Dashboard", layout="wide")

# ---------------- DB CONNECTION ----------------
@st.cache_resource
def init_engine():
    creds = st.secrets["postgres"]  # stored in .streamlit/secrets.toml
    engine = create_engine(
        f'postgresql+psycopg2://{creds["user"]}:{creds["password"]}@'
        f'{creds["host"]}:{creds["port"]}/{creds["dbname"]}'
    )
    return engine

@st.cache_data
def run_query(query, params=None):
    engine = init_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

# ---------------- PARTY PREF TABLE HELPER ----------------
def get_party_preference_table(df):
    vote_map = {
        'Aam Aadmi Party (AAP)': 'AAP',
        'Akali Dal (Waris Punjab De)': 'Akali Dal (WPD)',
        'Bahujan Samaj Party (BSP)': 'BSP',
        'Bharatiya Janata Party (BJP)': 'BJP',
        "Can't Say": "Can't Say",
        'Congress': 'INC',
        'NOTA': 'NOTA',
        'Other': 'Other',
        'Shiromani Akali Dal (Amritsar)': 'SAD (A)',
        'Shiromani Akali Dal (Badal)': 'SAD (B)'
    }

    colname = 'If elections were held in Punjab today, which party would you v'

    df[colname] = df[colname].replace(vote_map).fillna('Null').astype(str).str.strip()

    party_order = ['AAP', 'INC', 'BJP', 'SAD (B)', 'SAD (A)', 
                   'BSP', 'Akali Dal (WPD)', 'NOTA', 'Other', "Can't Say"]

    party_counts = df[colname].value_counts()
    party_pref_table = pd.DataFrame({'PARTY PREFERENCE': party_order})
    party_pref_table['COUNT'] = party_pref_table['PARTY PREFERENCE'].map(party_counts).fillna(0).astype(int)

    total_party_count = party_pref_table['COUNT'].sum()
    party_pref_table['PERCENTAGE'] = (party_pref_table['COUNT'] / total_party_count * 100).apply(lambda x: f"{x:.2f}%")

    total_row = {
        'PARTY PREFERENCE': 'TOTAL',
        'COUNT': total_party_count,
        'PERCENTAGE': '100.00%'
    }
    party_pref_table = pd.concat([party_pref_table, pd.DataFrame([total_row])], ignore_index=True)

    return party_pref_table

# ---------------- PLOT TABLE HELPER ----------------
def plot_party_table(party_pref_table):
    # Fonts (make sure Aptos TTF files are in repo)
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#0d375e'
    total_color = '#f1c232'
    border_color = '#000000'

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
    ax.axis('off')

    ax.text(
        -0.03, 1.05,
        "1) IF THERE ARE ASSEMBLY ELECTIONS IN PUNJAB TODAY, WHICH PARTY WOULD YOU VOTE FOR?",
        fontsize=12, fontproperties=bold_font, ha='left'
    )

    table = ax.table(
        cellText=party_pref_table.values.tolist(),
        colLabels=party_pref_table.columns.tolist(),
        cellLoc='center',
        loc='upper center'
    )

    n_rows = len(party_pref_table) + 1
    n_cols = len(party_pref_table.columns)
    max_idx = party_pref_table.iloc[:-1]['COUNT'].idxmax()

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor(border_color)

            if i == 0:  # Header
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color='white', fontsize=12)
                cell.get_text().set_fontproperties(aptos_font)

            elif i == max_idx + 1:  # Highlight max row
                cell.set_facecolor('#00FF00')
                cell.set_text_props(weight='bold', fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)

            elif i == n_rows - 1:  # Totals row
                cell.set_facecolor(total_color)
                cell.set_text_props(weight='bold', color='black', fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)

            else:
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.05, 1.3)

    return fig

# ---------------- HELPER: WRAPPER FOR TABLE SECTIONS ----------------
def add_table_section(fig):
    """Ensures spacing before and after each table."""
    st.markdown("<br><br>", unsafe_allow_html=True)  # space before
    st.pyplot(fig)
    st.markdown("<br><br>", unsafe_allow_html=True)  # space after

# ---------------- UI LAYOUT (your original top sections) ----------------
# Red header
st.markdown('<div style="background-color:#e74c3c; padding:12px; text-align:center; color:white; font-size:22px; font-weight:bold; border-radius:5px;">PRIMARY INTELLIGENCE AND DISCOVERY INTERFACE CATI</div>', unsafe_allow_html=True)

# Filters
col1, col2, col3 = st.columns([2, 6, 2])

df_dates = run_query(
    'SELECT MIN("Date") as min_date, MAX("Date") as max_date FROM "PUNJAB_2025"."CP_SURVEY_14_JULY";'
)
min_date = pd.to_datetime(df_dates["min_date"].iloc[0]).date()
max_date = pd.to_datetime(df_dates["max_date"].iloc[0]).date()

with col1:
    date_range = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)

with col2:
    st.markdown('<div style="background-color:#0077c8; padding:10px; margin-top:24px; text-align:center; color:white; font-size:18px; font-weight:bold; border-radius:5px;">STATE LEVEL</div>', unsafe_allow_html=True)

with col3:
    df_const = run_query(
        'SELECT DISTINCT "What is the name of your constituency?" '
        'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date '
        'ORDER BY "What is the name of your constituency?";',
        params={"start_date": start_date, "end_date": end_date}
    )
    const_list = df_const['What is the name of your constituency?'].tolist()
    const_list.insert(0, "All")   # Add All option
    constituency = st.selectbox("Constituency", const_list)

# Sample size bar
if constituency == "All":
    df_sample = run_query(
        'SELECT COUNT(*) as sample_size FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date;',
        params={"start_date": start_date, "end_date": end_date}
    )
else:
    df_sample = run_query(
        'SELECT COUNT(*) as sample_size FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date '
        'AND "What is the name of your constituency?" = :const;',
        params={"start_date": start_date, "end_date": end_date, "const": constituency}
    )

sample_size = int(df_sample["sample_size"].iloc[0])
st.markdown(f'<div style="background-color:#a0a0a0; padding:8px; text-align:center; color:white; font-size:16px; font-weight:bold; border-radius:5px;">SAMPLE SIZE: {sample_size:,}</div>', unsafe_allow_html=True)

# ---------------- MAIN TABLES BELOW ----------------
if constituency == "All":
    df = run_query(
        'SELECT "Date", "What is the name of your constituency?", '
        '"If elections were held in Punjab today, which party would you v" '
        'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date;',
        params={"start_date": start_date, "end_date": end_date}
    )
else:
    df = run_query(
        'SELECT "Date", "What is the name of your constituency?", '
        '"If elections were held in Punjab today, which party would you v" '
        'FROM "PUNJAB_2025"."CP_SURVEY_14_JULY" '
        'WHERE "Date" BETWEEN :start_date AND :end_date '
        'AND "What is the name of your constituency?" = :const;',
        params={"start_date": start_date, "end_date": end_date, "const": constituency}
    )

if not df.empty:
    party_pref_table = get_party_preference_table(df)
    fig1 = plot_party_table(party_pref_table)
    add_table_section(fig1)
else:
    st.warning("No data available for the selected filters.")
