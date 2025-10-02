import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

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

# ---------------- PARTY PREF TABLE (Table 1) ----------------
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

    df[colname] = (
        df[colname]
        .replace(vote_map)
        .fillna("Other")   # treat blanks as Other
        .astype(str)
        .str.strip()
    )

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

def plot_party_table(party_pref_table):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#0d375e'
    total_color = '#f1c232'
    border_color = '#000000'

    fig = plt.figure(figsize=(8, 3))
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

# ---------------- TABLE 2: Voter Swing Matrix ----------------
def get_voter_swing_matrix(df):
    prev_col = ' Which party did you vote for in the previous 2022 elections? '
    curr_col = 'If elections were held in Punjab today, which party would you v'

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
        'Shiromani Akali Dal (Badal)': 'SAD (B)',
        'Did Not Vote': 'Did Not Vote'
    }

    df[prev_col] = df[prev_col].replace(vote_map).fillna("Other").astype(str).str.strip()
    df[curr_col] = df[curr_col].replace(vote_map).fillna("Other").astype(str).str.strip()

    pivot_counts = pd.pivot_table(
        df,
        index=prev_col,
        columns=curr_col,
        aggfunc='size',
        fill_value=0
    )

    pivot_percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
    pivot_percent = pivot_percent.round(2)

    total_votes_per_party = pivot_counts.sum(axis=0)
    total_votes = total_votes_per_party.sum()
    true_totals_percent = (total_votes_per_party / total_votes * 100).round(2)
    pivot_percent.loc['TOTAL'] = true_totals_percent

    pivot_percent = pivot_percent.applymap(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)

    row_order = ['AAP', 'INC', 'BJP', 'SAD (B)', 'SAD (A)', 'BSP',
                 'Akali Dal (WPD)', 'Other', 'Did Not Vote', "Can't Say", 'TOTAL']
    col_order = ['AAP', 'INC', 'BJP', 'SAD (B)', 'SAD (A)', 'BSP',
                 'Akali Dal (WPD)', 'NOTA', 'Other', "Can't Say"]

    pivot_percent = pivot_percent.reindex(index=row_order, columns=col_order, fill_value='-')

    pivot_data = pivot_percent.reset_index()
    pivot_data.rename(columns={pivot_data.columns[0]: 'PARTY'}, inplace=True)
    data = [list(pivot_data.columns)] + pivot_data.values.tolist()

    return data, pivot_data, row_order, col_order

def plot_voter_swing_matrix(data, pivot_data, col_order):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    fig = plt.figure(figsize=(12, 9))
    ax2 = fig.add_axes([0.042, 0.24, 0.96, 0.45])
    ax2.axis('off')

    # === Title above the table ===
    ax2.text(
        0.017, 1.10,   # moved higher so it does not overlap
        "2) VOTER SWING MATRIX",
        fontsize=14, fontproperties=bold_font, ha='left'
    )

    # === Column widths ===
    left_col_width = 0.14
    table_col_widths = [0.075] * len(col_order)

    table2 = ax2.table(
        cellText=data,
        cellLoc='center',
        loc='center',
        colWidths=[left_col_width] + table_col_widths,
    )

    # Wrap header for Akali Dal (WPD)
    for col_idx, col_name in enumerate(pivot_data.columns):
        cell = table2[0, col_idx]
        if col_name == 'Akali Dal (WPD)':
            cell.get_text().set_text('Akali\nDal (WPD)')

    # === Styling ===
    for (row_idx, col_idx), cell in table2.get_celld().items():
        is_header = row_idx == 0
        is_first_col = col_idx == 0
        is_total_row = (row_idx == len(data) - 1)

        if is_total_row:
            cell.set_facecolor('#f1c232')  # Yellow total row
            cell.set_text_props(weight='bold', fontsize=10)
            cell.get_text().set_fontproperties(aptos_font)
            continue
        if is_header:
            cell.set_facecolor('#073763')  # Dark blue header
            cell.set_text_props(weight='bold', color='white', fontsize=10)
            cell.get_text().set_fontproperties(aptos_font)
            continue
        if is_first_col:
            cell.set_facecolor('#ffffff')
            cell.set_text_props(weight='bold', fontsize=10)
            cell.get_text().set_fontproperties(aptos_font)
            continue

        # diagonal highlight (same party row & col)
        row_party = data[row_idx][0]
        col_party = data[0][col_idx]
        if row_party == col_party:
            cell.set_facecolor('#d9d9d9')  # Grey diagonal
            cell.set_text_props(weight='bold', fontsize=10)
            cell.get_text().set_fontproperties(aptos_font)
        else:
            cell.set_facecolor('white')
            cell.set_text_props(fontsize=10)
            cell.get_text().set_fontproperties(aptos_font)

    table2.auto_set_font_size(False)
    table2.scale(1.08, 2.0)
    table2.set_fontsize(11)

    # === Side Label (2022 Assembly Election) ===
    fig.text(
        0.071, 0.480, '                               2022 ASSEMBLY ELECTION                               ',
        fontsize=12, fontproperties=bold_font,
        va='center', ha='center', rotation=90,
        bbox=dict(facecolor='#e4dfec', edgecolor='black', boxstyle='square,pad=0.9')
    )

    # === Top Banner (CURRENT TREND) ===
    table_x_start = 0.0905
    table_x_end = table_x_start + sum(table_col_widths)
    swing_table_top = 0.78  # moved higher so it sits above the table

    fig.patches.extend([
        Rectangle(
            (table_x_start, swing_table_top - 0.092),
            0.892,  # full width
            0.030,  # banner height
            transform=fig.transFigure,
            facecolor='#d9e1f2',
            edgecolor='black',
            linewidth=1,
            zorder=1
        )
    ])

    fig.text(
        (table_x_start + table_x_end) / 2,
        swing_table_top - 0.078,
        '                                      CURRENT TREND',
        fontsize=12, fontproperties=bold_font,
        ha='center',
        va='center',
        zorder=2
    )

    return fig


# ---------------- TABLE 3: MLA Change ----------------
def get_mla_change_table(df):
    mla_change_col = 'would you like to change your MLA this time?'

    df_mla = df.copy()
    df_mla = df_mla[df_mla[mla_change_col].str.upper() != 'CALL DISCONNECTED']

    # Group and count
    mla_counts = df_mla[mla_change_col].value_counts().reset_index()
    mla_counts.columns = ['RESPONSE', 'COUNT']

    # Calculate percentage
    total_count = mla_counts['COUNT'].sum()
    mla_counts['PERCENTAGE'] = (mla_counts['COUNT'] / total_count * 100).round(2).astype(str) + '%'

    # Response order
    response_order = [
        'Yes',
        'No',
        'Our MLA is doing a good job',
        'Other (please specify)',
        "Can't Say"
    ]
    mla_counts['RESPONSE'] = pd.Categorical(mla_counts['RESPONSE'], categories=response_order, ordered=True)
    mla_counts = mla_counts.sort_values('RESPONSE')

    # Append total row
    total_row = pd.DataFrame([{
        'RESPONSE': 'TOTAL',
        'COUNT': total_count,
        'PERCENTAGE': '100.00%'
    }])
    mla_counts = pd.concat([mla_counts, total_row], ignore_index=True)

    return mla_counts

def plot_mla_change_table(mla_counts):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#073763'
    total_color = '#f1c232'
    border_color = '#000000'

    fig = plt.figure(figsize=(7, 2))
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
    ax.axis('off')

    ax.text(
        -0.03, 1.05,
        "3) WOULD YOU LIKE TO CHANGE YOUR MLA THIS TIME?",
        fontsize=10, fontproperties=bold_font, ha='left'
    )

    table = ax.table(
        cellText=mla_counts.values.tolist(),
        colLabels=mla_counts.columns.tolist(),
        cellLoc='center',
        loc='upper center',
        colWidths=[0.45, 0.25, 0.30]
    )

    n_rows = len(mla_counts) + 1
    n_cols = len(mla_counts.columns)

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor(border_color)
            if i == 0:  # Header
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color='white', fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)
            elif i == n_rows - 1:  # Total row
                cell.set_facecolor(total_color)
                cell.set_text_props(weight='bold', fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)
            else:  # Data rows
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.3)

    return fig
    
import decimal

def excel_round(x, digits=2):
    """Excel-style rounding: ROUND_HALF_UP instead of Python's bankers rounding"""
    return float(decimal.Decimal(str(x)).quantize(
        decimal.Decimal('1.' + '0'*digits), rounding=decimal.ROUND_HALF_UP
    ))

def get_social_media_table(df):
    from collections import Counter
    
    social_col = 'Which social media platform do you use the most?'
    
    # Step 1: Copy DF
    df_social = df.copy()
    
    # Step 2: Split comma-separated platforms into lists
    df_social[social_col] = df_social[social_col].str.split(',')
    
    # Step 3: Explode into rows
    df_social = df_social.explode(social_col)
    
    # Step 4: Clean whitespace
    df_social[social_col] = df_social[social_col].str.strip()
    
    # Step 5: Remove invalids
    df_social = df_social[
        df_social[social_col].notna() &
        (df_social[social_col].str.upper() != 'CALL DISCONNECTED') &
        (df_social[social_col] != '')
    ]
    
    # Step 5.1: Replace "Whatsapp" → "Others"
    df_social[social_col] = df_social[social_col].replace(
        to_replace=r'(?i)^whatsapp$', value='Others', regex=True
    )
    
    # Step 6: Count mentions
    platform_counts = df_social[social_col].value_counts().reset_index()
    platform_counts.columns = ['RESPONSE', 'COUNT']
    
    # Step 7: Percentages (mention-based, sum = 100)
    platform_counts['PERCENTAGE'] = (
        platform_counts['COUNT'] / platform_counts['COUNT'].sum() * 100
    ).round(2).astype(str) + '%'
    
    # Final display
    social_counts_display = platform_counts[['RESPONSE', 'PERCENTAGE']]
    
    return social_counts_display



def plot_social_media_table(social_counts_display):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#073763'
    border_color = '#000000'

    fig = plt.figure(figsize=(6, 1.5))
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
    ax.axis('off')

    ax.text(
        0.001, 1.05,
        "4) SOCIAL MEDIA",
        fontsize=10, fontproperties=bold_font, ha='left'
    )

    table_data = [social_counts_display.columns.tolist()] + social_counts_display.values.tolist()

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='upper center',
        colWidths=[0.6, 0.4]
    )

    n_rows = len(table_data)
    n_cols = len(table_data[0])

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor(border_color)
            if i == 0:  # Header
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color='white', fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)
            else:
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.3)

    return fig



from difflib import SequenceMatcher

def get_mla_party_recall(df, mla_df):
    # --- Clean columns ---
    def clean_columns(df):
        df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '').str.replace('  ', ' ')
        return df
    
    df = clean_columns(df)
    mla_df = clean_columns(mla_df)

    df.rename(columns={
        'What is the name of your constituency?': 'Constituency',
        'What is the name of your MLA?': 'MLA_Response',
        'Which party does he/she belong to?': 'Party_Response'
    }, inplace=True)

    mla_df.rename(columns={
        'AC NAME': 'Constituency',
        '1. Who is your MLA ?': 'MLA_NAME',
        '2. MLA belongs to which party ?': 'MLA_PARTY'
    }, inplace=True)

    # --- Constituency cleaning ---
    def clean_constituency(val):
        if pd.isnull(val): 
            return val
        val = str(val).strip()
        if '-' in val:
            val = val.split('-')[0].strip()
        return val

    df['Constituency_Clean'] = df['Constituency'].apply(clean_constituency).str.lower().str.strip()
    mla_df['Constituency_Clean'] = mla_df['Constituency'].apply(clean_constituency).str.lower().str.strip()

    # --- Name normalization ---
    def normalize_name(name):
        if pd.isnull(name) or name == '':
            return ''
        name = str(name).lower().strip()
        prefixes = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'capt.', 'col.', 'maj.', 'lt.']
        suffixes = ['ji', 'sahib', 'saheb']
        words = name.split()
        if words and words[0].rstrip('.') in prefixes:
            words = words[1:]
        if words and words[-1] in suffixes:
            words = words[:-1]
        return ' '.join(words)

    def name_similarity(name1, name2):
        if not name1 or not name2:
            return 0
        return SequenceMatcher(None, name1, name2).ratio()

    merged_df = pd.merge(
        df,
        mla_df[['Constituency_Clean', 'MLA_NAME', 'MLA_PARTY']],
        on='Constituency_Clean',
        how='left'
    )

    merged_df['MLA_Response_Normalized'] = merged_df['MLA_Response'].apply(normalize_name)
    merged_df['MLA_NAME_Normalized'] = merged_df['MLA_NAME'].apply(normalize_name)

    # --- MLA Recall Matching ---
    def evaluate_mla_match(row):
        response_name = row['MLA_Response_Normalized']
        actual_name = row['MLA_NAME_Normalized']
        if not response_name or response_name in ['', 'nan', 'call disconnected', "don't know", "can't say"]:
            return 'NO RESPONSE'
        if not actual_name:
            return 'NO MLA DATA'
        if response_name == actual_name:
            return 'CORRECT'
        return 'INCORRECT'

    merged_df['MLA_Match_Status'] = merged_df.apply(evaluate_mla_match, axis=1)
    valid_responses = merged_df[~merged_df['MLA_Match_Status'].isin(['NO MLA DATA','NO RESPONSE'])]

    mla_summary = valid_responses['MLA_Match_Status'].value_counts().reset_index()
    mla_summary.columns = ['RESPONSE', 'COUNT']
    total_valid = mla_summary['COUNT'].sum()
    mla_summary['PERCENTAGE'] = (mla_summary['COUNT'] / total_valid * 100).round(2).astype(str) + '%'
    mla_summary = mla_summary[['RESPONSE','PERCENTAGE']].set_index('RESPONSE').reindex(['CORRECT','INCORRECT']).reset_index().dropna()

    # --- Party Recall Matching ---
    def normalize_party(name):
        if pd.isnull(name) or name == '':
            return ''
        return str(name).lower().strip()

    merged_df['Party_Response_Normalized'] = merged_df['Party_Response'].apply(normalize_party)
    merged_df['MLA_PARTY_Normalized'] = merged_df['MLA_PARTY'].apply(normalize_party)

    def evaluate_party_match(row):
        response = row['Party_Response_Normalized']
        actual = row['MLA_PARTY_Normalized']
        if response in ['', 'nan', 'call disconnected', "can't say"]:
            return 'NO RESPONSE'
        if not actual:
            return 'NO MLA DATA'
        if response == actual or actual in response or response in actual:
            return 'CORRECT'
        return 'INCORRECT'

    merged_df['Party_Match_Status'] = merged_df.apply(evaluate_party_match, axis=1)
    party_valid = merged_df[~merged_df['Party_Match_Status'].isin(['NO MLA DATA','NO RESPONSE'])]

    partymap_summary = party_valid['Party_Match_Status'].value_counts().reset_index()
    partymap_summary.columns = ['RESPONSE','COUNT']
    total_party_valid = partymap_summary['COUNT'].sum()
    partymap_summary['PERCENTAGE'] = (partymap_summary['COUNT'] / total_party_valid * 100).round(2).astype(str) + '%'
    partymap_summary = partymap_summary[['RESPONSE','PERCENTAGE']].set_index('RESPONSE').reindex(['CORRECT','INCORRECT']).reset_index().dropna()

    return mla_summary, partymap_summary

def plot_mla_party_tables(mla_summary, partymap_summary):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#073763'
    border_color = '#000000'

    # === MLA Recall (Table 5) ===
    fig1 = plt.figure(figsize=(5,1))
    ax1 = fig1.add_axes([0.05,0.2,0.9,0.7])
    ax1.axis('off')
    ax1.text(-0.099,1.05,"5) MLA RECALL: WHO IS YOUR MLA ?",fontsize=11,fontproperties=bold_font,ha='left')

    table_data_5 = [mla_summary.columns.tolist()] + mla_summary.values.tolist()
    table5 = ax1.table(cellText=table_data_5,cellLoc='center',loc='upper center',colWidths=[0.5,0.5])
    for i in range(len(table_data_5)):
        for j in range(2):
            cell = table5[i,j]
            cell.set_edgecolor(border_color)
            if i==0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold',color='white',fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)
            else:
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)
    table5.auto_set_font_size(False)
    table5.set_fontsize(11)
    table5.scale(1.2,1.3)

    # === MLA’s Party Recall (Table 6) ===
    fig2 = plt.figure(figsize=(5,1))
    ax2 = fig2.add_axes([0.05,0.2,0.9,0.7])
    ax2.axis('off')
    ax2.text(-0.099,1.05,"6) MLA’S PARTY RECALL",fontsize=11,fontproperties=bold_font,ha='left')

    table_data_6 = [partymap_summary.columns.tolist()] + partymap_summary.values.tolist()
    table6 = ax2.table(cellText=table_data_6,cellLoc='center',loc='upper center',colWidths=[0.5,0.5])
    for i in range(len(table_data_6)):
        for j in range(2):
            cell = table6[i,j]
            cell.set_edgecolor(border_color)
            if i==0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold',color='white',fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)
            else:
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)
    table6.auto_set_font_size(False)
    table6.set_fontsize(11)
    table6.scale(1.2,1.3)

    return fig1, fig2


def get_whatsapp_usage(df):
    df = df.copy()
    df.rename(columns={'Do you use WhatsApp?': 'WhatsApp'}, inplace=True)
    df['WhatsApp'] = df['WhatsApp'].str.upper()

    whatsapp_summary = (
        df['WhatsApp']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename('PERCENTAGE')
        .reset_index()
    )

    whatsapp_summary.columns = ['RESPONSE', 'PERCENTAGE']
    whatsapp_summary['PERCENTAGE'] = whatsapp_summary['PERCENTAGE'].astype(str) + '%'

    whatsapp_order = ['YES', 'NO']
    whatsapp_summary = whatsapp_summary.set_index('RESPONSE').reindex(whatsapp_order).reset_index().dropna()

    return whatsapp_summary
def plot_whatsapp_table(whatsapp_summary):
    bold_font = fm.FontProperties(fname="Aptos-Display-Bold.ttf")
    aptos_font = fm.FontProperties(fname="Aptos-Display.ttf")

    header_color = '#073763'
    border_color = '#000000'

    fig = plt.figure(figsize=(5,1))
    ax = fig.add_axes([0.05,0.2,0.9,0.7])
    ax.axis('off')

    ax.text(
        -0.099, 1.05,
        "7) WHATSAPP USAGE",
        fontsize=12, fontproperties=bold_font, ha='left'
    )

    table_data_7 = [whatsapp_summary.columns.tolist()] + whatsapp_summary.values.tolist()
    table7 = ax.table(
        cellText=table_data_7,
        cellLoc='center',
        loc='upper center',
        colWidths=[0.5,0.5]
    )

    for i in range(len(table_data_7)):
        for j in range(2):
            cell = table7[i,j]
            cell.set_edgecolor(border_color)
            if i==0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color='white', fontsize=10)
                cell.get_text().set_fontproperties(aptos_font)
            else:
                cell.set_facecolor('white')
                cell.set_text_props(fontsize=9)
                cell.get_text().set_fontproperties(aptos_font)

    table7.auto_set_font_size(False)
    table7.set_fontsize(11)
    table7.scale(1.2,1.3)

    return fig


# ---------------- HELPER: SPACING ----------------


# ---------------- UI LAYOUT ----------------
# Red header
st.markdown(
    '<div style="background-color:#e74c3c; padding:12px; text-align:center; '
    'color:white; font-size:22px; font-weight:bold; border-radius:5px;">'
    'PRIMARY INTELLIGENCE AND DISCOVERY INTERFACE CATI</div>',
    unsafe_allow_html=True
)

# Filters
col1, col2, col3 = st.columns([2, 6, 2])

df_dates = run_query(
    "SELECT MIN(\"Date\") as min_date, MAX(\"Date\") as max_date "
    "FROM \"PUNJAB_2025\".\"CP_SURVEY_14_JULY\" "
    "WHERE \"What is the name of your constituency?\" NOT IN "
    "('Call Disconnected','Don''t Know','','OUT of Assembly/ OUT of State');"
)
min_date = pd.to_datetime(df_dates["min_date"].iloc[0]).date()
max_date = pd.to_datetime(df_dates["max_date"].iloc[0]).date()

with col1:
    date_range = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)

with col2:
    st.markdown(
        '<div style="background-color:#0077c8; padding:10px; margin-top:24px; '
        'text-align:center; color:white; font-size:18px; font-weight:bold; border-radius:5px;">'
        'STATE LEVEL</div>',
        unsafe_allow_html=True
    )

with col3:
    df_const = run_query(
        "SELECT DISTINCT \"What is the name of your constituency?\" as const "
        "FROM \"PUNJAB_2025\".\"CP_SURVEY_14_JULY\" "
        "WHERE \"Date\" BETWEEN :start_date AND :end_date "
        "AND \"What is the name of your constituency?\" NOT IN "
        "('Call Disconnected','Don''t Know','','OUT of Assembly/ OUT of State');",
        params={"start_date": start_date, "end_date": end_date}
    )

    const_list = (
        df_const["const"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    const_list.insert(0, "All")
    constituency = st.selectbox("Constituency", const_list)

# ---------------- MAIN DATA QUERY ----------------
if constituency == "All":
    df = run_query(
        "SELECT \"Date\", "
        "\"What is the name of your constituency?\", "
        "\"If elections were held in Punjab today, which party would you v\", "
        "\" Which party did you vote for in the previous 2022 elections? \", "
        "\"would you like to change your MLA this time?\", "
        "\"Which social media platform do you use the most?\", "
        "\"What is the name of your MLA?\", "
        "\"Which party does he/she belong to?\", "
        "\"Do you use WhatsApp?\" "  
        "FROM \"PUNJAB_2025\".\"CP_SURVEY_14_JULY\" "
        "WHERE \"Date\" BETWEEN :start_date AND :end_date "
        "AND \"What is the name of your constituency?\" NOT IN "
        "('Call Disconnected','Don''Know','','OUT of Assembly/ OUT of State');",
        params={"start_date": start_date, "end_date": end_date}
    )
else:
    df = run_query(
        "SELECT \"Date\", "
        "\"What is the name of your constituency?\", "
        "\"If elections were held in Punjab today, which party would you v\", "
        "\" Which party did you vote for in the previous 2022 elections? \", "
        "\"would you like to change your MLA this time?\", "
        "\"Which social media platform do you use the most?\", "
        "\"What is the name of your MLA?\", "
        "\"Which party does he/she belong to?\", "
        "\"Do you use WhatsApp?\" "  
        "FROM \"PUNJAB_2025\".\"CP_SURVEY_14_JULY\" "
        "WHERE \"Date\" BETWEEN :start_date AND :end_date "
        "AND \"What is the name of your constituency?\" = :const "
        "AND \"What is the name of your constituency?\" NOT IN "
        "('Call Disconnected','Don''Know','','OUT of Assembly/ OUT of State');",
        params={"start_date": start_date, "end_date": end_date, "const": constituency}
    )




# ---------------- TABLES ----------------
if not df.empty:
    party_pref_table = get_party_preference_table(df)
    sample_size = int(party_pref_table.loc[party_pref_table["PARTY PREFERENCE"] == "TOTAL", "COUNT"].values[0])

    st.markdown(
        f'<div style="background-color:#a0a0a0; padding:8px; text-align:center; '
        f'color:white; font-size:16px; font-weight:bold; border-radius:5px;">'
        f'SAMPLE SIZE: {sample_size:,}</div>',
        unsafe_allow_html=True
    )

    # --- Keep proper gap after Sample Size ---
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    # Table 1
    fig1 = plot_party_table(party_pref_table)
    st.pyplot(fig1)

    # --- Small gap only between Table 1 and 2 ---
    

    # Table 2
    data, pivot_data, row_order, col_order = get_voter_swing_matrix(df)
    fig2 = plot_voter_swing_matrix(data, pivot_data, col_order)
    st.pyplot(fig2)
    
    # Table 3
    mla_counts = get_mla_change_table(df)
    fig3 = plot_mla_change_table(mla_counts)
    st.pyplot(fig3)
    
    # Table 4
    social_counts_display = get_social_media_table(df)
    fig4 = plot_social_media_table(social_counts_display)
    st.pyplot(fig4)

    # Load MLA list Excel once
    mla_df = pd.read_excel("MLALIST2.xlsx")

    # Get recall summaries
    mla_summary, partymap_summary = get_mla_party_recall(df, mla_df)

    # Plot & show
    fig5, fig6 = plot_mla_party_tables(mla_summary, partymap_summary)
    st.pyplot(fig5)
    st.pyplot(fig6)
    
    # Table 7
    whatsapp_summary = get_whatsapp_usage(df)
    fig7 = plot_whatsapp_table(whatsapp_summary)
    st.pyplot(fig7)


else:
    st.warning("No data available for the selected filters.")
