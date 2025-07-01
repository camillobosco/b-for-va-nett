# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:15:41 2025
@author: camillob
"""

import streamlit as st
import pandas as pd
import pyodbc
import tempfile
import os

st.title("Upload VA Data")

# ---------------- RESET STATE BUTTON ----------------
if st.button("🔁 Reset and Load New File"):
    for key in list(st.session_state.keys()):
        if key.startswith("df_") or "uploaded" in key or "filtered" in key:
            del st.session_state[key]
    st.experimental_rerun()

# ---------------- FILE UPLOAD ----------------

st.write("### Upload your data file")

# st.markdown("""
# You can upload either:

# - **A Microsoft Access database** (`.mdb` / `.accdb`) containing the `VA_LINE` and `VA_DIARY` tables  
#   👉 *Recommended for working with original VA data from Kommuner.*

# **OR**

# - **An Excel file** (`.xlsx`) with the following required columns:  
# `LSID, STATUS, COUNTY, LENGTH, DATECHANGED, MATERIAL, DIM, YEAR, GROUNDSURFACE, SONE, FROM_PSID, TO_PSID, DBR1`  
# 👉 *Useful for working on pre-cleaned datasets.*
# """)

st.markdown("""
<style>
.tooltip {
  position: relative;
  display: inline-block;
  cursor: help;
  color: #0366d6;
  font-weight: bold;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 280px;
  background-color: #f9f9f9;
  color: #000;
  text-align: left;
  border-radius: 6px;
  padding: 8px;
  border: 1px solid #ccc;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 0%;
  opacity: 0;
  transition: opacity 0.3s;
  font-weight: normal;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>

You can upload either:

- <span class="tooltip">Microsoft Access database
    <span class="tooltiptext">Files with `.mdb` or `.accdb` extension containing original VA data such as `VA_LINE` and `VA_DIARY` tables.</span>
  </span>  
  👉 <em>Recommended for working with original VA data from Kommuner.</em>

**OR**

- <span class="tooltip">Excel file
    <span class="tooltiptext">An `.xlsx` file that includes all required columns: LSID, STATUS, COUNTY, LENGTH, etc.</span>
  </span>  
  👉 <em>Useful for working on pre-cleaned datasets.</em>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Access (.mdb/.accdb) or Excel (.xlsx) file (max ~200MB)", type=["mdb", "accdb", "xlsx"])
temp_db_path = None

if uploaded_file is not None and uploaded_file.name.lower().endswith(".xlsx"):
    try:
        df_excel = pd.read_excel(uploaded_file)

        required_excel_cols = {
            'LSID', 'STATUS', 'COUNTY', 'LENGTH', 'DATECHANGED',
            'MATERIAL', 'DIM', 'YEAR', 'GROUNDSURFACE', 'SONE',
            'FROM_PSID', 'TO_PSID', 'DBR1'
        }

        if not required_excel_cols.issubset(df_excel.columns):
            st.error(f"❌ Excel file is missing required columns: {required_excel_cols - set(df_excel.columns)}")
            st.stop()

        st.session_state.df_excel = df_excel
        if 'df_line_uploaded' in st.session_state:
            del st.session_state.df_line_uploaded
            del st.session_state.df_line_filtered
            del st.session_state.df_diary
        
        #st.session_state.df_line_uploaded = df_excel.copy()
        #st.session_state.df_excel = df_excel
        #st.session_state.df_diary = pd.DataFrame()  # Set empty since no diary in Excel

        st.success("✅ Excel file loaded and stored in session state.")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error while reading Excel file: {e}")
        st.stop()

# Continue with Access file logic if not Excel
if uploaded_file is not None and uploaded_file.name.lower().endswith((".mdb", ".accdb")):
    if uploaded_file.size > 200 * 1024 * 1024:
        st.warning("⚠️ File too large. Please paste the full path to the Access file on your system.")
        manual_path = st.text_input("Enter full path to the Access .mdb or .accdb file:")
        if manual_path and os.path.exists(manual_path):
            temp_db_path = manual_path
        else:
            st.error("❌ No valid file path provided. Please check the path and try again.")
            st.stop()
    else:
        temp_dir = tempfile.mkdtemp()
        temp_db_path = os.path.join(temp_dir, "uploaded_db.mdb")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
else:
    st.info("Alternatively, paste the full path to your Access file:")
    manual_path = st.text_input("Enter full path to the Access .mdb or .accdb file:")
    if manual_path and os.path.exists(manual_path):
        temp_db_path = manual_path
    else:
        st.warning("Please upload or paste a valid Microsoft Access database path to continue.")
        st.stop()

# ---------------- CONNECT TO ACCESS DATABASE ----------------

conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={temp_db_path};"
)
conn = pyodbc.connect(conn_str)

# ---------------- VALIDATE TABLES ----------------

try:
    required_tables = {'VA_LINE', 'VA_DIARY'}
    cursor = conn.cursor()
    tables = {row.table_name for row in cursor.tables(tableType='TABLE')}

    if not required_tables.issubset(tables):
        missing = required_tables - tables
        st.error(f"❌ Missing required tables: {', '.join(missing)}. Please upload a valid Access file.")
        st.stop()

    required_va_line_cols = {'LSID', 'YEAR', 'MATERIAL', 'DIM', 'OLDDIM', 'OLDYEAR'}
    required_va_diary_cols = {'COBJID', 'CNDATE'}

    va_line_sample = pd.read_sql('SELECT TOP 1 * FROM VA_LINE', conn)
    va_diary_sample = pd.read_sql('SELECT TOP 1 * FROM VA_DIARY', conn)

    if not required_va_line_cols.issubset(set(va_line_sample.columns)):
        st.error(f"❌ VA_LINE table is missing expected columns: {required_va_line_cols - set(va_line_sample.columns)}.")
        st.stop()

    if not required_va_diary_cols.issubset(set(va_diary_sample.columns)):
        st.error(f"❌ VA_DIARY table is missing expected columns: {required_va_diary_cols - set(va_diary_sample.columns)}.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error while validating the Access file: {e}")
    st.stop()

# ---------------- LOAD TABLES ----------------

df_line = pd.read_sql("SELECT * FROM VA_LINE", conn)
df_diary = pd.read_sql("SELECT * FROM VA_DIARY", conn)

df_line = df_line[(df_line['FCODE'] == 'VL') & (df_line['STATUS'] != 'P')]

# ---------------- SAVE TO SESSION STATE ----------------

#st.session_state.df_line = df_line
st.session_state.df_line_uploaded = df_line.copy()
st.session_state.df_diary = df_diary.copy()
if 'df_excel' in st.session_state:
    del st.session_state.df_excel
st.success("✅ Access tables successfully loaded and stored in session state.")

# ---------------- OPTIONAL FILTERING SECTION ----------------

st.write("### (Optional) Apply Filters")

enable_filters = st.checkbox("🧪 Enable RESPONSIBLE and OWNER filters")

if enable_filters:

    if 'df_line_filtered' not in st.session_state:
        st.session_state.df_line_filtered = st.session_state.df_line_uploaded.copy()

    df_line_base = st.session_state.df_line_filtered.copy()

    responsible_options = sorted(st.session_state.df_line_uploaded['RESPONSIBLE'].dropna().unique())
    owner_options = sorted(st.session_state.df_line_uploaded['OWNER'].dropna().unique())

    default_responsibles = ["K"] if "K" in responsible_options else []
    default_owners = ["K"] if "K" in owner_options else []

    with st.form("va_filter_form"):
        st.markdown("Use the checkboxes to activate filters. Click **Apply Filters** to continue.")

        use_responsible = st.checkbox("Enable RESPONSIBLE filter", value=True)
        selected_responsibles = st.multiselect("Select RESPONSIBLE(s):", options=responsible_options, default=default_responsibles)

        use_owner = st.checkbox("Enable OWNER filter", value=True)
        selected_owners = st.multiselect("Select OWNER(s):", options=owner_options, default=default_owners)

        apply_filters = st.form_submit_button("✅ Apply Filters")

    if st.button("🔄 Reset Filters"):
        st.session_state.df_line_filtered = st.session_state.df_line_uploaded.copy()
        st.success("✅ Filters reset. You are now starting again from original uploaded data.")
        st.stop()

    if apply_filters:
        df_line_filtered = df_line_base

        if not (use_responsible or use_owner):
            st.error("❌ You must enable at least one filter (RESPONSIBLE or OWNER).")
            st.stop()

        if use_responsible and selected_responsibles:
            df_line_filtered = df_line_filtered[df_line_filtered['RESPONSIBLE'].isin(selected_responsibles)]

        if use_owner and selected_owners:
            df_line_filtered = df_line_filtered[df_line_filtered['OWNER'].isin(selected_owners)]

        if df_line_filtered.empty:
            st.error("❌ No records match the selected filters.")
            st.stop()

        st.session_state.df_line_filtered = df_line_filtered
        st.success(f"✅ {len(df_line_filtered)} records selected after filtering.")
        st.dataframe(df_line_filtered.head(50))
    else:
        st.info("Filters are ready to apply. Tick and submit the form.")
else:
    st.session_state.df_line_filtered = st.session_state.df_line_uploaded.copy()
    st.info("No additional filters applied. Full dataset will be used for preprocessing.")
