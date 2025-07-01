# to run this app, paste in your streamlit terminal the following command, changing to your local path: streamlit run c:\Users\camillob\Desktop\LeakNor\B_for_VA-Nett_Global_Model_App\leaknor\home.py

"""
Created on Wed May 21 13:11:40 2025

@author: camillob
"""

import streamlit as st

# Initialize session state variables if not already set
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

st.set_page_config(page_title="VA Data Processor", layout="wide")

st.title("Main Dashboard")

if st.session_state.uploaded_data is not None:
    st.success("Data uploaded and available for processing now.")
    st.dataframe(st.session_state.uploaded_data.head())
else:
    st.info("Please upload data using the **Upload** page.")


