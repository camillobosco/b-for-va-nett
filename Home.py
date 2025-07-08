# to run this app, paste in your streamlit terminal the following command, changing to your local path: 
# streamlit run c:\Users\camillob\Desktop\LeakNor\B_for_VA-Nett_Global_Model_App\leaknor\home.py

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

st.title("🚀 VA Data Processor Dashboard")

st.markdown("""
Welcome to the **VA Data Processor App!**  
This tool guides you through a smart and flexible workflow to process and analyze your pipe network data with advanced AI models.
""")

with st.expander("📌 Step 1 – Upload your data"):
    st.markdown("""
- Upload a valid **Excel** or **Access (.mdb)** file containing your network data.
- Reset the uploaded file and data at any time.
- Choose whether to **drag and drop** your file or **enter the full file path manually**.

> ⚠️ *Important:* If you plan to download processed CSV or Excel results, do it **before switching to another page**.  
> Otherwise, you’ll need to re-upload the file and repeat the steps.
""")

with st.expander("🔄 Step 2 – Full preprocessing (required for Access files)"):
    st.markdown("""
When you upload an **Access database**, you must go through the preprocessing page:
- This step runs the full pipeline to clean and process your data.
- It also provides valuable statistics about the pipes in your dataset.

Even for Excel users, exploring this page is useful to understand the dataset quality and characteristics.
""")

with st.expander("🤖 Step 3 – AI predictions (smart insights!)"):
    st.markdown("""
- If you upload an **Excel file**, you can go straight to this page after uploading.
- If you upload an **Access database**, you’ll need to finish the preprocessing step first.

Here you can:
- Explore and **download AI-based predictions** highlighting the pipes at highest risk.
- Discover results from multiple **machine learning models** tailored to your data.

> Curious about the algorithms?  
> 📄 [Read our detailed PDF about the AI models here](https://your-link-to-pdf.com)
""")

st.markdown("""
---
✨ Enjoy exploring your network data smarter than ever!
""")

if st.session_state.uploaded_data is not None:
    st.success("✅ Data uploaded and ready for processing!")
    st.dataframe(st.session_state.uploaded_data.head())
else:
    st.info("📂 Please upload data using the **Upload** page first.")
