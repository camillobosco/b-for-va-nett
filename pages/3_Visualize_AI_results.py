# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:19:54 2025

@author: camillob
"""
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump, load
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st
from io import BytesIO

def process_future_pipes(df_pipes: pd.DataFrame, forecast_years: int = 5) -> pd.DataFrame:
    df_pipes = df_pipes[~df_pipes['YEAR'].isna()]
    df_pipes = df_pipes[df_pipes['YEAR'] >= 1850]

    all_rows = []

    for i, row in df_pipes.iterrows():
        index_breaks = sorted([col for col in df_pipes.columns if col.startswith("DBR")])

        try:
            last_date = pd.to_datetime(f"{int(row['YEAR'])}/1/1", utc=True)
        except:
            continue

        # Parse and clean break dates
        unique_dates = pd.to_datetime(row[index_breaks], errors='coerce', utc=True).dropna()
        unique_dates = unique_dates.drop_duplicates().sort_values()
        n = len(unique_dates)

        # Only include pipes marked for decommissioning (but not "_old")
        if row['STATUS'] == "D" and "_old" not in str(row['LSID']):
            last_break_date = unique_dates.max() if n > 0 else last_date

            # Add forecast_years (converted to days)
            dato1 = pd.to_datetime("2024-12-31", utc=True) + timedelta(days=forecast_years * 365)
            age = (dato1.normalize() - last_break_date.normalize()).days

            all_rows.append({
                'LSID': row['LSID'],
                'LENGTH': row['LENGTH'],
                'MATERIAL': row['MATERIAL'],
                'DIM': row['DIM'],
                'YEAR': row['YEAR'],
                'previous_breaks': n,
                'Age': age,
                'status': "S"
            })

        if i % 900 == 0 and i > 0:
            print(f"Processed row: {i}")

    df_all = pd.DataFrame(all_rows)

    # Filter outliers
    df_all = df_all[df_all['LENGTH'] > 1]
    df_all = df_all[df_all['Age'] > 2]
    df_all = df_all.dropna()

    # Add binary status
    df_all['Status'] = 0  # Only 'S'

    # Group pipe materials
    pipes_ps = [m for m in df_all['MATERIAL'].unique() if m.startswith("P") and m != "PVC"]
    df_all['MATERIAL'] = df_all['MATERIAL'].apply(
        lambda x: "PPs" if x in pipes_ps else x
    )
    df_all['MATERIAL'] = df_all['MATERIAL'].apply(
        lambda x: x if x in ["SJG", "PVC", "SJK", "PPs"] else "others"
    )

    # 
    sub_data = df_all[['LENGTH', 'DIM',"YEAR", 'Age','MATERIAL','previous_breaks','Status']] # this is the pre-processed data 


    sub_data = sub_data.dropna()
    X = sub_data.drop(columns=['Status'])
    X = pd.get_dummies(X, dtype='int')
    y = sub_data['Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)



    # rf
    rf =  load('Trained_models/global_random_forest_model.joblib')
    y_pred_rf = rf.predict(X)
    y_pred_rf_prob = rf.predict_proba(X)[:, 1]

    #XGB
    xgb_model =  load('Trained_models/global_XGBoost_model.joblib')
    y_pred_xgb = xgb_model.predict(X)
    y_pred_xgb_prob = xgb_model.predict_proba(X)[:, 1]

    df_results1 = pd.DataFrame(
    np.column_stack((y_pred_rf, y_pred_xgb)),
    columns=['RF', 'XGB']
)

    df_results_binary = pd.concat([df_all.reset_index(drop=True), df_results1.reset_index(drop=True)], axis=1)

    df_results2 = pd.DataFrame(
        np.column_stack((y_pred_rf_prob, y_pred_xgb_prob)),
        columns=['RF', 'XGB']
    )

    df_results_prob = pd.concat([df_all.reset_index(drop=True), df_results2.reset_index(drop=True)], axis=1)

    results_list = [df_results_binary, df_results_prob]

    return results_list

def label_prediction(row):
    if row['RF'] == 1 and row['XGB'] == 1:
        return 'Both Fail'
    elif row['RF'] == 1 or row['XGB'] == 1:
        return 'Disagree'
    else:
        return 'Both Survive'

st.title("Results of AI models")

# Button for user to start the analysis
predict_results = st.button("Predict !")

if predict_results:
    if    'df_line_preprocessed' not in st.session_state and 'df_excel' not in st.session_state:
        st.warning("No data found. Please upload data in the Upload page and preprocess data if you upload an Access file.")
    elif  'df_line_preprocessed' in st.session_state:
        st.warning("You are using data loaded via an Access file.")
        df1 = st.session_state.df_line_preprocessed
        df_forecast_5y = process_future_pipes(df1)
    elif  'df_excel' in st.session_state:
        st.warning("You are using data loaded via an Excel file.")
        df1 = st.session_state.df_excel
        df_forecast_5y = process_future_pipes(df1)
    
    df = df_forecast_5y[0]
    df['Prediction'] = df.apply(label_prediction, axis=1)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 3000, 6000, 9000, 12000, 15000], labels=["0–3k", "3–6k", "6–9k", "9–12k", "12k+"])
    high_risk_df = df[(df['RF'] == 1) & (df['XGB'] == 1)]

    #Elhadi, you need to integrate your work starting from this line. You will plot results via an existing session state, either df_excel or df_line_preprocessed

    # EXPORT TO EXCEL
    # EXPORT TO EXCEL
    # Create a BytesIO buffer
    excel_buffer = BytesIO()

    # Write DataFrames to different sheets
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:

        df.to_excel(writer, sheet_name='Binary', index=False)
        high_risk_df.to_excel(writer, sheet_name='High risk pipes', index=False)
        df_forecast_5y[1].to_excel(writer, sheet_name='ProbFailure', index=False)
    

    # Reset buffer position
    excel_buffer.seek(0)

    # Provide download button in Streamlit
    st.download_button(
        label="Download Results Data",
        data=excel_buffer,
        file_name="Results_VA_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    # ---------------- VISUALIZATIONS ----------------
        
        # --- Tabs for Visualization ---
    st.markdown("## Predction results")



    df['Prediction'] = df.apply(label_prediction, axis=1)

    # High-risk pipes

    # Age group for heatmap
   


    # Tab layout
    tab_labels = [
        "📊 Predictions Overview",
        "⚠️ High-Risk Pipes",
        "📐 Age vs Length",
        "🔥 Material vs Age Heatmap",
        "📆 Risk by Installation Year",
    ]
    tabs = st.tabs(tab_labels)

    # --- Tab 1: Predictions Overview (Interactive Bar) ---
    with tabs[0]:
        st.markdown("### Pipe Failure Predictions by Models")
        bar_data = df['Prediction'].value_counts().reindex(['Both Fail', 'Disagree', 'Both Survive']).reset_index()
        bar_data.columns = ['Prediction', 'Count']

        fig1 = px.bar(bar_data, x='Prediction', y='Count', color='Prediction',
                    color_discrete_map={'Both Fail': 'crimson', 'Disagree': 'orange', 'Both Survive': 'green'},
                    title="Pipe Prediction Categories", text='Count')
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

    # --- Tab 2: High-Risk Pipes Table ---
    with tabs[1]:
        st.markdown("### Pipes Predicted to Fail by Both Models")
        st.dataframe(high_risk_df[['LSID', 'LENGTH', 'MATERIAL', 'YEAR', 'Age', 'Prediction']])
        st.markdown(f"**Total High-Risk Pipes:** {len(high_risk_df)}")
        st.download_button("Download CSV", high_risk_df.to_csv(index=False), "high_risk_pipes.csv")

    # --- Tab 3: Age vs Length Scatter ---
    with tabs[2]:
        st.markdown("### Pipe Age vs Length")
        fig2 = px.scatter(
            df, x='Age', y='LENGTH', color='Prediction',
            hover_data=['LSID', 'MATERIAL', 'YEAR'],
            title="Pipe Age vs Length Colored by Failure Prediction",
            color_discrete_map={'Both Fail': 'crimson', 'Disagree': 'orange', 'Both Survive': 'green'}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 4: Material vs Age Group Heatmap ---
    with tabs[3]:
        st.markdown("### Heatmap: Failure Risk by Material and Age Group")
        heat_data = df[df['Prediction'] == 'Both Fail'].groupby(['MATERIAL', 'Age_Group']).size().reset_index(name='Count')
        heat_pivot = heat_data.pivot(index='MATERIAL', columns='Age_Group', values='Count').fillna(0)

        fig3 = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(str),
            y=heat_pivot.index,
            colorscale='YlOrRd',
            hoverongaps=False,
            colorbar=dict(title='High-Risk Count')
        ))
        fig3.update_layout(
            title="High-Risk Pipes by Material and Age Group",
            xaxis_title="Age Group",
            yaxis_title="Material"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 5: Risk Trend by Installation Year ---
    with tabs[4]:
        st.markdown("### Failure Risk by Installation Year")
        year_trend = df.groupby(['YEAR', 'Prediction']).size().reset_index(name='Count')
        fig4 = px.line(
            year_trend, x='YEAR', y='Count', color='Prediction',
            markers=True,
            title="Trend of Failure Predictions by Installation Year",
            color_discrete_map={'Both Fail': 'crimson', 'Disagree': 'orange', 'Both Survive': 'green'}
        )
        st.plotly_chart(fig4, use_container_width=True)


