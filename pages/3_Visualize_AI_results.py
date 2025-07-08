import pandas as pd
import numpy as np
from datetime import timedelta
from joblib import load
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st
from io import BytesIO

# ----------------------------
# Load models and extract feature names (cached)
# ----------------------------
st.title("Results of AI models")
predict_button = st.button("Predict!")


# Se i grafici sono già in session_state, plotta subito e interrompi ulteriori elaborazioni
if ('fig1_3' in st.session_state and
    'fig2_3' in st.session_state and
    'fig3_3' in st.session_state):
    
    tabs = st.tabs([
        "📊 Predictions Overview",
        "⚠️ High-Risk Pipes",
        "📐 Age vs Length"
    ])

    with tabs[0]:
        st.markdown("📊 Predictions Overview")
        st.plotly_chart(st.session_state['fig1_3'])
    with tabs[1]:
        st.markdown("⚠️ High-Risk Pipes")
        st.plotly_chart(st.session_state['fig2_3'])
    with tabs[2]:
        st.markdown("📐 Age vs Length")
        st.plotly_chart(st.session_state['fig3_3'])
        
    st.stop()

@st.cache_resource
def load_models_and_features():
    try:
        rf = load('Trained_models/global_random_forest_model.joblib')
        xgb_model = load('Trained_models/global_XGBoost_model.joblib')

        # Extract feature names from RF and XGB
        feature_names_rf = list(rf.feature_names_in_)
        feature_names_xgb = xgb_model.get_booster().feature_names

        if feature_names_rf != feature_names_xgb:
            st.warning("Warning: Feature names in RF and XGB models differ!")

        # We trust RF feature order (or choose one)
        feature_names = feature_names_rf

        return rf, xgb_model, feature_names

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

rf, xgb_model, feature_names = load_models_and_features()

# ----------------------------
# Process pipes function
# ----------------------------
def process_future_pipes(df_pipes: pd.DataFrame, forecast_years: int = 5):
    df_pipes = df_pipes[df_pipes['YEAR'].notna() & (df_pipes['YEAR'] >= 1850)]

    all_rows = []
    forecast_date = pd.to_datetime("2024-12-31") + timedelta(days=forecast_years * 365)

    for _, row in df_pipes.iterrows():
        index_breaks = [col for col in df_pipes.columns if col.startswith("DBR")]
        try:
            last_date = pd.to_datetime(f"{int(row['YEAR'])}/1/1")
        except Exception:
            continue

        unique_dates = pd.to_datetime(row[index_breaks], errors='coerce').dropna().drop_duplicates()
        last_break_date = unique_dates.max() if not unique_dates.empty else last_date

        if row.get('STATUS') == "D" and "_old" not in str(row.get('LSID')):
            age_days = (forecast_date - last_break_date).days
            all_rows.append({
                'LSID': row['LSID'],
                'LENGTH': row['LENGTH'],
                'MATERIAL': row['MATERIAL'],
                'DIM': row['DIM'],
                'YEAR': row['YEAR'],
                'previous_breaks': len(unique_dates),
                'Age': age_days,
                'Status': 0
            })

    df_all = pd.DataFrame(all_rows)
    df_all = df_all[(df_all['LENGTH'] > 1) & (df_all['Age'] > 2)].dropna()

    pipes_ps = [m for m in df_all['MATERIAL'].unique() if m.startswith("P") and m != "PVC"]
    df_all['MATERIAL'] = df_all['MATERIAL'].apply(
        lambda x: "PPs" if x in pipes_ps else x if x in ["SJG", "PVC", "SJK"] else "others"
    )

    X = pd.get_dummies(df_all.drop(columns=['Status']), dtype='int')

    # Align columns to feature_names from models
    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        # Reorder columns exactly as in feature_names
        X = X[feature_names]
    else:
        st.warning("Feature names not available; predictions may fail.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # (optional scaling; depends on model training)

    if rf is not None and xgb_model is not None:
        y_pred_rf = rf.predict(X)
        y_pred_xgb = xgb_model.predict(X)
        y_prob_rf = rf.predict_proba(X)[:, 1]
        y_prob_xgb = xgb_model.predict_proba(X)[:, 1]
    else:
        y_pred_rf = y_pred_xgb = np.zeros(len(X), dtype=int)
        y_prob_rf = y_prob_xgb = np.zeros(len(X))

    df_binary = df_all.copy()
    df_binary['RF'] = y_pred_rf
    df_binary['XGB'] = y_pred_xgb

    df_prob = df_all.copy()
    df_prob['RF'] = y_prob_rf
    df_prob['XGB'] = y_prob_xgb

    return df_binary, df_prob

# ----------------------------
# Label prediction helper
# ----------------------------
def label_prediction(row):
    if row['RF'] == 1 and row['XGB'] == 1:
        return 'Both Fail'
    elif row['RF'] == 1 or row['XGB'] == 1:
        return 'Disagree'
    else:
        return 'Both Survive'

# ----------------------------
# Streamlit UI
# ----------------------------


#uploaded_file = st.file_uploader("Upload pipes data Excel file", type=['xlsx', 'xls'])
    # Check if Access data is loaded (session_state has these dfs)
if ('df_line_preprocessed' in st.session_state):
    st.info("Access data detected...")
    df_AI = st.session_state.df_line_preprocessed.copy()
    st.success(f"Loaded data with {len(df_AI)} rows.")
elif ('df_excel' in st.session_state):
    st.info("Excel data detected...")
    df_AI = st.session_state.df_excel.copy()
    st.success(f"Loaded data with {len(df_AI)} rows.")
else:
    st.info("No data detected, go to the upload page...")
    st.stop()
    


if predict_button:
    if df_AI is None:
        st.warning("No data found. Please upload data first.")
    else:
        df_binary, df_prob = process_future_pipes(df_AI)
        df_binary['Prediction'] = df_binary.apply(label_prediction, axis=1)
        df_binary['Age_Group'] = pd.cut(df_binary['Age'], bins=[0, 3000, 6000, 9000, 12000, 15000],
                                        labels=["0–3k", "3–6k", "6–9k", "9–12k", "12k+"])

        high_risk_df = df_binary[df_binary['Prediction'] == 'Both Fail']

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_binary.to_excel(writer, sheet_name='Binary', index=False)
            high_risk_df.to_excel(writer, sheet_name='High risk pipes', index=False)
            df_prob.to_excel(writer, sheet_name='ProbFailure', index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="Download Results Data",
            data=excel_buffer,
            file_name="Results_VA_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        tabs = st.tabs([
            "📊 Predictions Overview",
            "⚠️ High-Risk Pipes",
            "📐 Age vs Length"
        ])

        with tabs[0]:
            counts = df_binary['Prediction'].value_counts().reindex(['Both Fail', 'Disagree', 'Both Survive'], fill_value=0)
            counts_df = counts.reset_index()
            counts_df.columns = ['Prediction_Label', 'Count']
            fig1 = px.bar(counts_df, x='Prediction_Label', y='Count',
              color='Prediction_Label', text='Count',
              color_discrete_map={'Both Fail': 'crimson', 'Disagree': 'orange', 'Both Survive': 'green'})
            fig1.update_traces(textposition='outside')
            st.session_state['fig1_3'] = fig1
            st.plotly_chart(fig1, use_container_width=True)

        with tabs[1]:
            st.dataframe(high_risk_df[['LSID', 'LENGTH', 'MATERIAL', 'YEAR', 'Age', 'Prediction']])
            st.markdown(f"**Total High-Risk Pipes:** {len(high_risk_df)}")

            #conteggio dei materiali tra i tubi high risk
            material_counts = high_risk_df['MATERIAL'].value_counts().reset_index()
            material_counts.columns = ['MATERIAL', 'count']
            fig2 = px.bar(material_counts, x='MATERIAL', y='count', text='count')
            fig2.update_traces(textposition='outside')
    
            st.session_state['fig2_3'] = fig2
            st.plotly_chart(fig2, use_container_width=True)
    
            st.download_button("Download CSV", high_risk_df.to_csv(index=False), "high_risk_pipes.csv")

        with tabs[2]:
            # Scatter plot: Age vs LENGTH, colored by Prediction
            fig3 = px.scatter(
                df_binary,
                x='Age',
                y='LENGTH',
                color='Prediction',
                hover_data=['LSID', 'MATERIAL', 'YEAR']
            )
            fig3.update_layout(title="Age vs Length by Prediction", xaxis_title="Age (days)", yaxis_title="Length (m)")
        
            st.session_state['fig3_3'] = fig3
            st.plotly_chart(fig3, use_container_width=True)

