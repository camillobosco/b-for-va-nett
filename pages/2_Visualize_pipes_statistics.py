# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:37:59 2025

@author: camillob
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
st.title("Process and Plot VA Data")

# Button for user to start the analysis
run_analysis = st.button("Analyse and plot!")

if run_analysis:
    # Check if Access data is loaded (session_state has these dfs)
    if ('df_line_filtered' in st.session_state):
        st.info("Access data detected. Running full preprocessing...")

        # Retrieve the dataframes from session state
        df_line_for_preprocessing = st.session_state.df_line_filtered.copy()
        df_diary =  st.session_state.df_diary.copy()

        # Columns to keep
        columns_to_keep = [
            'LSID', 'STATUS', 'COUNTY', 'LENGTH', 'DATECHANGED', 'MATERIAL', 'DIM', 'YEAR', 'OLDMATERIAL', 'OLDDIM', 'OLDYEAR',
            'GROUNDSURFACE', 'SONE', 'FROM_PSID', 'TO_PSID'
        ]

        # Filter the dataframe to include only the columns in 'columns_to_keep'
        df_line_for_preprocessing = df_line_for_preprocessing[columns_to_keep]
        st.success("✅ Data successfully loaded from session state.")

        # --------- YOUR ORIGINAL PREPROCESSING SCRIPT STARTS HERE ---------
        # PIVOT DBRs INTO COLUMNS
        df_breaks = df_diary.sort_values(by='CNDATE')
        df_breaks['break_number'] = df_breaks.groupby('COBJID').cumcount() + 1
        df_breaks['column_name'] = 'DBR' + df_breaks['break_number'].astype(str)
        pivoted = df_breaks.pivot(index='COBJID', columns='column_name', values='CNDATE').reset_index()
        pivoted.columns.name = None
        pivoted = pivoted[['COBJID'] + sorted([col for col in pivoted.columns if col != 'COBJID'], key=lambda x: int(x[3:]))]

        # MERGE BREAKS INTO PIPE TABLE
        df_merged = df_line_for_preprocessing.merge(pivoted, left_on='LSID', right_on='COBJID', how='left').drop(columns=['COBJID'])

        # CREATE OLD PIPE ENTRIES
        df_old = df_merged[df_merged['OLDDIM'].notna()].copy()
        df_old['LSID'] = df_old['LSID'].astype(str) + '_old'
        df_old['IS_OLD'] = True
        df_old['MATERIAL'] = df_old['OLDMATERIAL']
        df_old['DIM'] = df_old['OLDDIM']
        df_old['YEAR'] = df_old['OLDYEAR']
        df_old.drop(columns=['OLDMATERIAL', 'OLDDIM', 'OLDYEAR'], inplace=True)

        # PREPARE CURRENT PIPE DATA
        df_merged['IS_OLD'] = False
        df_merged.drop(columns=['OLDMATERIAL', 'OLDDIM', 'OLDYEAR'], inplace=True)

        # COMBINE OLD AND CURRENT PIPES
        df_final = pd.concat([df_merged, df_old], ignore_index=True)
        df_final['LSID'] = df_final['LSID'].astype(str)
        df_final.sort_values(by='LSID', inplace=True)
        df_final.reset_index(drop=True, inplace=True)

        # FUNCTION TO FILTER CLOSE BREAKS
        def keep_first_xh_breaks(row, dbr_columns):
            break_dates = sorted([row[col] for col in dbr_columns if pd.notna(row[col])])
            to_keep, removed = [], []
            for bd in break_dates:
                if all(abs((bd - kept).total_seconds()) > 86400 for kept in to_keep):
                    to_keep.append(bd)
                else:
                    removed.append(bd)
            return to_keep, removed

        # IDENTIFY OLD PIPE IDS AND ASSIGN DBRs TO OLD/CURRENT PIPES
        pipe_ids_with_old = df_final[df_final['IS_OLD']]['LSID'].str.replace('_old', '')
        dbr_columns = sorted([col for col in df_final.columns if col.startswith('DBR')], key=lambda x: int(x[3:]))

        for pipe_id in pipe_ids_with_old:
            current_row = df_final[(df_final['LSID'] == pipe_id) & (~df_final['IS_OLD'])]
            old_row = df_final[(df_final['LSID'] == pipe_id + '_old') & (df_final['IS_OLD'])]
            if current_row.empty or old_row.empty:
                continue

            current_idx = current_row.index[0]
            old_idx = old_row.index[0]
            current_year = df_final.at[current_idx, 'YEAR']

            new_current_dbrs = []
            new_old_dbrs = []

            for col in dbr_columns:
                break_date = df_final.at[current_idx, col]
                if pd.notna(break_date):
                    if break_date.year > current_year:
                        new_current_dbrs.append(break_date)
                    else:
                        new_old_dbrs.append(break_date)

            # Reset DBRs
            for col in dbr_columns:
                df_final.at[current_idx, col] = pd.NaT
                df_final.at[old_idx, col] = pd.NaT

            # Assign DBRs in order
            for i, bd in enumerate(new_current_dbrs):
                df_final.at[current_idx, f'DBR{i+1}'] = bd
            for i, bd in enumerate(new_old_dbrs):
                df_final.at[old_idx, f'DBR{i+1}'] = bd

        # REMOVE BREAKS < 24H APART
        removed_xh = []
        df_final['HAS_REDUNDANT_BREAKS'] = False  # <- NEW FLAG
        for idx, row in df_final.iterrows():
            keep, remove = keep_first_xh_breaks(row, dbr_columns)
            if remove:
                removed_row = row.copy()
                removed_row['REMOVED_BREAKS'] = remove
                removed_xh.append(removed_row)
                df_final.at[idx, 'HAS_REDUNDANT_BREAKS'] = True  # <- FLAG SET

            # Reset and reassign cleaned breaks
            for col in dbr_columns:
                df_final.at[idx, col] = pd.NaT
            for i, bd in enumerate(sorted(keep)):
                df_final.at[idx, f'DBR{i+1}'] = bd

        removed_xh_df = pd.DataFrame(removed_xh)
        removed_xh_df = removed_xh_df.drop(['HAS_REDUNDANT_BREAKS'], axis=1)

        # ALERT & INCONSISTENCY FLAGS
        df_final['YEAR_PRESENT'] = df_final['YEAR'].notna()
        df_final['ALERT_YEAR'] = False
        df_final['INCONSISTENT_BREAKS'] = False

        pipe_ids_with_old = df_final[df_final['IS_OLD']]['LSID'].str.replace('_old', '')
        mask_current = (df_final['IS_OLD'] == False) & (df_final['LSID'].isin(pipe_ids_with_old)) & (df_final['YEAR_PRESENT'])

        # ALERT_YEAR: if current pipe's year <= old pipe's year
        for idx, row in df_final[mask_current].iterrows():
            old_row = df_final[(df_final['LSID'] == row['LSID'] + '_old') & (df_final['IS_OLD'])]
            if not old_row.empty:
                old_year = old_row.iloc[0]['YEAR']
                if pd.notna(old_year) and row['YEAR'] <= old_year:
                    df_final.at[idx, 'ALERT_YEAR'] = True

        # INCONSISTENT_BREAKS: break year earlier than pipe year
        for col in dbr_columns:
            df_final['INCONSISTENT_BREAKS'] |= df_final.apply(
                lambda row: pd.notna(row[col]) and pd.notna(row['YEAR']) and row[col].year < row['YEAR'], axis=1)

        # SUMMARY
        num_alert_year = df_final['ALERT_YEAR'].sum()
        num_inconsistent_dbrs = df_final['INCONSISTENT_BREAKS'].sum()
        num_redundant_breaks = df_final['HAS_REDUNDANT_BREAKS'].sum()

        st.subheader("Summary")
        st.write(f"Pipes with ALERT_YEAR flag: {num_alert_year}")
        st.write(f"Pipes with INCONSISTENT_BREAKS flag: {num_inconsistent_dbrs}")
        st.write(f"Pipes with removed redundant breaks (<=24h apart): {num_redundant_breaks}")

        # SIDEBAR FILTERS
        # st.sidebar.header("Filters")
        # show_alerts = st.sidebar.checkbox("Show only ALERT_YEAR pipes")
        # show_inconsistent = st.sidebar.checkbox("Show only INCONSISTENT_BREAKS pipes")
        # show_redundant = st.sidebar.checkbox("Show only HAS_REDUNDANT_BREAKS pipes")

        # filtered_df = df_final.copy()
        # if show_alerts:
        #     filtered_df = filtered_df[filtered_df['ALERT_YEAR']]
        # if show_inconsistent:
        #     filtered_df = filtered_df[filtered_df['INCONSISTENT_BREAKS']]
        # if show_redundant:
        #     filtered_df = filtered_df[filtered_df['HAS_REDUNDANT_BREAKS']]

        # st.subheader("Filtered Data")
        # st.dataframe(filtered_df)

        # Save the final preprocessed dataframe in session state
        st.session_state.df_line_preprocessed = df_final
        st.success("✅ Tables successfully preprocessed in session state. You can now move to the AI model page.")

        # EXPORT TO EXCEL
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, sheet_name='CleanedData', index=False)
            df_final[df_final['ALERT_YEAR']].to_excel(writer, sheet_name='AlertYear', index=False)
            df_final[df_final['INCONSISTENT_BREAKS']].to_excel(writer, sheet_name='InconsistentBreaks', index=False)
            df_final[df_final['HAS_REDUNDANT_BREAKS']].to_excel(writer, sheet_name='RedundantBreaks_cleaned', index=False)
            removed_xh_df.to_excel(writer, sheet_name='Removed_redundant_original', index=False)

        excel_buffer.seek(0)
        st.download_button(
            label="Download Processed Data",
            data=excel_buffer,
            file_name="processed_VA_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ---------------- VISUALIZATIONS ----------------
        
        # --- Tabs for Visualization ---
        st.markdown("## Breaks Analysis")
                
                # Tabs layout
        #tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Breaks by Decade", "Breaks by Material", "Breaks by Material and Diameter", "Breaks per km by Material", "Pipe Count by Material and Decade", "Pipe Length by Material and Decade", "Expected Failures Over Time"])


# Assuming df_final and dbr_columns are already defined and loaded

        # Tabs layout
        tab_labels = [
            "Breaks by Decade", "Breaks by Material", "Breaks by Material and Diameter",
            "Breaks per km by Material", "Pipe Count by Material and Decade",
            "Pipe Length by Material and Decade",
        ]
        tabs = st.tabs(tab_labels)
        # Ensure DECADE and LENGTH_KM columns exist
        df_final['YEAR'] = pd.to_numeric(df_final['YEAR'], errors='coerce')
        df_final = df_final.dropna(subset=['YEAR'])
        df_final['YEAR'] = df_final['YEAR'].astype(int)
        df_final['DECADE'] = (df_final['YEAR'] // 10) * 10
        df_final['LENGTH_KM'] = df_final['LENGTH'] / 1000

        # --- Tab 0: Breaks by Decade ---
        with tabs[0]:
            st.markdown("### Number of Breaks by Installation Decade")
            decade_breaks = df_final.melt(
                id_vars=['LSID', 'DECADE'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            decade_counts = decade_breaks['DECADE'].value_counts().reset_index()
            decade_counts.columns = ['DECADE', 'Breaks']
            decade_counts = decade_counts.sort_values('DECADE')

            fig1 = px.bar(decade_counts, x='DECADE', y='Breaks',
                        title="Number of Breaks by Installation Decade",
                        labels={"DECADE": "Installation Decade", "Breaks": "Number of Breaks"})
            st.plotly_chart(fig1, use_container_width=True)

        # --- Tab 1: Breaks by Material ---
        with tabs[1]:
            st.markdown("### Number of Breaks by Material")
            material_breaks = df_final.melt(
                id_vars=['LSID', 'MATERIAL'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            material_counts = material_breaks['MATERIAL'].value_counts().reset_index()
            material_counts.columns = ['MATERIAL', 'Breaks']

            fig2 = px.bar(material_counts, x='MATERIAL', y='Breaks',
                        title="Number of Breaks by Material")
            st.plotly_chart(fig2, use_container_width=True)

        # --- Tab 2: Breaks by Material and Diameter (Bubble Chart) ---
        with tabs[2]:
            st.markdown("### Breaks by Material and Diameter")
            bubble_data = df_final.melt(
                id_vars=['LSID', 'MATERIAL', 'DIM'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            agg_bubble = bubble_data.groupby(['MATERIAL', 'DIM']).size().reset_index(name='COUNT')
            agg_bubble = agg_bubble[agg_bubble['DIM'].notna()]

            fig3 = px.scatter(
                agg_bubble, x='DIM', y='MATERIAL', size='COUNT', color='COUNT',
                color_continuous_scale='YlOrRd', title="Breaks by Material and Diameter",
                labels={'DIM': 'Diameter', 'MATERIAL': 'Material', 'COUNT': 'Break Count'}
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- Tab 3: Breaks per km by Material ---
        with tabs[3]:
            st.markdown("### Number of Breaks per km by Material")
            material_breaks = df_final.melt(
                id_vars=['LSID', 'MATERIAL', 'LENGTH'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            total_lengths = df_final.groupby('MATERIAL')['LENGTH'].sum() / 1000
            break_counts = material_breaks.groupby('MATERIAL').size()
            breaks_per_km = (break_counts / total_lengths).reset_index(name='BREAKS_PER_KM')

            fig4 = px.bar(breaks_per_km, x='MATERIAL', y='BREAKS_PER_KM',
                        title="Breaks per km by Material",
                        labels={"BREAKS_PER_KM": "Breaks per km"})
            st.plotly_chart(fig4, use_container_width=True)

        # --- Tab 4: Pipe Count by Material and Decade ---
        with tabs[4]:
            st.markdown("### Number of Pipes by Material and Installation Decade")
            pipes_per_decade = df_final.groupby(['DECADE', 'MATERIAL']).size().reset_index(name='COUNT')

            fig5 = px.bar(
                pipes_per_decade, x='DECADE', y='COUNT', color='MATERIAL',
                title="Pipe Count by Material and Decade",
                labels={"COUNT": "Number of Pipes"},
                barmode='group'
            )
            st.plotly_chart(fig5, use_container_width=True)

        # --- Tab 5: Pipe Length by Material and Decade ---
        with tabs[5]:
            st.markdown("### Kilometers of Pipe by Material and Installation Decade")
            length_per_decade = df_final.groupby(['DECADE', 'MATERIAL'])['LENGTH_KM'].sum().reset_index()

            fig6 = px.bar(
                length_per_decade, x='DECADE', y='LENGTH_KM', color='MATERIAL',
                title="Total Pipe Length by Material and Decade (km)",
                labels={"LENGTH_KM": "Length (km)"},
                barmode='group'
            )
            st.plotly_chart(fig6, use_container_width=True)




# Optional enhancements (filters, export, etc.) can be implemented next

        


    # If Access data NOT found but Excel data loaded
    elif 'df_excel' in st.session_state:
        if 'df_line_preprocessed' in st.session_state:
         del st.session_state.df_line_preprocessed          
        st.info("Excel data detected. Plotting basic summary chart...")

        df_excel = st.session_state.df_excel.copy()
        # Extract DBR columns
        dbr_columns = [col for col in df_excel.columns if col.startswith('DBR') and col[3:].isdigit()]

        df_final = df_excel
        # Simple example plot: count pipes by MATERIAL
        # ---------------- VISUALIZATIONS ----------------
        
        # --- Tabs for Visualization ---
        st.markdown("## Breaks Analysis")
                
        # Tabs layout
        tab_labels = [
            "Breaks by Decade", "Breaks by Material", "Breaks by Material and Diameter",
            "Breaks per km by Material", "Pipe Count by Material and Decade",
            "Pipe Length by Material and Decade",
        ]
        # Ensure DECADE and LENGTH_KM columns exist
        df_final['YEAR'] = pd.to_numeric(df_final['YEAR'], errors='coerce')
        df_final = df_final.dropna(subset=['YEAR'])
        df_final['YEAR'] = df_final['YEAR'].astype(int)
        df_final['DECADE'] = (df_final['YEAR'] // 10) * 10
        df_final['LENGTH_KM'] = df_final['LENGTH'] / 1000
        tabs = st.tabs(tab_labels)

        # --- Tab 0: Breaks by Decade ---
        with tabs[0]:
            st.markdown("### Number of Breaks by Installation Decade")
            decade_breaks = df_final.melt(
                id_vars=['LSID', 'DECADE'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            decade_counts = decade_breaks['DECADE'].value_counts().reset_index()
            decade_counts.columns = ['DECADE', 'Breaks']
            decade_counts = decade_counts.sort_values('DECADE')

            fig1 = px.bar(decade_counts, x='DECADE', y='Breaks',
                        title="Number of Breaks by Installation Decade",
                        labels={"DECADE": "Installation Decade", "Breaks": "Number of Breaks"})
            st.plotly_chart(fig1, use_container_width=True)

        # --- Tab 1: Breaks by Material ---
        with tabs[1]:
            st.markdown("### Number of Breaks by Material")
            material_breaks = df_final.melt(
                id_vars=['LSID', 'MATERIAL'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            material_counts = material_breaks['MATERIAL'].value_counts().reset_index()
            material_counts.columns = ['MATERIAL', 'Breaks']

            fig2 = px.bar(material_counts, x='MATERIAL', y='Breaks',
                        title="Number of Breaks by Material")
            st.plotly_chart(fig2, use_container_width=True)

        # --- Tab 2: Breaks by Material and Diameter (Bubble Chart) ---
        with tabs[2]:
            st.markdown("### Breaks by Material and Diameter")
            bubble_data = df_final.melt(
                id_vars=['LSID', 'MATERIAL', 'DIM'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            agg_bubble = bubble_data.groupby(['MATERIAL', 'DIM']).size().reset_index(name='COUNT')
            agg_bubble = agg_bubble[agg_bubble['DIM'].notna()]

            fig3 = px.scatter(
                agg_bubble, x='DIM', y='MATERIAL', size='COUNT', color='COUNT',
                color_continuous_scale='YlOrRd', title="Breaks by Material and Diameter",
                labels={'DIM': 'Diameter', 'MATERIAL': 'Material', 'COUNT': 'Break Count'}
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- Tab 3: Breaks per km by Material ---
        with tabs[3]:
            st.markdown("### Number of Breaks per km by Material")
            material_breaks = df_final.melt(
                id_vars=['LSID', 'MATERIAL', 'LENGTH'],
                value_vars=dbr_columns,
                var_name='BREAK',
                value_name='BREAK_DATE'
            ).dropna(subset=['BREAK_DATE'])

            total_lengths = df_final.groupby('MATERIAL')['LENGTH'].sum() / 1000
            break_counts = material_breaks.groupby('MATERIAL').size()
            breaks_per_km = (break_counts / total_lengths).reset_index(name='BREAKS_PER_KM')

            fig4 = px.bar(breaks_per_km, x='MATERIAL', y='BREAKS_PER_KM',
                        title="Breaks per km by Material",
                        labels={"BREAKS_PER_KM": "Breaks per km"})
            st.plotly_chart(fig4, use_container_width=True)

        # --- Tab 4: Pipe Count by Material and Decade ---
        with tabs[4]:
            st.markdown("### Number of Pipes by Material and Installation Decade")
            pipes_per_decade = df_final.groupby(['DECADE', 'MATERIAL']).size().reset_index(name='COUNT')

            fig5 = px.bar(
                pipes_per_decade, x='DECADE', y='COUNT', color='MATERIAL',
                title="Pipe Count by Material and Decade",
                labels={"COUNT": "Number of Pipes"},
                barmode='group'
            )
            st.plotly_chart(fig5, use_container_width=True)

        # --- Tab 5: Pipe Length by Material and Decade ---
        with tabs[5]:
            st.markdown("### Kilometers of Pipe by Material and Installation Decade")
            length_per_decade = df_final.groupby(['DECADE', 'MATERIAL'])['LENGTH_KM'].sum().reset_index()

            fig6 = px.bar(
                length_per_decade, x='DECADE', y='LENGTH_KM', color='MATERIAL',
                title="Total Pipe Length by Material and Decade (km)",
                labels={"LENGTH_KM": "Length (km)"},
                barmode='group'
            )
            st.plotly_chart(fig6, use_container_width=True)


    else:
        st.error("❌ No valid data found. Please upload your Access or Excel file first.")
else:
    st.info("Press the button above to run preprocessing (only for Access file) and plot (for both Access and Excel files).")
