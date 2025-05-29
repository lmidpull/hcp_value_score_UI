import pandas as pd
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import seaborn as sns
import plotly.express as px

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def load_excel_file(uploaded_file):
    """Cache Excel file loading to avoid re-reading on every interaction"""
    return pd.read_excel(uploaded_file)

@st.cache_data
def load_mapping_file():
    """Cache mapping file loading"""
    return pd.read_excel("phases_mapping.xlsx")

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def calculate_hcp_value_score_vectorized(df, metrics, weights):
    """Vectorized calculation of HCP value scores"""
    list1 = []
    
    # Process all metrics in one go using vectorized operations
    for i, metric in enumerate(metrics):
        # Log normalization
        df[f"{metric} log norm"] = np.log2(1 + df[metric])
        
        # Min-max normalization
        log_col = f"{metric} log norm"
        min_val = df[log_col].min()
        max_val = df[log_col].max()
        df[f"{metric} min-max norm"] = (df[log_col] - min_val) / (max_val - min_val)
        
        # Apply weight
        weight_col = f"{metric} calculation with weight"
        df[weight_col] = df[f"{metric} min-max norm"] * weights[i]
        list1.append(weight_col)
    
    return df, list1

def calculate_phases_vectorized(df, segment_label, competitive_flag, competitive_prescriber_score, 
                              referral_flag, segment_score, forecast):
    """Vectorized phase calculations"""
    
    # High Value Category calculation
    conditions = [
        df[segment_label] == 'CHURNED',
        df[segment_label].isin(['CTL_NON_FIRST_TIME', 'CTL_MAYBE_FIRST_TIME', 'FIRST_TIME']),
        df[segment_label].isin(['DECLINING', 'NEUTRAL', 'GROWING'])
    ]
    values = ['CHURNED', 'Non-Prescriber', 'Prescriber']
    df['High Value Category'] = np.select(conditions, values, default='default')

    # NrX Prob Tiers calculation
    conditions2 = [
        df['High Value Category'] == 'Non-Prescriber',
        df[segment_score] <= 0.5,
        (df[segment_score] > 0.5) & (df[segment_score] <= 0.8),
        df[segment_score] > 0.8
    ]
    values2 = ['No NrX', 'Low NrX', 'Med NrX', 'High NrX']
    df['NrX Prob Tiers'] = np.select(conditions2, values2, default='default')

    # Competitive Prescriber Segment calculation
    conditions3 = [
        df[competitive_flag] == 'N',
        df[competitive_prescriber_score] <= 0.3,
        (df[competitive_prescriber_score] > 0.3) & (df[competitive_prescriber_score] <= 0.5),
        df[competitive_prescriber_score] > 0.5
    ]
    values3 = ['Non-Comp-Prescbr', 'Low-Comp-Prescbr', 'Med-Comp-Prescbr', 'High-Comp-Prescbr']
    df['Competitive Prescriber Segment'] = np.select(conditions3, values3, default='default')

    # Referring HCP calculation
    conditions4 = [
        df[referral_flag].isnull(),
        df[referral_flag] == 'Y',
        df[referral_flag] == 'N'
    ]
    values4 = ['No Data', 'Referring', 'Non Referring']
    df['Referring HCP'] = np.select(conditions4, values4, default='default')
    
    # Forecast calculations - vectorized normal distribution
    forecast_norm_col = f"{forecast} min-max norm"
    mean_val = df[forecast_norm_col].mean()
    std_val = df[forecast_norm_col].std()
    
    df['forecast Norm used in phase calculations'] = (
        (np.pi * std_val) * np.exp(-0.5 * ((df[forecast_norm_col] - mean_val) / std_val) ** 2)
    )
    
    conditions5 = [
        df['High Value Category'] == 'Non-Prescriber',
        df['forecast Norm used in phase calculations'] <= 0.75,
        (df['forecast Norm used in phase calculations'] > 0.75) & (df['forecast Norm used in phase calculations'] <= 0.95),
        df['forecast Norm used in phase calculations'] > 0.95
    ]
    values5 = ['No TrX', 'Low TrX', 'Med TrX', 'High TrX']
    df['TrX Forecast Label'] = np.select(conditions5, values5, default='default')
    
    return df

def process_budget_calculations_vectorized(df, client_segment, npi, campaign_budget):
    """Vectorized budget calculations"""
    # Use more efficient groupby operations
    df_grouped = df.groupby(client_segment).agg({
        npi: 'count',
        'norm_score': 'mean'
    }).reset_index()
    
    df_grouped.columns = [client_segment, 'Count of NPIs', 'Average Hcp Value Score']
    
    # Vectorized budget calculations
    df_grouped['Score Dist'] = df_grouped['Average Hcp Value Score'] / df_grouped['Average Hcp Value Score'].sum()
    df_grouped['Score Dist*Count'] = df_grouped['Score Dist'] * df_grouped['Count of NPIs']
    df_grouped['% Budget Allocation'] = df_grouped['Score Dist*Count'] / df_grouped['Score Dist*Count'].sum()
    df_grouped['Budget Per Segment'] = df_grouped['% Budget Allocation'] * campaign_budget
    df_grouped['Average Budget Per HCP'] = df_grouped['Budget Per Segment'] / df_grouped['Count of NPIs']
    
    return df_grouped

st.title("HCP Value Score & M1 Budget Calculator")
uploaded_file = st.file_uploader("Upload the file: ", type=['xlsx', 'xls'])

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def submitted():
    st.session_state.submitted = True

st.button('Submit', on_click=submitted, key=1)

if st.session_state.submitted and uploaded_file is not None:
    phases = st.checkbox("Calculations with M1 Phases")
    no_phases = st.checkbox("Calculations without M1 Phases")
    
    if phases:
        st.write("Calculations with M1 Phases")
        
        # Load and optimize data
        with st.spinner("Loading data..."):
            df = load_excel_file(uploaded_file)
            df = optimize_dataframe_memory(df)
        
        st.success(f"Loaded {len(df):,} rows")

        # Column selection
        options = st.multiselect(
            "Select all relevant columns (make sure to include NPI Number, Client Segment, Segment_Label, Referral Flag, Competitive Prescriber Flag, Competitive Prescriber Score, Segment Score, and Forecast 3 months): ",
            df.columns
        )
        
        if options:
            st.write("You selected:", options)
            df = df[options].copy()  # More efficient than dropping columns
            st.dataframe(df.head(10))

            # Metric selection
            metrics = st.multiselect(
                "Select Metrics To Use For HCP Value Score Calculations (only measurable fields ie Segment Score, Forecast 3 months etc..): ",
                options
            )
            
            if metrics:
                # Column mappings
                segment_label = st.selectbox("Choose Segment Label Column", df.columns, index=None, key=10000)
                competitive_flag = st.selectbox("Choose Competitive Pres Flag Column", df.columns, index=None, key=10001)
                competitive_prescriber_score = st.selectbox("Choose Competitive Pres Score Column", df.columns, index=None, key=10002)
                referral_flag = st.selectbox("Choose Referral Flag Column", df.columns, index=None, key=10003)
                segment_score = st.selectbox("Choose Segment Score Column", df.columns, index=None, key=10004)
                forecast = st.selectbox("Choose Forecast 3 Months Mean Column", df.columns, index=None, key=10005)
                
                # Collect all weights first
                weights = []
                for metric in metrics:
                    weight = st.number_input(f"Enter the weight you want to use for: {metric}", min_value=0.0, step=0.1)
                    weights.append(weight)
                
                if all(col is not None for col in [segment_label, competitive_flag, competitive_prescriber_score, referral_flag, segment_score, forecast]):
                    with st.spinner("Calculating HCP Value Scores..."):
                        # Vectorized HCP value score calculations
                        df, list1 = calculate_hcp_value_score_vectorized(df, metrics, weights)
                        
                        # Calculate final scores
                        df['sum of metrics'] = df[list1].sum(axis=1)
                        df['log_score'] = np.log2(1 + df['sum of metrics'])
                        df["norm_score"] = (df['log_score'] - df['log_score'].min()) / (df['log_score'].max() - df['log_score'].min())
                    
                    with st.spinner("Calculating M1 Phases..."):
                        # Vectorized phase calculations
                        df = calculate_phases_vectorized(df, segment_label, competitive_flag, 
                                                       competitive_prescriber_score, referral_flag, 
                                                       segment_score, forecast)
                        
                        # Fix specific values
                        df.loc[df['NrX Prob Tiers'] == '0', 'NrX Prob Tiers'] = 'No NrX'
                        df.loc[df['Referring HCP'] == '0', 'Referring HCP'] = 'No Data'
                        
                        # Create lookup string more efficiently
                        cols = [segment_label, 'NrX Prob Tiers', 'TrX Forecast Label', 'Competitive Prescriber Segment', 'Referring HCP']
                        df['Lookup String'] = df[cols].apply(lambda row: '| '.join(row.values.astype(str)), axis=1)
                        
                        # Load and merge mapping data
                        df_mapping = load_mapping_file()
                        df_mapping['String for Vlookup'] = df_mapping['String for Vlookup'].str.strip()
                        df['Lookup String'] = df['Lookup String'].str.strip()
                        df = pd.merge(df, df_mapping, left_on='Lookup String', right_on='String for Vlookup', how='left')
                    
                    st.subheader("HCP Value Score Raw Data", divider=True)
                    st.dataframe(df.head(10))
                    csv = convert_df(df)
                    st.download_button(
                        label="Download HCP Value Score Raw Data",
                        data=csv,
                        file_name="hcp_value_score_data.csv",
                        mime="text/csv"
                    )

                    # Phase analysis
                    phase = st.selectbox("Choose column name with phase information: ", df.columns, index=None, key=30000)
                    npi = st.selectbox("Choose column name with NPI Number: ", df.columns, index=None, key=30001)
                    client_segment = st.text_input("Enter Client Segment Column: ")
                    
                    if phase and npi and client_segment:
                        with st.spinner("Generating phase analysis..."):
                            # More efficient groupby operations
                            df_count = df.groupby([client_segment, phase])[npi].count()
                            st.dataframe(df_count)

                            df_count_2 = df.groupby([phase])[npi].count().reset_index()
                            total = df_count_2[npi].sum()
                            df_count_2['% Breakdown'] = (df_count_2[npi] / total) * 100
                            st.dataframe(df_count_2)

                            fig = px.pie(df_count_2, values=npi, names=phase, title="Total NPIs by Phase")
                            st.plotly_chart(fig, theme=None)
                        
                        # Budget calculations
                        campaign_budget = st.number_input("Enter Campaign Budget: ", min_value=0.0)
                        
                        if campaign_budget > 0:
                            with st.spinner("Calculating budget allocations..."):
                                df3 = process_budget_calculations_vectorized(df, client_segment, npi, campaign_budget)
                                
                                st.subheader("Segment Level Budget Allocation", divider=True)
                                st.dataframe(df3)
                                csv2 = convert_df(df3)
                                st.download_button(
                                    label="Download Segment Level Budget Allocation",
                                    data=csv2,
                                    file_name="segment_budget_allocation.csv",
                                    mime="text/csv"
                                )

                                # Gradient graph (optimized)
                                def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
                                    new_cmap = colors.LinearSegmentedColormap.from_list(
                                        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
                                        cmap(np.linspace(min_val, max_val, n))
                                    )
                                    return new_cmap

                                x = df3[client_segment]
                                y2 = df3['Average Hcp Value Score']
                                y = df3['Count of NPIs']
                                
                                fig, ax = plt.subplots(figsize=(15, 10))  # Smaller default size
                                bars = ax.bar(x, y)
                                y_min, y_max = ax.get_ylim()
                                y_min2 = 0
                                y_max2 = max(y2)
                                grad = np.atleast_2d(np.linspace(0, 1, 256)).T
                                
                                for i, bar in enumerate(bars):
                                    bar.set_zorder(1)
                                    bar.set_facecolor("none")
                                    x_pos, _ = bar.get_xy()
                                    w, h = bar.get_width(), bar.get_height()
                                    h2 = y2.iloc[i]
                                    c_map = truncate_colormap(plt.cm.Blues, min_val=0,
                                                            max_val=(h2 - y_min2) / (y_max2 - y_min2))
                                    ax.imshow(grad, extent=[x_pos, x_pos+w, h, y_min], aspect="auto", zorder=0, cmap=c_map)
                                
                                ax.spines.top.set_visible(False)
                                ax.spines.right.set_visible(False)
                                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                                plt.xticks(size=12)
                                plt.yticks(size=12)
                                
                                st.subheader("Gradient Graph", divider=True)
                                st.pyplot(fig)

                                # HCP Level Budget calculation (optimized)
                                st.subheader("HCP Level Budget", divider=True)
                                
                                # Collect all segment budgets first
                                segment_budgets = {}
                                key_count = 100
                                for segment in df[client_segment].unique():
                                    budget = st.number_input(f"Enter the segment level budget for: {segment}", 
                                                           min_value=0.0, key=key_count)
                                    segment_budgets[segment] = budget
                                    key_count += 1
                                
                                # Vectorized budget calculation per HCP
                                df['Score Distribution'] = 0.0
                                df['Budget'] = 0.0
                                
                                for segment, budget in segment_budgets.items():
                                    if budget > 0:
                                        mask = df[client_segment] == segment
                                        total_score = df.loc[mask, 'norm_score'].sum()
                                        if total_score > 0:
                                            df.loc[mask, "Score Distribution"] = df.loc[mask, 'norm_score'] / total_score
                                            df.loc[mask, "Budget"] = df.loc[mask, 'Score Distribution'] * budget
                                
                                st.dataframe(df.head(10))
                                csv_final = convert_df(df)
                                st.download_button(
                                    label="Download HCP Level Budget",
                                    data=csv_final,
                                    file_name="hcp_level_budget.csv",
                                    mime="text/csv"
                                )

    if no_phases:
        st.write("Regular Calculations")
        
        with st.spinner("Loading data..."):
            df = load_excel_file(uploaded_file)
            df = optimize_dataframe_memory(df)
        
        st.success(f"Loaded {len(df):,} rows")
        
        # Column selection
        options = st.multiselect(
            "Select all relevant columns (make sure to include NPI Number and Client segment (if required)): ",
            df.columns
        )
        
        if options:
            st.write("You selected:", options)
            df = df[options].copy()
            st.dataframe(df.head(10))
            
            metrics = st.multiselect(
                "Select Metrics To Use For HCP Value Score Calculations (only measurable fields ie Segment Score, Forecast 3 months etc..): ",
                options
            )
            
            if metrics:
                # Collect all weights first
                weights = []
                for metric in metrics:
                    weight = st.number_input(f"Enter the weight you want to use for: {metric}", min_value=0.0, step=0.1)
                    weights.append(weight)
                
                with st.spinner("Calculating HCP Value Scores..."):
                    # Vectorized HCP value score calculations
                    df, list1 = calculate_hcp_value_score_vectorized(df, metrics, weights)
                    
                    df['sum of metrics'] = df[list1].sum(axis=1)
                    df['log_score'] = np.log2(1 + df['sum of metrics'])
                    df["norm_score"] = (df['log_score'] - df['log_score'].min()) / (df['log_score'].max() - df['log_score'].min())
                
                st.subheader("HCP Value Score Raw Data", divider=True)
                st.dataframe(df.head(10))
                csv = convert_df(df)
                st.download_button(
                    label="Download HCP Value Score Raw Data",
                    data=csv,
                    file_name="hcp_value_score_data.csv",
                    mime="text/csv"
                )
                
                # Budget calculations
                client_segment = st.text_input("Enter Client Segment Column: ")
                npi_number = st.text_input("Enter NPI Column: ")
                
                if client_segment and npi_number and client_segment in df.columns and npi_number in df.columns:
                    campaign_budget = st.number_input("Enter Campaign Budget: ", min_value=0.0)
                    
                    if campaign_budget > 0:
                        with st.spinner("Calculating budget allocations..."):
                            df3 = process_budget_calculations_vectorized(df, client_segment, npi_number, campaign_budget)
                            
                            st.subheader("Segment Level Budget Allocation", divider=True)
                            st.dataframe(df3)
                            csv2 = convert_df(df3)
                            st.download_button(
                                label="Download Segment Level Budget Allocation",
                                data=csv2,
                                file_name="segment_budget_allocation.csv",
                                mime="text/csv"
                            )
                            
                            # Gradient graph and HCP level budget calculations
                            # (Same optimized code as in phases section)
                            def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
                                new_cmap = colors.LinearSegmentedColormap.from_list(
                                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
                                    cmap(np.linspace(min_val, max_val, n))
                                )
                                return new_cmap

                            x = df3[client_segment]
                            y2 = df3['Average Hcp Value Score']
                            y = df3['Count of NPIs']
                            
                            fig, ax = plt.subplots(figsize=(15, 10))
                            bars = ax.bar(x, y)
                            y_min, y_max = ax.get_ylim()
                            y_min2 = 0
                            y_max2 = max(y2)
                            grad = np.atleast_2d(np.linspace(0, 1, 256)).T
                            
                            for i, bar in enumerate(bars):
                                bar.set_zorder(1)
                                bar.set_facecolor("none")
                                x_pos, _ = bar.get_xy()
                                w, h = bar.get_width(), bar.get_height()
                                h2 = y2.iloc[i]
                                c_map = truncate_colormap(plt.cm.Blues, min_val=0,
                                                        max_val=(h2 - y_min2) / (y_max2 - y_min2))
                                ax.imshow(grad, extent=[x_pos, x_pos+w, h, y_min], aspect="auto", zorder=0, cmap=c_map)
                            
                            ax.spines.top.set_visible(False)
                            ax.spines.right.set_visible(False)
                            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                            plt.xticks(size=12)
                            plt.yticks(size=12)
                            
                            st.subheader("Gradient Graph", divider=True)
                            st.pyplot(fig)

                            # HCP Level Budget
                            st.subheader("HCP Level Budget", divider=True)
                            
                            segment_budgets = {}
                            key_count = 100
                            for segment in df[client_segment].unique():
                                budget = st.number_input(f"Enter the segment level budget for: {segment}", 
                                                       min_value=0.0, key=key_count)
                                segment_budgets[segment] = budget
                                key_count += 1
                            
                            df['Score Distribution'] = 0.0
                            df['Budget'] = 0.0
                            
                            for segment, budget in segment_budgets.items():
                                if budget > 0:
                                    mask = df[client_segment] == segment
                                    total_score = df.loc[mask, 'norm_score'].sum()
                                    if total_score > 0:
                                        df.loc[mask, "Score Distribution"] = df.loc[mask, 'norm_score'] / total_score
                                        df.loc[mask, "Budget"] = df.loc[mask, 'Score Distribution'] * budget
                            
                            st.dataframe(df.head(10))
                            csv_final = convert_df(df)
                            st.download_button(
                                label="Download HCP Level Budget",
                                data=csv_final,
                                file_name="hcp_level_budget.csv",
                                mime="text/csv"
                            )
