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


st.title("HCP Value Score & M1 Budget Calculator")
uploaded_file = st.file_uploader("Upload the file: ", type=['xlsx', 'xls'])
#df = pd.read_excel(uploaded_file)

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def submitted():
    st.session_state.submitted = True

st.button('Submit', on_click = submitted, key=1)

if st.session_state.submitted:
    phases = st.checkbox("Calculations with M1 Phases")
    no_phases = st.checkbox("Calculations without M1 Phases")
    if phases:
        #Code below includes all parts related to M1 phase calculations
        
        st.write("Calculations with M1 Phases")
        df = pd.read_excel(uploaded_file)

        #column selection to allow for any file regardless of file name to work with the script
        options = st.multiselect("Select all relevant columns (make sure to include NPI Number, Client Segment, Segment_Label, Referral Flag, Competitive Prescriber Flag, Competitive Prescriber Score, Segment Score, and Forecast 3 months ): ",df.columns,)
        st.write("You selected:", options)
        df = df.drop(columns=[col for col in df if col not in options])
        st.dataframe(df.head(10))

        #metric and column selection to make sure correct columns are used for hcp value score and phase calculations
        metrics = st.multiselect("Select Metrics To Use For HCP Value Score Calculations (only measurable fields ie Segment Score, Forecast 3 months etc..): ",options,)
        list1 = []
        segment_label = st.selectbox("Choose Segment Label Column", df.columns, index=None,key=10000)
        competitive_flag =  st.selectbox("Choose Competitive Pres Flag Column", df.columns, index=None,key=10001)
        compeitive_prescriber_score = st.selectbox("Choose Competive Pres Score Column", df.columns, index=None,key=10002)
        referral_flag =  st.selectbox("Choose Referral Flag Column", df.columns, index=None,key=10003)
        segment_score =  st.selectbox("Choose Segment Score Column", df.columns, index=None,key=10004)
        forecast = st.selectbox("Choose Forecast 3 Months Mean Column", df.columns, index=None,key=10005)
        
        #hcp value score calculations begin here
        for x in metrics:
            df[x+" log norm"]=np.log2(1+(df[x]))
            df[x+" min-max norm"] = (df[x+" log norm"]-  df[x+" log norm"].min()) / ( df[x+" log norm"].max() -  df[x+" log norm"].min())
            weight = st.number_input("Enter the weight you want to use for: "+ x)
            df[x+ " calculation with weight"] = df[x+" min-max norm"] * float(weight)
            list1.append(x+ " calculation with weight")
        
        #m1 phase calculations begin here
        df['High Value Category'] = 'searching'
        df['High Value Category'] =  df['High Value Category'].astype('str')
        condition = [(df[segment_label]=='CHURNED'),(df[segment_label]=='CTL_NON_FIRST_TIME'), (df[segment_label]=='CTL_MAYBE_FIRST_TIME'), (df[segment_label]=='FIRST_TIME'), (df[segment_label]=='DECLINING'), (df[segment_label]=='NEUTRAL'), (df[segment_label]=='GROWING')]
        values = ['CHURNED', 'Non-Prescriber', 'Non-Prescriber', 'Non-Prescriber', 'Prescriber', 'Prescriber', 'Prescriber']
        df['High Value Category']=  np.select(condition, values, default=np.array(['default'])) 

        df['NrX Prob Tiers']='searching'
        df['NrX Prob Tiers']=df['NrX Prob Tiers'].astype('str')
        condition2 = [(df['High Value Category']!='Non-Prescriber'),(df[segment_score]<=0.5), (df[segment_score]>0.5) & (df[segment_score]<=0.8), (df[segment_score]>0.8)]
        values2 = ['No NrX', 'Low NrX', 'Med NrX', 'High NrX']
        df['NrX Prob Tiers']=np.select(condition2, values2, default=np.array(['default']))

        df['Competitive Prescriber Segment'] = 'searching'
        df['Competitive Prescriber Segment'] = df['Competitive Prescriber Segment'].astype('str')
        condition3 = [(df[competitive_flag]=='N'),(df[compeitive_prescriber_score]<=0.3), (df[compeitive_prescriber_score]>0.3) & (df[compeitive_prescriber_score]<=0.5), (df[compeitive_prescriber_score]>0.5)]
        values3 = ['Non-Comp-Prescbr', 'Low-Comp-Prescbr', 'Med-Comp-Prescbr', 'High-Comp-Prescbr']
        df['Competitive Prescriber Segment']=np.select(condition3, values3, default=np.array(['default']))

        df['Referring HCP'] = 'searching'
        df['Referring HCP'] =  df['Referring HCP'].astype('str')
        condition4 = [(df[referral_flag].isnull()),(df[referral_flag]=='Y'), (df[referral_flag]=='N')]
        values4 = ['No Data', 'Referring', 'Non Referring']                                           
        df['Referring HCP']=np.select(condition4, values4, default=np.array(['default']))
        
        def normal_dist(x, mean, sd):
            prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
            return prob_density                                            
        
        for x in df[forecast+" min-max norm"]:  
            df['forecast Norm used in phase calculations'] = normal_dist(df[forecast+" min-max norm"], df[forecast+" min-max norm"].mean(), df[forecast+" min-max norm"].std())
        condition5 = [(df['High Value Category']=='Non-Prescriber'),(df['forecast Norm used in phase calculations']<=0.75), (df['forecast Norm used in phase calculations']>0.75) & (df['forecast Norm used in phase calculations']<=0.95), (df['forecast Norm used in phase calculations']>0.95)]
        values5 = ['No TrX', 'Low TrX', 'Med TrX', 'High TrX']
        df['TrX Forecast Label']=np.select(condition5, values5, default=np.array(['default']))
        
        df['sum of metrics'] = df[list1].sum(axis=1)
        df['log_score'] = np.log2(1+df['sum of metrics'])
        df["norm_score"]=(df['log_score'] - df['log_score'].min()) / (df['log_score'].max() - df['log_score'].min())
        
        df.loc[df['NrX Prob Tiers'] == '0', 'NrX Prob Tiers'] = 'No NrX'
        df.loc[df['Referring HCP'] == '0', 'Referring HCP'] = 'No Data'
        
        cols = [segment_label, 'NrX Prob Tiers', 'TrX Forecast Label','Competitive Prescriber Segment','Referring HCP']
        df['Lookup String'] = df[cols].apply(lambda row: '| '.join(row.values.astype(str)), axis=1)
        
        df_mapping = pd.read_excel("phases_mapping.xlsx")
        
        df_mapping['String for Vlookup']= df_mapping['String for Vlookup'].str.strip()
        df['Lookup String'] = df['Lookup String'].str.strip()
        df=pd.merge(df, df_mapping, left_on = 'Lookup String', right_on = 'String for Vlookup', how = 'left')
      
        #hcp value score and m1 phase calculations end here
        
        st.subheader("HCP Value Score Raw Data", divider=True)
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
        label="Download HCP Value Score Raw Data",
        data=csv,
        file_name="large_df.csv",
        mime="text/csv",)

        #selection below so that grouped by table is pulling from the correct phase and npi column
        phase = st.selectbox("Choose column name with phase information: ", df.columns, index=None,key=30000)
        npi = st.selectbox("Choose column name with NPI Number: ", df.columns, index=None,key=30001)
        client_segment = st.text_input("Enter Client Segment Column: ")
        df_count=df.groupby([client_segment,phase])[npi].count()
        st.dataframe(df_count)

        df_count_2=df.groupby([phase])[npi].count()
        st.dataframe(df_count_2)

        fig = px.pie(df_count_2, values='Total NPIs', names='Phases', title="Total NPIs by Phase",)
        st.plotly_chart(fig, theme=None)
        
       #gradient graph calculations begin here
        def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
            cmap(np.linspace(min_val, max_val, n)))
            return new_cmap

    #Input the segment labels below
        #client_segment = st.text_input("Enter Client Segment Column: ")
        #npi_number = st.text_input("Enter NPI Column: ")

        #segment level budget allocation calculations begin here
        df1=df.groupby([client_segment])[npi].count()
        df2=df.groupby([client_segment])['norm_score'].mean()
        df3=pd.concat([df1,df2],axis=1).reset_index().rename(columns={npi:'Count of NPIs', 'norm_score': 'Average Hcp Value Score'})
        df3['Score Dist'] = df3['Average Hcp Value Score']/(df3['Average Hcp Value Score'].sum())
        df3['Score Dist*Count'] = df3['Score Dist']*df3['Count of NPIs']
        campaign_budget = st.number_input("Enter Campaign Budget: ")
        df3['% Budget Allocation'] = df3['Score Dist*Count']/(df3['Score Dist*Count'].sum())
        df3['Budget Per Segment'] = df3['% Budget Allocation']*campaign_budget
        df3['Average Budget Per HCP']= df3['Budget Per Segment']/df3['Count of NPIs']
        
        #segment level budget allocation calculations end here
        
        st.subheader("Segment Level Budget Allocation", divider=True)
        st.dataframe(df3)
        csv2 = convert_df(df3)
        st.download_button(
        label="Download Segment Level Budget Allocation",
        data=csv2,
        file_name="large_df.csv",
        mime="text/csv",)
        x = df3[client_segment]
    # y2 is for gradient
    #Input the hcp value scores cooresponding to the segment labels below
        y2 = df3['Average Hcp Value Score']
    # y is for y-axis

    #Input the count of hcps in the cooresponding segment labels below
        y = df3['Count of NPIs']
        fig, ax = plt.subplots()
    #bars = ax.bar(x, y, edgecolor = "black")
        bars = ax.bar(x, y)
        y_min, y_max = ax.get_ylim()
        y_min2 = 0
        y_max2 = max(y2)
        grad = np.atleast_2d(np.linspace(0, 1, 256)).T
        ax = bars[0].axes 
        lim = ax.get_xlim()+ax.get_ylim()
        x1=0
        for bar in bars:
            bar.set_zorder(1)  
            bar.set_facecolor("none")  
            x, _ = bar.get_xy()  
            w, h = bar.get_width(), bar.get_height() 
            h2 = y2[x1]
            c_map = truncate_colormap(plt.cm.Blues, min_val=0,
                                    max_val=(h2 - y_min2) / (y_max2 - y_min2))
        #c_map = truncate_colormap(plt.cm.summer_r, min_val=0,
        #                          max_val=(h2 - y_min2) / (y_max2 - y_min2))

            ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,
                cmap=c_map)
            x1=x1+1
        ax.axis(lim)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.xticks(size = 15)
        plt.yticks(size = 15)
    #fig.autofmt_xdate() 
        fig.set_size_inches(25, 15, forward=True)
        st.subheader("Gradient Graph", divider=True)
        st.pyplot(plt.gcf())
        st.subheader("HCP Level Budget", divider=True)
        key_count =100
        df['Score Distribution']=0
        df['Budget']=0
        for x in df[client_segment].unique():
            budget = st.number_input("Enter the segment level budget for: "+ x, key=key_count)
            total_score =  df.loc[df[client_segment] == x, 'norm_score'].sum()
            df.loc[df[client_segment] == x, "Score Distribution"] = df['norm_score']/total_score
            df.loc[df[client_segment] == x, "Budget"] =  df['Score Distribution']*budget
            key_count = key_count+1
        st.dataframe(df.head(10))
        csv_final = convert_df(df)
        st.download_button(
        label="Download HCP Level Budget",
        data=csv_final,
        file_name="large_df.csv",
        mime="text/csv",)
   #gradient graph code ends here
    
    if no_phases:
        #Normal hcp value score and m1 calculations WITHOUT phases begins here
        
        st.write("Regular Calculations")
        df = pd.read_excel(uploaded_file)
        
        #column selection to allow for any file regardless of file name to work with the script
        options = st.multiselect("Select all relevant columns (make sure to include NPI Number and Client segment (if required) ): ",df.columns,)
        st.write("You selected:", options)
        df = df.drop(columns=[col for col in df if col not in options])
        st.dataframe(df.head(10))
        metrics = st.multiselect("Select Metrics To Use For HCP Value Score Calculations (only measurable fields ie Segment Score, Forecast 3 months etc..): ",options,)
        list1 = []

        #hcp value score calculations begin here
        for x in metrics:
            df[x+" log norm"]=np.log2(1+(df[x]))
            df[x+" min-max norm"] = (df[x+" log norm"]-  df[x+" log norm"].min()) / ( df[x+" log norm"].max() -  df[x+" log norm"].min())
            weight = st.number_input("Enter the weight you want to use for: "+ x)
            df[x+ " calculation with weight"] = df[x+" min-max norm"] * float(weight)
            list1.append(x+ " calculation with weight")
        df['sum of metrics'] = df[list1].sum(axis=1)
        df['log_score'] = np.log2(1+df['sum of metrics'])
        df["norm_score"]=(df['log_score'] - df['log_score'].min()) / (df['log_score'].max() - df['log_score'].min())
        
        st.subheader("HCP Value Score Raw Data", divider=True)
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
        label="Download HCP Value Score Raw Data",
        data=csv,
        file_name="large_df.csv",
        mime="text/csv",)
        #hcp value score calculations end here
        
        #gradient graph code begins here
        def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
            cmap(np.linspace(min_val, max_val, n)))
            return new_cmap
            
        # Segment level budget allocation calculations begin here
    #Input the segment labels below
        client_segment = st.text_input("Enter Client Segment Column: ")
        npi_number = st.text_input("Enter NPI Column: ")
        df1=df.groupby([client_segment])[npi_number].count()
        df2=df.groupby([client_segment])['norm_score'].mean()
        df3=pd.concat([df1,df2],axis=1).reset_index().rename(columns={npi_number:'Count of NPIs', 'norm_score': 'Average Hcp Value Score'})
        df3['Score Dist'] = df3['Average Hcp Value Score']/(df3['Average Hcp Value Score'].sum())
        df3['Score Dist*Count'] = df3['Score Dist']*df3['Count of NPIs']
        campaign_budget = st.number_input("Enter Campaign Budget: ")
        df3['% Budget Allocation'] = df3['Score Dist*Count']/(df3['Score Dist*Count'].sum())
        df3['Budget Per Segment'] = df3['% Budget Allocation']*campaign_budget
        df3['Average Budget Per HCP']= df3['Budget Per Segment']/df3['Count of NPIs']

        #Segment level budget allocations end here
        st.subheader("Segment Level Budget Allocation", divider=True)
        st.dataframe(df3)
        csv2 = convert_df(df3)
        st.download_button(
        label="Download Segment Level Budget Allocation",
        data=csv2,
        file_name="large_df.csv",
        mime="text/csv",)
        x = df3[client_segment]
    # y2 is for gradient
    #Input the hcp value scores cooresponding to the segment labels below
        y2 = df3['Average Hcp Value Score']
    # y is for y-axis

    #Input the count of hcps in the cooresponding segment labels below
        y = df3['Count of NPIs']
        fig, ax = plt.subplots()
    #bars = ax.bar(x, y, edgecolor = "black")
        bars = ax.bar(x, y)
        y_min, y_max = ax.get_ylim()
        y_min2 = 0
        y_max2 = max(y2)
        grad = np.atleast_2d(np.linspace(0, 1, 256)).T
        ax = bars[0].axes 
        lim = ax.get_xlim()+ax.get_ylim()
        x1=0
        for bar in bars:
            bar.set_zorder(1)  
            bar.set_facecolor("none")  
            x, _ = bar.get_xy()  
            w, h = bar.get_width(), bar.get_height() 
            h2 = y2[x1]
            c_map = truncate_colormap(plt.cm.Blues, min_val=0,
                                    max_val=(h2 - y_min2) / (y_max2 - y_min2))
        #c_map = truncate_colormap(plt.cm.summer_r, min_val=0,
        #                          max_val=(h2 - y_min2) / (y_max2 - y_min2))

            ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,
                cmap=c_map)
            x1=x1+1
        ax.axis(lim)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.xticks(size = 15)
        plt.yticks(size = 15)
    #fig.autofmt_xdate() 
        fig.set_size_inches(25, 15, forward=True)
        st.subheader("Gradient Graph", divider=True)
        st.pyplot(plt.gcf())
        st.subheader("HCP Level Budget", divider=True)
        key_count =100
        df['Score Distribution']=0
        df['Budget']=0
        for x in df[client_segment].unique():
            budget = st.number_input("Enter the segment level budget for: "+ x, key=key_count)
            total_score =  df.loc[df[client_segment] == x, 'norm_score'].sum()
            df.loc[df[client_segment] == x, "Score Distribution"] = df['norm_score']/total_score
            df.loc[df[client_segment] == x, "Budget"] =  df['Score Distribution']*budget
            key_count = key_count+1
        st.dataframe(df.head(10))
        csv_final = convert_df(df)
        st.download_button(
        label="Download HCP Level Budget",
        data=csv_final,
        file_name="large_df.csv",
        mime="text/csv",)
        
        #gradient graph code ends here
    
    
