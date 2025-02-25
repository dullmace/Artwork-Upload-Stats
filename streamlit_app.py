import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Last.fm Artwork Dashboard",
    page_icon="🎵",
    layout="wide"
)

# Load the CSV data
@st.cache_data
def load_data():
    df = pd.read_csv('album_counts.csv')
    df.columns = ['Artist', 'Artworks_Uploaded', 'Date_Modified']
    df['Date_Modified'] = pd.to_datetime(df['Date_Modified'])
    df['Year'] = df['Date_Modified'].dt.year
    df['Month'] = df['Date_Modified'].dt.month
    df['Month_Name'] = df['Date_Modified'].dt.month_name()
    
    # Create categories
    conditions = [
        (df['Artworks_Uploaded'] > 50),
        (df['Artworks_Uploaded'] > 20),
        (df['Artworks_Uploaded'] > 10),
        (df['Artworks_Uploaded'] > 5),
        (df['Artworks_Uploaded'] > 0)
    ]
    categories = ['Exceptional (50+)', 'Major (21-50)', 'Significant (11-20)', 
                  'Notable (6-10)', 'Started (1-5)']
    df['contribution_category'] = pd.Series(
        np.select(conditions, categories, default='None'),
        index=df.index
    )
    
    # Ensure Artworks_Uploaded is numeric
    df['Artworks_Uploaded'] = pd.to_numeric(df['Artworks_Uploaded'])
    
    # Sort by number of uploaded artworks (descending)
    df = df.sort_values('Artworks_Uploaded', ascending=False)
    
    return df

df = load_data()

# Title
st.title("Last.fm Album Artwork Contribution Dashboard")
st.subheader("Tracking Your Artwork Upload Achievements")

# Summary statistics
col1, col2, col3, col4 = st.columns(4)

# Get the top contribution (first row after sorting)
top_artist = df.iloc[0]['Artist']
top_contribution = int(df.iloc[0]['Artworks_Uploaded'])  # Convert to int to avoid decimal display

with col1:
    st.metric("Total Artists", len(df))
with col2:
    st.metric("Total Uploads", int(df['Artworks_Uploaded'].sum()))
with col3:
    st.metric("Top Contribution", top_contribution, f"by {top_artist}")
with col4:
    st.metric("Latest Upload", df['Date_Modified'].max().strftime('%b %d, %Y'))

# Filters
st.subheader("Explore Your Contributions")
col1, col2 = st.columns(2)
with col1:
    categories = ['Exceptional (50+)', 'Major (21-50)', 'Significant (11-20)', 
                  'Notable (6-10)', 'Started (1-5)']
    selected_categories = st.multiselect("Contribution Level", categories, default=categories)
with col2:
    date_range = st.date_input(
        "Date Range",
        [df['Date_Modified'].min().date(), df['Date_Modified'].max().date()]
    )

# Apply filters
filtered_df = df.copy()
if selected_categories:
    filtered_df = filtered_df[filtered_df['contribution_category'].isin(selected_categories)]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Date_Modified'].dt.date >= date_range[0]) & 
                             (filtered_df['Date_Modified'].dt.date <= date_range[1])]

# Search
search_term = st.text_input("Search Artist")
if search_term:
    filtered_df = filtered_df[filtered_df['Artist'].str.contains(search_term, case=False)]

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Time Analysis", "Artist Details"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.pie(
                filtered_df, 
                names='contribution_category', 
                values='Artworks_Uploaded',
                title='Distribution of Your Artwork Contributions',
                color='contribution_category',
                color_discrete_sequence=px.colors.sequential.Viridis
            ),
            use_container_width=True
        )
    with col2:
        # Get top 10 artists by contribution count
        top10_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False).head(10)
        
        # Create bar chart with proper sorting
        fig = px.bar(
            top10_df,
            x='Artworks_Uploaded',
            y='Artist',
            orientation='h',
            title='Top 10 Artists by Your Artwork Contributions',
            color='Artworks_Uploaded',
            color_continuous_scale='Viridis'
        )
        
        # Ensure the y-axis is ordered by contribution count (descending)
        fig.update_layout(
            yaxis={'categoryorder': 'array', 'categoryarray': top10_df['Artist'].tolist()[::-1]}
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Monthly trend
    monthly_uploads = filtered_df.groupby(['Year', 'Month', 'Month_Name'])['Artworks_Uploaded'].sum().reset_index()
    monthly_uploads = monthly_uploads.sort_values(['Year', 'Month'])
    
    st.plotly_chart(
        px.line(
            monthly_uploads,
            x='Month_Name',
            y='Artworks_Uploaded',
            color='Year',
            title='Monthly Upload Trends',
            labels={'Artworks_Uploaded': 'Artworks Uploaded', 'Month_Name': 'Month'},
            markers=True
        ).update_layout(xaxis={'categoryorder': 'array', 
                              'categoryarray': list(calendar.month_name)[1:]}),
        use_container_width=True
    )
    
    # Heatmap
    st.plotly_chart(
        px.density_heatmap(
            filtered_df,
            x='Month_Name',
            y='Year',
            z='Artworks_Uploaded',
            title='Upload Activity Heatmap',
            color_continuous_scale='Viridis'
        ).update_layout(xaxis={'categoryorder': 'array', 
                              'categoryarray': list(calendar.month_name)[1:]}),
        use_container_width=True
    )

with tab3:
    sort_option = st.radio(
        "Sort by:",
        ["Most Contributions", "Alphabetical", "Most Recent"],
        horizontal=True
    )
    
    if sort_option == "Most Contributions":
        display_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False)
    elif sort_option == "Alphabetical":
        display_df = filtered_df.sort_values('Artist')
    else:  # Most Recent
        display_df = filtered_df.sort_values('Date_Modified', ascending=False)
    
    # Display table
    st.dataframe(
        display_df[['Artist', 'Artworks_Uploaded', 'contribution_category', 'Date_Modified']].rename(
            columns={'Date_Modified': 'Last Updated'}
        ),
        use_container_width=True,
        hide_index=True
    )
