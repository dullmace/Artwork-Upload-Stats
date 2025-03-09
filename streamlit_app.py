import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime

# Page config with improved mobile support
st.set_page_config(
    page_title="Last.fm Artwork Upload Stats",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    /* Responsive base styles */
    .stApp {
        padding: 1rem !important;
    }
    
    /* Dynamic font sizes */
    h1 { font-size: calc(1.75rem + 1vw) !important; }
    h2 { font-size: calc(1.25rem + 0.5vw) !important; }
    h3 { font-size: calc(1rem + 0.5vw) !important; }
    
    /* Improved touch targets */
    button[role="tab"] { min-height: 2.5rem; }
    
    /* Better spacing for mobile */
    .stMetric { padding: 0.5rem !important; }
    
    /* Responsive data tables */
    .dataframe { 
        font-size: 0.85rem !important;
        overflow-x: auto !important;
    }
    
    /* Hide fullscreen button on mobile */
    @media (max-width: 768px) {
        .js-plotly-plot .plotly .modebar { display: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('album_counts.csv')
    df.columns = ['Artist', 'Artworks_Uploaded', 'Date_Modified']
    df['Date_Modified'] = pd.to_datetime(df['Date_Modified'])
    df['Year'] = df['Date_Modified'].dt.year
    df['Month'] = df['Date_Modified'].dt.month
    df['Month_Name'] = df['Date_Modified'].dt.month_name()
    
    # Create contribution categories
    bins = [0, 5, 10, 20, 50, np.inf]
    labels = ['Started (1-5)', 'Notable (6-10)', 'Significant (11-20)', 
             'Major (21-50)', 'Exceptional (50+)']
    df['contribution_category'] = pd.cut(
        df['Artworks_Uploaded'],
        bins=bins,
        labels=labels,
        right=False
    )
    
    return df.sort_values('Artworks_Uploaded', ascending=False)

df = load_data()

# Title Section
st.title("🎨 SpotFM Artwork Upload Stats")
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="font-size: 1.1rem; color: #666;">Tracking Artwork Uploads by Dullmace</div>
    <div style="font-size: 0.9rem; color: #999;">Last Updated: {}</div>
</div>
""".format(df['Date_Modified'].max().strftime('%b %d, %Y')), unsafe_allow_html=True)

# Summary Metrics
cols = st.columns(4)
metrics = [
    ("Total Artists", len(df), ""),
    ("Total Uploads", f"{df['Artworks_Uploaded'].sum():,}", ""),
    ("Top Artist", df.iloc[0]['Artist'], f"{df.iloc[0]['Artworks_Uploaded']} uploads"),
    ("Avg Uploads/Artist", f"{df['Artworks_Uploaded'].mean():.1f}", "")
]

for col, (label, value, delta) in zip(cols, metrics):
    with col:
        st.metric(label, value, delta)

# Filters
with st.expander("🔍 Filter Data", expanded=False):
    st.subheader("Filter Options")
    
    # Date Range Picker
    try:
        default_start = df['Date_Modified'].min().date()
        default_end = df['Date_Modified'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end
        )
    except:
        date_range = (default_start, default_end)

# Handle date range selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# Convert to pandas datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply filters
filtered_df = df[
    (df['contribution_category'].isin(categories)) &
    (df['Date_Modified'].between(start_date, end_date)) &
    (df['Artist'].str.contains(search_term, case=False))
]
]

# Visualization Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📅 Timeline", "🎸 Artists"])

with tab1:
    # Contribution Distribution
    st.subheader("Contribution Distribution")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        fig = px.pie(
            filtered_df,
            names='contribution_category',
            values='Artworks_Uploaded',
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.treemap(
            filtered_df,
            path=['contribution_category'],
            values='Artworks_Uploaded',
            color='contribution_category',
            color_discrete_sequence=px.colors.sequential.Viridis_r
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Timeline Visualizations
    st.subheader("Upload Activity Over Time")
    
    # Prepare time data
    time_df = filtered_df.groupby(
        [pd.Grouper(key='Date_Modified', freq='M'), 'contribution_category']
    )['Artworks_Uploaded'].sum().reset_index()
    
    # Line Chart
    fig = px.line(
        time_df,
        x='Date_Modified',
        y='Artworks_Uploaded',
        color='contribution_category',
        markers=True,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Uploads")
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Monthly Heatmap")
    heatmap_df = filtered_df.groupby(
        ['Year', 'Month_Name']
    )['Artworks_Uploaded'].sum().reset_index()
    heatmap_df['Month'] = heatmap_df['Month_Name'].apply(
        lambda x: list(calendar.month_abbr).index(x[:3])
    )
    heatmap_df = heatmap_df.sort_values(['Year', 'Month'])
    
    fig = px.imshow(
        heatmap_df.pivot(index='Year', columns='Month_Name', values='Artworks_Uploaded'),
        labels=dict(x="Month", y="Year", color="Uploads"),
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Artist Details
    st.subheader("Artist Contributions")
    
    # Interactive controls
    cols = st.columns([2, 1, 2])
    with cols[0]:
        sort_by = st.selectbox(
            "Sort by",
            options=['Uploads (High to Low)', 'Artist Name (A-Z)', 'Recent Activity']
        )
    with cols[1]:
        items_per_page = st.selectbox("Items per page", [10, 25, 50])
    with cols[2]:
        search_artist = st.text_input("Search within results")
    
    # Sort data
    if sort_by == 'Uploads (High to Low)':
        sorted_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False)
    elif sort_by == 'Artist Name (A-Z)':
        sorted_df = filtered_df.sort_values('Artist')
    else:
        sorted_df = filtered_df.sort_values('Date_Modified', ascending=False)
    
    # Pagination
    total_pages = len(sorted_df) // items_per_page + 1
    current_page = st.number_input("Page", 1, total_pages, 1)
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    # Display table
    display_cols = ['Artist', 'Artworks_Uploaded', 'contribution_category', 'Date_Modified']
    st.dataframe(
        sorted_df[display_cols][start_idx:end_idx],
        column_config={
            'Date_Modified': 'Last Updated',
            'contribution_category': 'Category'
        },
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #666;">
    <hr style="margin-bottom: 0.5rem;">
    <div style="font-size: 0.9rem;">
        Powered by Streamlit • Data from Last.fm • Updated hourly
    </div>
</div>
""", unsafe_allow_html=True)