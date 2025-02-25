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
    initial_sidebar_state="collapsed"  # Collapse sidebar on mobile by default
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    /* Improve readability on mobile */
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    
    /* Make text more readable on small screens */
    h1 {
        font-size: calc(1.5rem + 1vw) !important;
    }
    h2, h3, .metric-label {
        font-size: calc(1rem + 0.5vw) !important;
    }
    p, .metric-value {
        font-size: calc(0.8rem + 0.3vw) !important;
    }
    
    /* Improve table display on mobile */
    .dataframe {
        font-size: 0.8rem !important;
        white-space: nowrap;
    }
    
    /* Better padding for cards */
    div.block-container {
        padding: 2rem 1rem 10rem 1rem;
    }
    
    /* Improve metric cards on mobile */
    [data-testid="stMetricValue"] {
        font-size: calc(1.2rem + 0.5vw) !important;
    }
    
    /* Ensure charts are responsive */
    .js-plotly-plot, .plotly, .plot-container {
        width: 100% !important;
    }
    
    /* Improve tabs on mobile */
    button[role="tab"] {
        min-width: auto !important;
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Title - simplified for mobile
st.title("SpotFM Artwork Upload Stats")
st.caption("Tracking Artwork Uploaded by Dullmace (and their script)")

# Detect if we're on mobile (approximate method)
# This helps us adjust layouts dynamically
def is_mobile():
    # A simple heuristic - we'll use session state to remember the result
    if 'is_mobile' not in st.session_state:
        # We'll use a container width check as proxy for mobile
        container_width = st.get_option("theme.base")
        st.session_state.is_mobile = True if container_width == "light" else False
    return st.session_state.is_mobile

# Summary statistics - responsive layout
if is_mobile():
    # Stack metrics vertically on mobile
    col1 = st.container()
    with col1:
        st.metric("Total Artists", len(df))
        st.metric("Total Uploads", int(df['Artworks_Uploaded'].sum()))
        st.metric("Top Contribution", int(df.iloc[0]['Artworks_Uploaded']), f"by {df.iloc[0]['Artist']}")
        st.metric("Latest Upload", df['Date_Modified'].max().strftime('%b %d, %Y'))
else:
    # Use columns on desktop
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Artists", len(df))
    with col2:
        st.metric("Total Uploads", int(df['Artworks_Uploaded'].sum()))
    with col3:
        st.metric("Top Contribution", int(df.iloc[0]['Artworks_Uploaded']), f"by {df.iloc[0]['Artist']}")
    with col4:
        st.metric("Latest Upload", df['Date_Modified'].max().strftime('%b %d, %Y'))

# Filters - simplified for mobile
with st.expander("Filters & Search", expanded=False):
    # Categories filter
    categories = ['Exceptional (50+)', 'Major (21-50)', 'Significant (11-20)', 
                'Notable (6-10)', 'Started (1-5)']
    selected_categories = st.multiselect("Contribution Level", categories, default=categories)
    
    # Date range - simplified for mobile
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", df['Date_Modified'].min().date())
    with date_col2:
        end_date = st.date_input("End Date", df['Date_Modified'].max().date())
    
    # Search
    search_term = st.text_input("Search Artist")
    
    # Reset button
    if st.button("Reset Filters"):
        selected_categories = categories
        start_date = df['Date_Modified'].min().date()
        end_date = df['Date_Modified'].max().date()
        search_term = ""

# Apply filters
filtered_df = df.copy()
if selected_categories:
    filtered_df = filtered_df[filtered_df['contribution_category'].isin(selected_categories)]
if start_date and end_date:
    filtered_df = filtered_df[(filtered_df['Date_Modified'].dt.date >= start_date) & 
                             (filtered_df['Date_Modified'].dt.date <= end_date)]
if search_term:
    filtered_df = filtered_df[filtered_df['Artist'].str.contains(search_term, case=False)]

# Tabs - mobile friendly
tab1, tab2, tab3 = st.tabs(["Overview", "Time", "Artists"])

with tab1:
    # Responsive layout for charts
    if is_mobile():
        # Stack charts vertically on mobile
        # Pie chart
        st.subheader("Contribution Categories")
        st.plotly_chart(
            px.pie(
                filtered_df, 
                names='contribution_category', 
                values='Artworks_Uploaded',
                color='contribution_category',
                color_discrete_sequence=px.colors.sequential.Viridis
            ).update_layout(
                margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins for mobile
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)  # Horizontal legend below chart
            ),
            use_container_width=True
        )
        
        # Top artists chart
        st.subheader("Top 25 Artists")
        # Number slider for mobile
        num_artists = st.slider("Number to display", 5, 25, 10, 5)
        top_artists_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False).head(num_artists)
        
        fig = px.bar(
            top_artists_df,
            x='Artworks_Uploaded',
            y='Artist',
            orientation='h',
            color='Artworks_Uploaded',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'array', 'categoryarray': top_artists_df['Artist'].tolist()[::-1]},
            height=max(400, num_artists * 25),  # Dynamic height based on number of artists
            margin=dict(l=10, r=10, t=10, b=10)  # Tighter margins for mobile
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Desktop layout with columns
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
            # Get top 25 artists by contribution count
            top25_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False).head(25)
            
            # Create bar chart with proper sorting
            fig = px.bar(
                top25_df,
                x='Artworks_Uploaded',
                y='Artist',
                orientation='h',
                title='Top 25 Artists by Your Artwork Contributions',
                color='Artworks_Uploaded',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'array', 'categoryarray': top25_df['Artist'].tolist()[::-1]},
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Monthly trend - simplified for mobile
    st.subheader("Monthly Upload Trends")
    monthly_uploads = filtered_df.groupby(['Year', 'Month', 'Month_Name'])['Artworks_Uploaded'].sum().reset_index()
    monthly_uploads = monthly_uploads.sort_values(['Year', 'Month'])
    
    # Simplified line chart for mobile
    line_fig = px.line(
        monthly_uploads,
        x='Month_Name',
        y='Artworks_Uploaded',
        color='Year',
        markers=True
    )
    
    line_fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': list(calendar.month_name)[1:]},
        margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins for mobile
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)  # Horizontal legend below chart
    )
    
    st.plotly_chart(line_fig, use_container_width=True)
    
    # Heatmap - simplified for mobile
    st.subheader("Upload Activity Heatmap")
    heatmap_fig = px.density_heatmap(
        filtered_df,
        x='Month_Name',
        y='Year',
        z='Artworks_Uploaded',
        color_continuous_scale='Viridis'
    )
    
    heatmap_fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': list(calendar.month_name)[1:]},
        margin=dict(l=10, r=10, t=30, b=10)  # Tighter margins for mobile
    )
    
    st.plotly_chart(heatmap_fig, use_container_width=True)

with tab3:
    # Artist details - simplified for mobile
    st.subheader("Artist Details")
    
    # Simplified sort options
    sort_option = st.radio(
        "Sort by:",
        ["Most Contributions", "Alphabetical", "Most Recent"],
        horizontal=True if not is_mobile() else False
    )
    
    if sort_option == "Most Contributions":
        display_df = filtered_df.sort_values('Artworks_Uploaded', ascending=False)
    elif sort_option == "Alphabetical":
        display_df = filtered_df.sort_values('Artist')
    else:  # Most Recent
        display_df = filtered_df.sort_values('Date_Modified', ascending=False)
    
    # Mobile-friendly table with fewer columns
    if is_mobile():
        # Simplified table for mobile
        st.dataframe(
            display_df[['Artist', 'Artworks_Uploaded']],
            use_container_width=True,
            hide_index=True
        )
        
        # Show details on demand
        with st.expander("Show Full Details"):
            st.dataframe(
                display_df[['Artist', 'Artworks_Uploaded', 'contribution_category', 'Date_Modified']].rename(
                    columns={'Date_Modified': 'Last Updated'}
                ),
                use_container_width=True,
                hide_index=True
            )
    else:
        # Full table for desktop
        st.dataframe(
            display_df[['Artist', 'Artworks_Uploaded', 'contribution_category', 'Date_Modified']].rename(
                columns={'Date_Modified': 'Last Updated'}
            ),
            use_container_width=True,
            hide_index=True
        )

# Footer with responsive design
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; font-size: 0.8rem; color: #666;">
    Last.fm Artwork Upload Dashboard<br>
    Created with Streamlit and Plotly
</div>
""", unsafe_allow_html=True)
