import calendar
import json
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from thefuzz import process
from wordcloud import WordCloud, STOPWORDS

# Page config (only once)
st.set_page_config(
    page_title="🎶 SpotFM Artwork Upload Stats",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 🎨 Dark Mode Styles
st.markdown(
    """
    <style>
    .stDataFrame { 
        background-color: #1E1E1E; 
        color: white; 
    }
    .stButton button { 
        background-color: #2196F3; 
        color: white;
    }
    .stTextInput > div > div > input {
        background: #2D2D2D !important; 
        color: white !important;
    }
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Boogaloo&display=swap');

    /* Apply Roboto font globally */
    * {
        font-family: 'Roboto', sans-serif;
    }

    /* Optionally, tweak tab styles as well */
    div[data-baseweb="tab"] button {
        height: 50px !important;
        font-size: 20px !important;
        padding: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and preprocess data
@st.cache_data
def load_album_data():
    with open("artist_albums.json") as f:
        data = json.load(f)

    normalized = {}
    for artist, albums in data.items():
        key = artist.lower()
        album_list = [
            {
                "album": album["album"],
                "uploaded_date": pd.to_datetime(album["creation_date"]),
            }
            for album in albums
        ]
        if key in normalized:
            normalized[key]["albums"].extend(album_list)
            normalized[key]["count"] += len(album_list)
        else:
            normalized[key] = {
                "display_name": artist,
                "albums": album_list,
                "count": len(album_list),
            }
    return normalized


@st.cache_data
def load_data():
    df = pd.read_csv("album_counts.csv")
    df.columns = ["Artist", "Artworks_Uploaded", "Date_Modified"]
    df["Date_Modified"] = pd.to_datetime(df["Date_Modified"])
    df["Year"] = df["Date_Modified"].dt.year
    df["Month"] = df["Date_Modified"].dt.month
    df["Month_Name"] = df["Date_Modified"].dt.month_name()

    album_data = load_album_data()
    df["Albums_Count"] = df["Artist"].str.lower().map(
        lambda x: album_data.get(x, {}).get("count", 0)
    )
    df["Albums"] = df["Artist"].str.lower().map(
        lambda x: tuple(album["album"] for album in album_data.get(x, {}).get("albums", []))
    )

    df["Album_Uploaded_Dates"] = df["Artist"].str.lower().map(
        lambda x: tuple(
            pd.to_datetime(album["uploaded_date"], unit="ms")
            for album in album_data.get(x, {}).get("albums", [])
        )
    )


    bins = [0, 5, 10, 20, 50, np.inf]
    labels = [
        "Started (1-5)",
        "Notable (6-10)",
        "Significant (11-20)",
        "Major (21-50)",
        "Exceptional (50+)",
    ]
    df["contribution_category"] = pd.cut(
        df["Artworks_Uploaded"], bins=bins, labels=labels, right=False
    )

    # Remove underscores from contribution_category labels if any
    df["contribution_category"] = df["contribution_category"].astype(str).str.replace(
        "_", " "
    )
    return df.sort_values("Artworks_Uploaded", ascending=False)


@st.cache_data
def preprocess_data(df):
    monthly_uploads = df.groupby(pd.Grouper(key="Date_Modified", freq="ME")).size()
    category_dist = df["contribution_category"].value_counts(normalize=True)
    cumulative_uploads = df.sort_values("Date_Modified").assign(
        cumulative=lambda x: x["Artworks_Uploaded"].cumsum()
    )
    return {
        "monthly_uploads": monthly_uploads,
        "category_dist": category_dist,
        "cumulative_uploads": cumulative_uploads,
    }


df = load_data()
preprocessed = preprocess_data(df)


def create_lastfm_artist_url(artist_name):
    encoded_artist_name = urllib.parse.quote_plus(artist_name)
    return f"https://www.last.fm/music/{encoded_artist_name}"


def create_lastfm_release_url(artist_name, release_name):
    encoded_artist_name = urllib.parse.quote_plus(artist_name)
    encoded_release_name = urllib.parse.quote_plus(release_name)
    return f"https://www.last.fm/music/{encoded_artist_name}/{encoded_release_name}"


# Title Section
st.title("🎶 SpotFM Artwork Uploads 🎶")
st.subheader("🎵 Visualizing Missing Artwork Contributions by Dullmace on Last.fm")

# Filters
with st.expander("🔍 Filter Data", expanded=False):
    st.subheader("🎵 Advanced Filters")
    search_term = st.text_input("Search Artists", help="Search for specific artists.")
    album_search = st.text_input(
        "🔍 Search Albums", help="Filter artists by album name", key="album_search"
    )

    col1, col2 = st.columns(2)
    with col1:
        all_categories = df["contribution_category"].unique().tolist()
        selected_categories = st.multiselect(
            "Contribution Level",
            all_categories,
            default=all_categories,
            help="Filter by the contribution level of the artist.",
        )
    with col2:
        default_start = df["Date_Modified"].min().date()
        default_end = df["Date_Modified"].max().date()
        date_range = st.date_input(
            "Date Range",
            (default_start, default_end),
            min_value=default_start,
            max_value=default_end,
            help="Filter data by the date range of artwork uploads.",
        )

    album_date_range = st.date_input(
        "Album Uploaded Date Range",
        (
            pd.to_datetime(df["Album_Uploaded_Dates"].explode().min()).date(),
            pd.to_datetime(df["Album_Uploaded_Dates"].explode().max()).date(),
        ),
        help="Filter artists based on the upload date of their albums.",
    )

# Date handling for filtering
start_date, end_date = (pd.to_datetime(d) for d in date_range)
album_start_date, album_end_date = (pd.to_datetime(d) for d in album_date_range)

# Filter data based on main filters
filtered_df = df[
    (df["contribution_category"].isin(selected_categories))
    & (df["Date_Modified"] >= start_date)
    & (df["Date_Modified"] <= end_date)
]

# Further filter based on album uploaded date range
def filter_by_album_date(df, start, end):
    def album_in_date_range(album_dates, start_date, end_date):
        if not isinstance(album_dates, tuple):
            return False
        return any(start_date <= date <= end_date for date in album_dates)

    return df[
        df["Album_Uploaded_Dates"].apply(lambda x: album_in_date_range(x, start, end))
    ]


filtered_df = filter_by_album_date(filtered_df, album_start_date, album_end_date)

# Fuzzy search for artist names
if search_term:
    with st.spinner("Searching artists..."):
        artists = filtered_df["Artist"].unique()
        matches = process.extract(search_term, artists, limit=50)
        filtered_df = filtered_df[filtered_df["Artist"].isin([m[0] for m in matches])]

# Filter by album search if provided
if album_search:
    filtered_df = filtered_df[
        filtered_df["Albums"].apply(
            lambda albums: any(
                album_search.lower() in a.lower() for a in albums if isinstance(a, str)
            )
        )
    ]

st.download_button(
    "📥 Download Data",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="spotfm_data.csv",
    mime="text/csv",
    key="download_button",
)

# Display filtered data
st.dataframe(filtered_df)

if filtered_df.empty:
    st.error(
        """
        🎭 No artists match these filters!
        Try:
        - Widening date range
        - Including more categories
        - Clearing search terms
        """
    )
    st.stop()

# Metrics
st.subheader("📊 Key Metrics")
total_artists = len(filtered_df)
total_uploads = filtered_df["Artworks_Uploaded"].sum()
total_albums = filtered_df["Albums_Count"].sum()
top_artist = filtered_df.iloc[0]["Artist"]
top_uploads = filtered_df.iloc[0]["Artworks_Uploaded"]
avg_uploads = filtered_df["Artworks_Uploaded"].mean()
earliest_album_date = filtered_df["Album_Uploaded_Dates"].explode().min()
latest_album_date = filtered_df["Album_Uploaded_Dates"].explode().max()

cols = st.columns(5)
metrics = [
    (
        "Total Artists",
        total_artists,
        "Total number of artists in the filtered dataset.",
    ),
    (
        "Total Uploads",
        f"{total_uploads:,}",
        "Total number of artworks uploaded.",
    ),
    (
        "Total Albums",
        f"{total_albums:,}",
        "Total number of albums.",
    ),
    (
        "Earliest Album Date",
        earliest_album_date.strftime("%b %d, %Y")
        if pd.notnull(earliest_album_date)
        else "N/A",
        "Date of the earliest uploaded album.",
    ),
    (
        "Latest Album Date",
        latest_album_date.strftime("%b %d, %Y")
        if pd.notnull(latest_album_date)
        else "N/A",
        "Date of the latest uploaded album.",
    ),
]

for col, (label, value, description) in zip(cols, metrics):
    with col:
        st.metric(label, value, help=description)

st.markdown(
    """
    <style>
    /* Increase the height and font-size of the tab buttons */
    div[data-baseweb="tab"] button {
        height: 50px !important;
        font-size: 20px !important;
        padding: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Visualization Tabs
tab1, tab2, tab3 = st.tabs(["🎸 Artists", "📊 Stats", "📅 Timeline"])

with tab1:
    st.header("🎸 Artist Hub")
    st.write("Explore individual artist contributions.")
    viz_choice = st.radio(
        "Choose Visualization:",
        options=[
            "Top Contributors",
            "Category Breakdown",
            "Album Explorer",
            "Artist Timeline",
            "Artist Word Cloud",
        ],
        horizontal=True,
        help="Select a visualization to explore artist-related data.",
    )

    if viz_choice == "Category Breakdown":
        st.subheader("Contribution Distribution")
        category_counts = (
            filtered_df["contribution_category"]
            .value_counts()
            .reset_index()
        )
        category_counts.columns = ["category", "count"]
        fig = px.bar(
            category_counts,
            x="category",
            y="count",
            title="Contribution Category Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Album Explorer":
        st.subheader("Album Explorer")
        st.write("Dive into each artist's album contributions.")
        artist_album = filtered_df.explode("Albums").copy()

        fig = px.treemap(
            artist_album,
            path=["Artist", "Albums"],
            values="Artworks_Uploaded",
            color="Artworks_Uploaded",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(textinfo="label+value")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)

        search_album = st.text_input(
            "Search Albums within Explorer",
            help="Search for albums within the Album Explorer.",
            key="search_album_explorer",
        )
        if search_album:
            matches = artist_album[
                artist_album["Albums"].str.contains(search_album, case=False, na=False)
            ]
            st.dataframe(matches[["Artist", "Albums", "Artworks_Uploaded"]])

    elif viz_choice == "Top Contributors":
        num_artists = st.slider(
            "Number of artists to show",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            help="Select the number of top artists to display.",
        )
        top_artists = filtered_df.nlargest(num_artists, "Artworks_Uploaded").copy()

        fig = px.bar(
            top_artists,
            x="Artworks_Uploaded",
            y="Artist",
            orientation="h",
            color="Artworks_Uploaded",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            height=max(500, num_artists * 20),
            xaxis_title="Artworks Uploaded",
            yaxis_title="Artist",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Artist Timeline":
        st.subheader("Artist Timeline")
        st.write("See artwork uploads over time by artist.")
        artist_timeline = filtered_df.explode("Album_Uploaded_Dates").copy()

        artist_timeline["Artworks_Uploaded"] = pd.to_numeric(
            artist_timeline["Artworks_Uploaded"], errors="coerce"
        ).fillna(0)

        artist_timeline = (
            artist_timeline.groupby(["Album_Uploaded_Dates", "Artist"])[
                "Artworks_Uploaded"
            ]
            .sum()
            .reset_index()
        )
        fig = px.line(
            artist_timeline,
            x="Album_Uploaded_Dates",
            y="Artworks_Uploaded",
            color="Artist",
            title="Artist Upload Timeline",
            labels={"Album_Uploaded_Dates": "Date", "Artworks_Uploaded": "Uploads"},
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Artist Word Cloud":
        st.subheader("Artist Word Cloud")
        stopwords = set(STOPWORDS) | {"the", "and", "of"}
        albums_text = " ".join(
            filtered_df["Albums"].explode().dropna().astype(str)
        )
        wordcloud = (
            WordCloud(
                width=800,
                height=400,
                background_color="black",
                stopwords=stopwords,
            )
            .generate(albums_text)
        )
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf(), use_container_width=True)

# Album Badges
st.subheader("🏅 Artist Badges")
num_badges = st.slider(
    "Number of badges to display",
    min_value=5,
    max_value=100,
    value=24,
    help="Choose how many artist badges to show.",
    key="num_badges",
)
num_cols = st.slider(
    "Columns layout",
    min_value=1,
    max_value=7,
    value=4,
    key="num_cols",
    help="Adjust the layout of artist badges across the page.",
)
cols = st.columns(num_cols)
top_badges = filtered_df.nlargest(num_badges, "Artworks_Uploaded").copy()

colors = ["#267A9E", "#51A885", "#F5A936", "#ED8C37", "#DB7476", "#986B9B"]

for idx, (_, row) in enumerate(top_badges.iterrows()):
    bg_color = colors[idx % len(colors)]
    with cols[idx % num_cols]:
        badge_html = f"""
        <div style="
            background: {bg_color};
            padding: 15px;
            border: 1px solid #444;
            border-radius: 8px;
            margin: 5px;
            color: #000;
            font-family: "Boogaloo", sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);">
            <div>
                <strong style="font-size:1.2em;">{row["Artist"]}</strong>
                <p style="margin: 5px 0; color: #EEE;">{row["Artworks_Uploaded"]} uploads</p>
                <a href="{create_lastfm_artist_url(row["Artist"])}" 
                   target="_blank" style="
                       color: #FFF;
                       text-decoration: none;
                       font-weight: bold;
                       background: rgba(0,0,0,0.3);
                       padding: 4px 8px;
                       border-radius: 4px;
                       text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);">
                    View on Last.fm &rarr;
                </a>
            </div>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        with st.expander("View Releases Uploaded", expanded=False):
            if row["Albums"]:
                st.markdown("<strong>Albums:</strong>", unsafe_allow_html=True)
                album_list = "<ul style='padding-left:20px; margin:5px 0; color:#ccc;'>"
                for album in row["Albums"]:
                    album_list += (
                        f"""<li>
                            <a href="{create_lastfm_release_url(row["Artist"], album)}" """
                        f"""target="_blank" style="color:#1E90FF; text-decoration:none;">"""
                        f"""{album}</a>
                        </li>"""
                    )
                album_list += "</ul>"
                st.markdown(album_list, unsafe_allow_html=True)





with tab2:
    st.subheader("Album Analysis")
    st.write("Album uploads across different months and years.")
    album_timeline = (
        filtered_df.explode("Albums")
        .groupby(["Year", "Month_Name"])["Albums"]
        .count()
        .reset_index()
        .rename(columns={"Albums": "Count"})
    )
    
    st.subheader("Top Albums by Uploads")
    album_counts = (
        filtered_df.explode("Albums")["Albums"]
        .value_counts()
        .rename_axis("Album")
        .reset_index(name="Uploads")
        .head(10)
    )
    fig = px.bar(
        album_counts,
        x="Album",
        y="Uploads",
        title="Top 10 Albums by Uploads",
        labels={"Album": "Album", "Uploads": "Uploads"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.area(
        album_timeline,
        x="Month_Name",
        y="Count",
        color="Year",
        title="Album Artwork Uploads Timeline",
    )
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Album Title Length Analysis")
    st.write("Distribution of album title lengths.")
    album_titles = filtered_df.explode("Albums").copy()
    album_titles["title_length"] = album_titles["Albums"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    fig = px.histogram(album_titles, x="title_length", nbins=20, marginal="box",
                       title="Album Title Length Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Upload Activity Over Time")
    st.write("Cumulative uploads over time.")
    fig = px.area(
        preprocessed["cumulative_uploads"],
        x="Date_Modified",
        y="cumulative",
        title="Total Uploads Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Heatmap")
    st.write("Artworks uploaded each month.")
    heatmap_df = (
        filtered_df.groupby(["Year", "Month_Name"])["Artworks_Uploaded"]
        .sum()
        .reset_index()
    )
    heatmap_df["Month_Name"] = pd.Categorical(
        heatmap_df["Month_Name"],
        categories=[calendar.month_name[i] for i in range(1, 13)],
        ordered=True,
    )
    pivot_table = heatmap_df.pivot(index="Year",
                                   columns="Month_Name",
                                   values="Artworks_Uploaded")
    fig = px.imshow(
        pivot_table,
        color_continuous_scale="Viridis",
        title="Monthly Upload Activity Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown(
    """
    <div style="
        background: #1E1E1E;
        padding: 15px;
        text-align: center;
        color: #999;
    ">
        <p>🎵 Built with ❤️ using <a href="https://streamlit.io" target="_blank">
        Streamlit</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global Theme Configuration
def configure_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#2D2D2D",
        font_color="white",
        hoverlabel_bgcolor="#333"
    )
    return fig
