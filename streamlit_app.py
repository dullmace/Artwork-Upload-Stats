import calendar
import json
from datetime import datetime
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from thefuzz import process
from wordcloud import WordCloud

# Page config
st.set_page_config(
    page_title="Last.fm Artwork Upload Stats",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    df["Albums_Count"] = (
        df["Artist"]
        .str.lower()
        .map(lambda x: album_data.get(x, {}).get("count", 0))
    )
    df["Albums"] = df["Artist"].str.lower().map(
        lambda x: tuple(
            album["album"] for album in album_data.get(x, {}).get("albums", [])
        )
    )

    # Convert Album_Uploaded_Dates to datetime objects
    df["Album_Uploaded_Dates"] = df["Artist"].str.lower().map(
        lambda x: tuple(
            pd.to_datetime(album["uploaded_date"], unit="ms")  # Convert to datetime
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

    # Remove underscores from contribution_category labels
    df["contribution_category"] = df["contribution_category"].astype(str).str.replace(
        "_", " "
    )
    return df.sort_values("Artworks_Uploaded", ascending=False)


@st.cache_data
def preprocess_data(df):
    return {
        "monthly_uploads": df.groupby(pd.Grouper(key="Date_Modified", freq="ME")).size(),
        "category_dist": df["contribution_category"].value_counts(normalize=True),
        "cumulative_uploads": df.sort_values("Date_Modified").assign(
            cumulative=lambda x: x["Artworks_Uploaded"].cumsum()
        ),
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
st.title("🎨 SpotFM Artwork Upload Stats")

# Filters
with st.expander("🔍 Filter Data", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        all_categories = df["contribution_category"].unique().tolist()
        selected_categories = st.multiselect(
            "Contribution Level", all_categories, default=all_categories
        )
    with col2:
        default_start = df["Date_Modified"].min().date()
        default_end = df["Date_Modified"].max().date()
        date_range = st.date_input(
            "Date Range",
            (default_start, default_end),
            min_value=default_start,
            max_value=default_end,
        )

    album_date_range = st.date_input(
        "Album Uploaded Date Range",
        (
            pd.to_datetime(df["Album_Uploaded_Dates"].explode().min()).date(),
            pd.to_datetime(df["Album_Uploaded_Dates"].explode().max()).date(),
        ),
    )

    search_term = st.text_input("Search Artists")

# Date handling
start_date, end_date = (pd.to_datetime(d) for d in date_range)
album_start_date, album_end_date = (pd.to_datetime(d) for d in album_date_range)

# Filter data
filtered_df = df[
    (df["contribution_category"].isin(selected_categories))
    & (df["Date_Modified"] >= start_date)
    & (df["Date_Modified"] <= end_date)
]


def filter_by_album_date(df, start_date, end_date):
    def album_in_date_range(album_dates, start, end):
        if not isinstance(album_dates, tuple):
            return False
        for date in album_dates:
            if start <= date <= end:
                return True
        return False

    filtered = df[
        df["Album_Uploaded_Dates"].apply(
            lambda x: album_in_date_range(x, album_start_date, album_end_date)
        )
    ]
    return filtered


filtered_df = filter_by_album_date(filtered_df, album_start_date, album_end_date)

# Fuzzy search
if search_term:
    with st.spinner("Searching artists..."):
        artists = filtered_df["Artist"].unique()
        matches = process.extract(search_term, artists, limit=50)
        filtered_df = filtered_df[
            filtered_df["Artist"].isin([m[0] for m in matches])
        ]

# Display filtered data
st.dataframe(filtered_df)

# Empty state
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
    ("Total Artists", total_artists, ""),
    ("Total Uploads", f"{total_uploads:,}", ""),
    ("Total Albums", f"{total_albums:,}", ""),
    ("Earliest Album Date", earliest_album_date.strftime("%b %d, %Y"), ""),
    ("Latest Album Date", latest_album_date.strftime("%b %d, %Y"), ""),
]
for col, (label, value, delta) in zip(cols, metrics):
    with col:
        st.metric(label, value, delta)

# Visualization Tabs
tab1, tab2, tab3 = st.tabs(["🎸 Artists", "📊 Overview", "📅 Timeline"])

with tab1:
    st.header("🎸 Artist Spotlight")
    viz_choice = st.radio(
        "Choose Visualization:",
        [
            "Top Contributors",
            "Category Breakdown",
            "Album Explorer",
            "Artist Timeline",  # ADDED: Artist Timeline
            "Artist Word Cloud",  # ADDED: Artist Word Cloud
        ],
        horizontal=True,
    )
    if viz_choice == "Category Breakdown":
        st.subheader("Contribution Distribution")
        category_counts = filtered_df['contribution_category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']

        fig = px.bar(category_counts, x='category', y='count',
                     title='Contribution Category Breakdown')
        st.plotly_chart(fig, use_container_width=True)

    if viz_choice == "Album Explorer":
        st.subheader("Album Explorer")
        artist_album = filtered_df.explode("Albums").copy()

        fig = px.treemap(
            artist_album,
            path=["Artist", "Albums"],
            values="Artworks_Uploaded",
            color="Artworks_Uploaded",
            color_continuous_scale="Viridis",
        )

        fig.update_traces(
            textinfo="label+value",
        )
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25)
        )  # Adjust margins as needed
        st.plotly_chart(fig, use_container_width=True)

        search_album = st.text_input("Search Albums")
        if search_album:
            matches = artist_album[
                artist_album["Albums"].str.contains(search_album, case=False, na=False)
            ]
            st.dataframe(matches[["Artist", "Albums", "Artworks_Uploaded"]])

    elif viz_choice == "Top Contributors":
        num_artists = st.slider("Number of artists to show", 10, 100, 25)
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
        artist_timeline = filtered_df.explode("Album_Uploaded_Dates").copy()

        # Ensure 'Artworks_Uploaded' is numeric and not a string or mixed type
        artist_timeline["Artworks_Uploaded"] = pd.to_numeric(
            artist_timeline["Artworks_Uploaded"], errors="coerce"
        ).fillna(
            0
        )  # Fill NaN values with 0 after coercion

        # Group by 'Album_Uploaded_Dates' and sum 'Artworks_Uploaded' for each date
        artist_timeline = (
            artist_timeline.groupby(["Album_Uploaded_Dates", "Artist"])[
                "Artworks_Uploaded"
            ]
            .sum()
            .reset_index()
        )

        artist_timeline = artist_timeline.sort_values("Album_Uploaded_Dates")
        fig = px.line(
            artist_timeline,
            x="Album_Uploaded_Dates",
            y="Artworks_Uploaded",
            color="Artist",
            title="Artwork Uploads Timeline by Artist",
            hover_data=["Artist", "Artworks_Uploaded"]
        )
        fig.update_traces(
             hovertemplate="Artist: %{customdata[0]}<br>Artworks Uploaded: %{customdata[1]}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    # ADDED: Artist Word Cloud
    elif viz_choice == "Artist Word Cloud":
        st.subheader("Artist Word Cloud")
        all_albums = " ".join(filtered_df["Albums"].explode().dropna())
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate(all_albums)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    st.subheader("🏅 Artist Badges")
    num_badges = st.slider("Number of badges to display", 5, 50, 15)
    top_badges = filtered_df.nlargest(num_badges, "Artworks_Uploaded").copy()

    cols = st.columns(st.session_state.get("num_cols", 3))
    st.slider("Columns layout", 1, 5, 3, key="num_cols")

    for idx, (_, row) in enumerate(top_badges.iterrows()):
        with cols[idx % st.session_state.num_cols]:
            artist_name = row["Artist"]
            #artist_url = create_lastfm_artist_url(artist_name) #Not needed anymore, it's now plaintext
            with st.expander(
                f"**{artist_name}** - {row['Artworks_Uploaded']} uploads",
                expanded=False,
            ):

                # Display a list of hyperlinked albums
                if row["Albums"]:
                    album_links = []
                    for album in row["Albums"]:
                        artist_name = row["Artist"] #You need to bring the Artist name into this scope for the albums to render
                        album_url = create_lastfm_release_url(artist_name, album)
                        album_links.append(
                            f"- <a href='{album_url}' target='_blank'>{album}</a>"
                        )
                    st.markdown("<br>".join(album_links), unsafe_allow_html=True)

with tab2:
    st.subheader("Contribution Distribution")
    col1, col2 = st.columns([2, 3])
    with col1:
        fig = px.pie(
            preprocessed["category_dist"],
            names=preprocessed["category_dist"].index,
            values="proportion",
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.treemap(
            filtered_df,
            path=["contribution_category"],
            values="Artworks_Uploaded",
            color="contribution_category",
            color_discrete_sequence=px.colors.sequential.Viridis_r,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Album Analysis")
    album_timeline = filtered_df.explode("Albums")
    album_timeline = (
        album_timeline.groupby(["Year", "Month_Name"])["Albums"]
        .count()
        .reset_index()
    )
    fig = px.area(
        album_timeline,
        x="Month_Name",
        y="Albums",
        color="Year",
        title="Album Artwork Uploads Timeline",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Album Title Length Analysis")
    album_timeline = filtered_df.explode("Albums")
    album_timeline["title_length"] = album_timeline["Albums"].apply(len)
    fig = px.histogram(album_timeline, x="title_length", nbins=20, marginal="box")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Upload Activity Over Time")
    fig = px.area(
        preprocessed["cumulative_uploads"],
        x="Date_Modified",
        y="cumulative",
        title="Total Uploads Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Heatmap")
    heatmap_df = (
        filtered_df.groupby(["Year", "Month_Name"])["Artworks_Uploaded"]
        .sum()
        .reset_index()
    )
    heatmap_df["Month_Name"] = pd.Categorical(
        heatmap_df["Month_Name"], categories=list(calendar.month_name)[1:], ordered=True
    )
    heatmap_df = heatmap_df.sort_values(["Year", "Month_Name"])
    fig = px.imshow(
        heatmap_df.pivot(
            index="Year", columns="Month_Name", values="Artworks_Uploaded"
        ),
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
<div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #666;">
    <hr style="margin-bottom: 0.5rem;">
    <div style="font-size: 0.9rem;">
        Powered by Streamlit • Data from Last.fm • Updated hourly
    </div>
</div>
""",
    unsafe_allow_html=True,
)
