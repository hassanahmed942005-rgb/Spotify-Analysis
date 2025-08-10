import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import humanize
from PIL import Image
import requests
from io import BytesIO
import base64
import os

# -------------------- Data Loading --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("uu_preprocessed_analysis.csv")
        if df.empty:
            st.error("The CSV file is empty!")
            return pd.DataFrame()

        required_cols = ['song', 'artist']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing required columns: {missing_cols}")

        return df
    except FileNotFoundError:
        st.error("CSV file 'uu_preprocessed_analysis.csv' not found!")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty or corrupted!")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# -------------------- Image Functions --------------------
def load_local_image(image_path="qw.png"):
    """Load local image file (fixed to qw.png)"""
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        st.warning(f"Local image '{image_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading image '{image_path}': {str(e)}")
        return None

def get_artist_image_placeholder(artist_name):
    """Generate a placeholder image or fetch from a service"""
    local_img = load_local_image("qw.png")
    if local_img:
        return local_img

    # Fallback to online placeholder
    placeholder_url = f"https://ui-avatars.com/api/?name={artist_name.replace(' ', '+')}&size=200&background=random"
    try:
        response = requests.get(placeholder_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def display_artist_image(artist_name, size=(150, 150)):
    """Display artist image with fallback to placeholder"""
    img = get_artist_image_placeholder(artist_name)
    if img:
        img = img.resize(size)
        st.image(img, caption=artist_name, width=size[0])
    else:
        st.write(f"🎤 {artist_name}")

def display_custom_logo():
    """Display custom logo from qw.png"""
    logo_img = load_local_image("qw.png")
    if logo_img:
        logo_img = logo_img.resize((150, 150))
        st.image(logo_img, width=150)
        return True
    else:
        st.image("https://via.placeholder.com/150x150/1DB954/FFFFFF?text=🎵", width=150)
        return False

# -------------------- KPIs --------------------
@st.cache_data
def get_unique_artists(data):
    return sorted(data['artist'].unique())

@st.cache_data
def get_number_unique_genres(data):
    return len(data['genre'].unique())

@st.cache_data
def get_mean_duration_ms(data):
    duration_ms = data['duration_ms'].mean()
    duration_sec = duration_ms / 1000
    return humanize.precisedelta(duration_sec, minimum_unit="seconds", format="%0.2f")

@st.cache_data
def get_average_song_per_artist(data):
    return round(data.groupby("artist")["song"].count().mean())

# -------------------- Enhanced Filtering --------------------
def apply_filters(df, filters):
    try:
        filtered_df = df.copy()

        if filters.get('genres'):
            if 'genre' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['genre'].isin(filters['genres'])]

        if filters.get('year_range') and 'year' in filtered_df.columns:
            min_year, max_year = filters['year_range']
            filtered_df = filtered_df[(filtered_df['year'] >= min_year) & (filtered_df['year'] <= max_year)]

        if filters.get('popularity_range') and 'popularity' in filtered_df.columns:
            min_pop, max_pop = filters['popularity_range']
            filtered_df = filtered_df[(filtered_df['popularity'] >= min_pop) & (filtered_df['popularity'] <= max_pop)]

        if filters.get('duration_range') and 'duration_ms' in filtered_df.columns:
            min_dur, max_dur = filters['duration_range']
            filtered_df = filtered_df[(filtered_df['duration_ms'] >= min_dur) & (filtered_df['duration_ms'] <= max_dur)]

        if filters.get('artists'):
            if 'artist' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['artist'].isin(filters['artists'])]

        if filters.get('search_text'):
            search_text = filters['search_text'].lower()
            song_mask = pd.Series([False] * len(filtered_df))
            artist_mask = pd.Series([False] * len(filtered_df))

            if 'song' in filtered_df.columns:
                song_mask = filtered_df['song'].str.lower().str.contains(search_text, na=False)
            if 'artist' in filtered_df.columns:
                artist_mask = filtered_df['artist'].str.lower().str.contains(search_text, na=False)

            filtered_df = filtered_df[song_mask | artist_mask]

        audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
        for feature in audio_features:
            if filters.get(f'{feature}_range') and feature in filtered_df.columns:
                min_val, max_val = filters[f'{feature}_range']
                feature_mask = (filtered_df[feature].notna() &
                                (filtered_df[feature] >= min_val) &
                                (filtered_df[feature] <= max_val))
                filtered_df = filtered_df[feature_mask]

        return filtered_df

    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        return df.copy()

# -------------------- Charts --------------------
@st.cache_data
def plot_duration_by_genre(df):
    avg_duration = df.groupby('genre')['duration_ms'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_duration.index,
        y=avg_duration.values,
        title='Average Song Duration (ms) per Genre',
        labels={'x': 'Genre', 'y': 'Duration (ms)'},
        color=avg_duration.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig

@st.cache_data
def plot_popularity_by_year(df):
    avg_popularity = df.groupby('year')['popularity'].mean()
    fig = px.line(
        x=avg_popularity.index,
        y=avg_popularity.values,
        title='Average Popularity per Year',
        labels={'x': 'Year', 'y': 'Popularity'},
        markers=True
    )
    fig.update_traces(line_color='#1f77b4', line_width=3)
    return fig

@st.cache_data
def plot_genre_distribution(df):
    genre_counts = df['genre'].value_counts()
    fig = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title='Genre Distribution'
    )
    return fig

@st.cache_data
def plot_audio_features_heatmap(df):
    features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
    correlation_matrix = df[features].corr()

    fig = px.imshow(
        correlation_matrix,
        title='Audio Features Correlation Heatmap',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    return fig

# -------------------- Display --------------------
def display_song_card(song, show_image=True):
    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            if show_image:
                display_artist_image(song['artist'], (100, 100))

        with col2:
            st.markdown(f"### 🎵 {song['song']}")
            st.write(f"*🗣️ Artist:* {song['artist']}")
            st.write(f"*🎼 Genre:* {song['genre']}")
            st.write(f"*📅 Year:* {song['year']}")
            st.write(f"*🔥 Popularity:* {song['popularity']}")
            st.write(f"*⏱️ Duration:* {humanize.precisedelta(song['duration_ms']/1000, minimum_unit='seconds', format='%0.1f')}")

            features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
            feature_values = [song[f] for f in features]

            fig_mini = go.Figure(data=[
                go.Bar(x=features, y=feature_values, marker_color='lightblue')
            ])
            fig_mini.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=20, b=0),
                title="Audio Features",
                title_font_size=12
            )
            st.plotly_chart(fig_mini, use_container_width=True)

# -------------------- Recommendation --------------------
def get_recommended_songs(data, song_name, n_recommendations=3):
    selected_row = data[data['song'] == song_name].iloc[0]
    same_genre = data[data['genre'] == selected_row['genre']]
    rec = same_genre[same_genre['song'] != song_name]

    if not rec.empty:
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']

        for idx in rec.index:
            distance = sum((selected_row[f] - rec.loc[idx, f])**2 for f in features)
            rec.loc[idx, 'similarity_score'] = distance

        return rec.nsmallest(n_recommendations, 'similarity_score')
    return pd.DataFrame()

# -------------------- Polar Plot --------------------
def plot_scatter_polar(selected_song, recommendations, columns_to_display):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=selected_song[columns_to_display],
        theta=columns_to_display,
        fill='toself',
        name="Selected Song",
        line=dict(color='blue', width=3),
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))

    colors = ['red', 'green', 'orange']
    for i, (_, rec) in enumerate(recommendations.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=rec[columns_to_display],
            theta=columns_to_display,
            fill='toself',
            name=f"Recommendation {i+1}",
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=f'rgba({255 if colors[i%len(colors)]=="red" else 0}, {255 if colors[i%len(colors)]=="green" else 0}, 0, 0.1)'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Audio Features Comparison"
    )

    return fig

# -------------------- Streamlit Layout --------------------
st.set_page_config(page_title="🎵 Spotify Music Dashboard", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    display_custom_logo()
with col2:
    st.title("🎵 Spotify Music Dashboard")
    st.markdown("Discover, explore, and analyze your music data")

# -------------------- Sidebar Filters --------------------
st.sidebar.header("🔍 Filters & Search")
filters = {}

try:
    filters['search_text'] = st.sidebar.text_input("🔎 Search songs/artists:")

    if 'genre' in df.columns:
        all_genres = sorted(df['genre'].dropna().unique())
        filters['genres'] = st.sidebar.multiselect("🎼 Select Genres:", all_genres)
    else:
        filters['genres'] = []

    if 'year' in df.columns:
        year_min, year_max = int(df['year'].min()), int(df['year'].max())
        filters['year_range'] = st.sidebar.slider("📅 Year Range:", year_min, year_max, (year_min, year_max))
    else:
        filters['year_range'] = None

    if 'popularity' in df.columns:
        pop_min, pop_max = int(df['popularity'].min()), int(df['popularity'].max())
        filters['popularity_range'] = st.sidebar.slider("🔥 Popularity Range:", pop_min, pop_max, (pop_min, pop_max))
    else:
        filters['popularity_range'] = None

    if 'duration_ms' in df.columns:
        dur_min, dur_max = int(df['duration_ms'].min()), int(df['duration_ms'].max())
        filters['duration_range'] = st.sidebar.slider("⏱️ Duration Range (ms):", dur_min, dur_max, (dur_min, dur_max))
    else:
        filters['duration_range'] = None

    if 'artist' in df.columns:
        all_artists = sorted(df['artist'].dropna().unique())
        filters['artists'] = st.sidebar.multiselect("🗣️ Select Artists:", all_artists)
    else:
        filters['artists'] = []

    st.sidebar.subheader("🎚️ Audio Features")
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
    for feature in audio_features:
        if feature in df.columns and not df[feature].isna().all():
            feature_min, feature_max = float(df[feature].min()), float(df[feature].max())
            filters[f'{feature}_range'] = st.sidebar.slider(
                f"{feature.title()}:",
                feature_min, feature_max, (feature_min, feature_max),
                step=0.01
            )
        else:
            filters[f'{feature}_range'] = None

except Exception as e:
    st.sidebar.error(f"Error loading filters: {str(e)}")
    filters = {
        'search_text': '',
        'genres': [],
        'year_range': None,
        'popularity_range': None,
        'duration_range': None,
        'artists': []
    }

filtered_df = apply_filters(df, filters)

# -------------------- Main Dashboard --------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Songs", len(filtered_df))
col2.metric("Unique Artists", len(filtered_df['artist'].unique()) if not filtered_df.empty else 0)
col3.metric("Unique Genres", len(filtered_df['genre'].unique()) if not filtered_df.empty else 0)
col4.metric("Avg Duration", get_mean_duration_ms(filtered_df) if not filtered_df.empty else "N/A")
col5.metric("Avg Popularity", f"{filtered_df['popularity'].mean():.1f}" if not filtered_df.empty else "N/A")

if filtered_df.empty:
    st.warning("No songs match your current filters.")
else:
    st.write("---")
    st.subheader("📈 Analytics")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_duration_by_genre(filtered_df), use_container_width=True)
        st.plotly_chart(plot_genre_distribution(filtered_df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(plot_popularity_by_year(filtered_df), use_container_width=True)
        st.plotly_chart(plot_audio_features_heatmap(filtered_df), use_container_width=True)

    st.write("---")
    st.subheader("🎵 Song Browser")

    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox("Sort by:", ['popularity', 'year', 'duration_ms', 'song', 'artist'])
    with sort_col2:
        sort_ascending = st.checkbox("Ascending order")

    display_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)

    items_per_page = st.slider("Songs per page:", 5, 50, 10)
    total_pages = len(display_df) // items_per_page + (1 if len(display_df) % items_per_page > 0 else 0)

    if total_pages > 1:
        page = st.selectbox("Page:", range(1, total_pages + 1))
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_df = display_df.iloc[start_idx:end_idx]
    else:
        page_df = display_df

    for idx, (_, song) in enumerate(page_df.iterrows()):
        with st.expander(f"🎵 {song['song']} - {song['artist']}", expanded=False):
            display_song_card(song, show_image=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🎯 Get Recommendations", key=f"rec_btn_{idx}"):
                    recommendations = get_recommended_songs(filtered_df, song['song'], 3)
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} recommendations:")
                        for rec_idx, (_, rec) in enumerate(recommendations.iterrows()):
                            st.write(f"*{rec_idx + 1}.* {rec['song']} by {rec['artist']} (Similarity: {rec['similarity_score']:.3f})")

                        polar_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
                        if all(col in song.index for col in polar_cols):
                            fig = plot_scatter_polar(song, recommendations, polar_cols)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No similar songs found.")

            with col2:
                if st.button(f"📊 Detailed Analysis", key=f"analysis_btn_{idx}"):
                    st.write("*Detailed Audio Features:*")
                    for feature in audio_features:
                        if feature in song.index:
                            st.write(f"- {feature.title()}: {song[feature]:.3f}")

# -------------------- Footer --------------------
st.write("---")
st.markdown("Enhanced Music Dashboard with qw.png as the default image")
