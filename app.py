
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from youtube_comment_downloader import YoutubeCommentDownloader
from googleapiclient.discovery import build
import streamlit as st

# UI: Set the page configuration for a wider layout and a nice title/icon
st.set_page_config(layout="wide", page_title="YouTube Comment Analyzer", page_icon="📊")

# --- Setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download("vader_lexicon")

# --- IMPORTANT: REPLACE WITH YOUR YOUTUBE API KEY ---
API_KEY = "AIzaSyBwkbiZVISIbi4I2Lbq3UT8e7VydkpjgmI"

youtube = build("youtube", "v3", developerKey=API_KEY)
downloader = YoutubeCommentDownloader()

# --- Functions ---
def extract_video_id(url):
    match = re.search(r"(?:v=|/embed/|/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_video_info(video_id):
    try:
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()
        if not response["items"]:
            return None
        video = response["items"][0]
        # UI: Format the date to be more readable
        publish_date = pd.to_datetime(video["snippet"]["publishedAt"]).strftime('%b %d, %Y')
        return {
            "Channel Name": video["snippet"]["channelTitle"],
            "Video Title": video["snippet"]["title"],
            "Date Posted": publish_date,
            "Likes": int(video["statistics"].get("likeCount", 0)),
            "Comments Count": int(video["statistics"].get("commentCount", 0)),
        }
    except Exception as e:
        st.error(f"Error fetching video info: {e}")
        return None

def analyze_sentiment(comments_df):
    sia = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        score = sia.polarity_scores(str(text))['compound']
        if score >= 0.05: return "Positive"
        elif score <= -0.05: return "Negative"
        else: return "Neutral"
    comments_df["Sentiment"] = comments_df["Comment"].apply(get_sentiment)
    return comments_df

# --- Streamlit UI ---
# UI: Use a more descriptive title with an icon
st.title("📈 YouTube Comment Sentiment Analyzer")
st.markdown("---")

# UI: Add a placeholder to guide the user
url = st.text_input("Enter the YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if st.button("Analyze Comments"):
    if not url:
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid or unsupported YouTube URL. Please enter a valid video URL.")
        else:
            with st.spinner("🚀 Fetching comments and running analysis... This might take a moment!"):
                info = get_video_info(video_id)

                try:
                    # The fixed line from our previous conversation
                    comments_generator = downloader.get_comments_from_url(url)
                    comments_list = [c["text"] for c in comments_generator]
                    df = pd.DataFrame({"Comment": comments_list})
                    df = analyze_sentiment(df)
                except Exception as e:
                    st.error(f"Could not fetch comments. The video may have comments disabled. Error: {e}")
                    st.stop()

                if info:
                    # UI: Display key video stats using st.metric in columns for a dashboard feel
                    st.header(f"Results for: \"{info['Video Title']}\"")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Channel", info['Channel Name'])
                    col2.metric("Total Likes 👍", f"{info['Likes']:,}")
                    col3.metric("Total Comments 💬", f"{info['Comments Count']:,}")
                    col4.metric("Publish Date 🗓️", info['Date Posted'])
                    st.markdown("---")

                st.header("Sentiment Analysis Breakdown")

                # UI: Use columns to place the pie chart and word cloud side-by-side
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df["Sentiment"].value_counts()

                    # Pie chart
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%",
                           startangle=90, colors=['#66b3ff','#ff9999','#99ff99'])
                    ax.axis("equal")
                    st.pyplot(fig)

                with col2:
                    st.subheader("Most Common Words")
                    all_comments_text = " ".join(str(c) for c in df["Comment"])
                    if all_comments_text:
                        wordcloud = WordCloud(width=800, height=600, background_color="white", colormap='viridis').generate(all_comments_text)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wordcloud, interpolation="bilinear")
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)

                st.markdown("---")
                # UI: Put the raw data table inside an expander so it doesn't clutter the view
                with st.expander("📂 View Raw Comments and Sentiments"):
                    st.dataframe(df)
