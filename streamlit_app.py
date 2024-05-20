import os
import requests as r
import pandas as pd
import numpy as np
import regex as re
from bs4 import BeautifulSoup
from dateutil import parser
import streamlit as st
from transformers import AutoTokenizer
import replicate
from sentence_transformers import SentenceTransformer
from typing import List


# List of RSS feeds to fetch
rss_feeds = [
    "https://www.snowflake.com/feed/",
    "https://rss.aws-news.com/custom_feeds/FEzdG/rss",
]

# Set Replicate API token
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]


def get_top_5_documents(query, df):
    model = SentenceTransformer("snowflake/snowflake-arctic-embed-l")
    
    # Extracting the descriptions from the dataframe
    documents = df['description'].tolist()
    
    # Encode the query and documents
    query_embeddings = model.encode([query], prompt_name="query")
    document_embeddings = model.encode(documents)
    
    # Compute the scores
    scores = query_embeddings @ document_embeddings.T
    
    # Zip scores with documents and sort
    doc_score_pairs = list(zip(documents, scores[0]))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    # Get the top 5 documents
    top_5_doc_score_pairs = doc_score_pairs[:5]
    
    # Find the indices of the top 5 documents
    top_5_indices = [documents.index(doc) for doc, score in top_5_doc_score_pairs]
    
    # Select the top 5 rows from the dataframe
    top_5_df = df.iloc[top_5_indices]
    
    return top_5_df



@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """
    Get a tokenizer to ensure the text sent to the model is not too long.
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def get_num_tokens(prompt):
    """
    Get the number of tokens in a given prompt.

    Args:
        prompt (str): The input text to tokenize.

    Returns:
        int: The number of tokens in the prompt.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


def arctic_summary(text):
    """
    Generate a summary for the given text using the Arctic model.

    Args:
        text (str): The input text to summarize.

    Yields:
        str: The generated summary text.
    """
    # st.write(get_num_tokens(text))
    for event_index, event in enumerate(
        replicate.stream(
            "snowflake/snowflake-arctic-instruct",
            input={
                "prompt": text,
                "prompt_template": r"Note the following release webpage: {prompt}. Summarize the core content of this release article.",
                "max_new_tokens": 512,
            },
        )
    ):
        if (event_index + 0) % 50 == 0:
            if not check_safety(text):
                st.write("I cannot answer this question.")
        yield str(event)


def check_safety(text) -> bool:
    """
    Check the safety of the text.

    Args:
        text (str): The input text to check.

    Returns:
        bool: True if the text is safe, False otherwise.
    """
    # For now, always return True
    return True


def date_time_parser(dt):
    """
    Calculate the time elapsed since the news was published.

    Args:
        dt (str): The published date of the news item.

    Returns:
        int: The time elapsed in minutes.
    """
    return int(np.round((dt.now(dt.tz) - dt).total_seconds() / 60, 0))


def elapsed_time_str(mins):
    """
    Convert the elapsed time in minutes to a human-readable string.

    Args:
        mins (int): The time elapsed in minutes.

    Returns:
        str: The elapsed time as a human-readable string.
    """
    time_str = ""
    hours = int(mins / 60)
    days = np.round(mins / (60 * 24), 1)
    remaining_mins = int(mins - (hours * 60))

    if days >= 1:
        time_str = f"{days} days ago" if days != 1 else "a day ago"
    elif (days < 1) and (hours < 24) and (mins >= 60):
        time_str = f"{hours} hours and {remaining_mins} mins ago"
        if (hours == 1) and (remaining_mins > 1):
            time_str = f"an hour and {remaining_mins} mins ago"
        if (hours == 1) and (remaining_mins == 1):
            time_str = f"an hour and a min ago"
        if (hours > 1) and (remaining_mins == 1):
            time_str = f"{hours} hours and a min ago"
        if (hours > 1) and (remaining_mins == 0):
            time_str = f"{hours} hours ago"
        if ((mins / 60) == 1) and (remaining_mins == 0):
            time_str = "an hour ago"
    elif (days < 1) and (hours < 24) and (mins == 0):
        time_str = "Just in"
    else:
        time_str = f"{mins} minutes ago"
        if mins == 1:
            time_str = "a minute ago"
    return time_str


def text_clean(desc):
    """
    Clean the text by removing unparsed HTML characters.

    Args:
        desc (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    desc = desc.replace("&lt;", "<")
    desc = desc.replace("&gt;", ">")
    desc = re.sub("<.*?>", "", desc)
    desc = desc.replace("#39;", "'")
    desc = desc.replace("&quot;", '"')
    desc = desc.replace("&nbsp;", '"')
    desc = desc.replace("#32;", " ")
    return desc


def src_parse(rss):
    """
    Extract the source domain from the RSS feed URL.

    Args:
        rss (str): The RSS feed URL.

    Returns:
        str: The source domain.
    """
    if rss.find("ndtvprofit") >= 0:
        return "ndtv profit"
    rss = rss.replace("https://www.", "")
    rss = rss.split("/")
    return rss[0]


def rss_parser(item):
    """
    Process an individual news item.

    Args:
        item (bs4.element.Tag): A single news item (<item>) of an RSS Feed.

    Returns:
        DataFrame: A data frame containing the processed news item.
    """
    b1 = BeautifulSoup(str(item), "xml")
    title = "" if b1.find("title") is None else b1.find("title").get_text()
    title = text_clean(title)
    url = "" if b1.find("link") is None else b1.find("link").get_text()
    desc = "" if b1.find("description") is None else b1.find("description").get_text()
    desc = text_clean(desc)
    desc = f"{desc[:300]}..." if len(desc) >= 300 else desc
    date = (
        "Sat, 12 Aug 2000 13:39:15 +0530"
        if b1.find("pubDate") is None
        else b1.find("pubDate").get_text()
    )
    if url.find("businesstoday.in") >= 0:
        date = date.replace("GMT", "+0530")
    date1 = parser.parse(date)
    return pd.DataFrame(
        {
            "title": title,
            "url": url,
            "description": desc,
            "date": date,
            "parsed_date": date1,
        },
        index=[0],
    )


def news_agg(rss):
    """
    Process each RSS Feed URL.

    Args:
        rss (str): The RSS feed URL.

    Returns:
        DataFrame: A data frame containing the processed data from the RSS feed.
    """
    try:
        resp = r.get(rss, headers={"user-agent": "Mozilla/5.0"})
        resp.raise_for_status()
        b = BeautifulSoup(resp.content, "xml")
        items = b.find_all("item")
        rss_data = [rss_parser(i) for i in items]
        if not rss_data:
            return pd.DataFrame()
        rss_df = pd.concat(rss_data, ignore_index=True)
        rss_df["description"] = rss_df["description"].replace([" NULL", ""], np.nan)
        rss_df.dropna(inplace=True)
        rss_df["src"] = src_parse(rss)
        rss_df["elapsed_time"] = rss_df["parsed_date"].apply(date_time_parser)
        rss_df["elapsed_time_str"] = rss_df["elapsed_time"].apply(elapsed_time_str)
        return rss_df
    except r.exceptions.RequestException as e:
        st.error(f"Error fetching RSS feed: {e}")
        return pd.DataFrame()


def summarize_article(paragraph_list):
    """
    Summarize the article from a list of paragraphs.

    Args:
        paragraph_list (list): List of paragraphs from the article.

    Returns:
        str: The summary of the article.
    """
    text_to_summarize = ""
    summary_tokens = []
    num_tokens = 0
    # st.write(paragraph_list)
    for paragraph in paragraph_list:
        num_tokens += get_num_tokens(paragraph)
        # st.write(paragraph)
        if num_tokens > 1500:
            summary_tokens.extend(
                [token for token in arctic_summary(text_to_summarize)]
            )
            text_to_summarize = paragraph
            num_tokens = get_num_tokens(paragraph)
        else:
            text_to_summarize += paragraph
    summary_tokens.extend([token for token in arctic_summary(text_to_summarize)])
    # st.write(summary_tokens)
    total_summary_tokens = arctic_summary("".join(summary_tokens))
    return "".join([token for token in total_summary_tokens])


def fetch_webpage_summary(url):
    """
    Fetch and summarize the webpage content.

    Args:
        url (str): The URL of the webpage.

    Returns:
        str: The summary of the webpage content.
    """
    try:
        response = r.get(url, headers={"user-agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = [para.get_text() for para in paragraphs if para.get_text().strip()]
        if not content:
            st.error("No content found to summarize.")
            return ""
        summary = summarize_article(content)
        return summary
    except r.exceptions.RequestException as e:
        st.error(f"Error fetching webpage content: {e}")
        return ""

def show_news(news_df):
    """
    Show the news feed in a table.

    Args:
        news_df (DataFrame): The data frame containing the news feed.
    """
    for n, i in news_df.iterrows():
        href = i["url"]
        description = i["description"]
        url_txt = i["title"]
        src_time = i["src_time"]
    
        st.markdown(
            f"""
        <div style="border:1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h3 style="margin-bottom: 5px;"><a href="{href}" target="_blank" style="text-decoration: none; color: #007bff;">{url_txt}</a></h3>
            <p style="margin-bottom: 5px;">{description}</p>
            <p style="color: #6c757d; font-size: 0.9em;">{src_time}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        # Add a button to summarize the article
        if st.button(f"Summarize Article {n+1}"):
            summary = fetch_webpage_summary(href)
            st.write(f"**Summary:** {summary}")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="RSS Feed Summarizer")
    st.title("RSS Feed Summarizer")

    query = st.text_input("Ask a question about the news")

    # Initialize empty dataframe to store all news
    all_news = pd.DataFrame()

    for feed in rss_feeds:
        feed_data = news_agg(feed)
        all_news = pd.concat([all_news, feed_data], ignore_index=True)
    
    all_news.sort_values(by="elapsed_time", inplace=True)
    all_news["src_time"] = all_news["src"] + ("&nbsp;" * 5) + all_news["elapsed_time_str"]


    if not all_news.empty:
        st.subheader("Aggregated News Feed")
        show_news(all_news)
        if query:
            scores = get_top_5_documents(query, all_news)
            st.subheader("Answer")
            st.write(scores)

    else:
        st.write("No news available from the provided RSS feeds.")


if __name__ == "__main__":
    main()
