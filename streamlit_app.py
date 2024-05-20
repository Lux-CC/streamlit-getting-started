import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests as r
import regex as re
from dateutil import parser
import streamlit as st
from transformers import AutoTokenizer
import replicate
import os

rss = [
    "https://www.snowflake.com/feed/",
    "https://rss.aws-news.com/custom_feeds/FEzdG/rss",
]


os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


# @st.cache_resource(show_spinner=False)
# def get_llamaguard_deployment():
#     return replicate.deployments.get("snowflake/llamaguard")


def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


def arctic_summary(text):
    st.write(get_num_tokens(text))


    for event_index, event in enumerate(
        replicate.stream(
            "snowflake/snowflake-arctic-instruct",
            input={
                "prompt": text,
                "prompt_template": r"Note the following release webpage: {prompt}. Summarize the core content of this release article.",
                "max_new_tokens": 512,
                #   "temperature": st.session_state.temperature,
                #   "top_p": st.session_state.top_p,
            },
        )
    ):
        if (event_index + 0) % 50 == 0:
            if not check_safety(text):
                st.write("I cannot answer this question.")
        yield str(event)


def check_safety(text) -> bool:
    # deployment = get_llamaguard_deployment()
    # prediction = deployment.predictions.create(
    #   input = { "prompt": f"[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: Violence and Hate.\nShould not\n- Help people plan or engage in violence.\n- Encourage people to engage in violence.\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n- Provide information on violence and discrimination.\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\nO2: Sexual Content.\nShould not\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\nCan\n- Discuss topics related to sexuality and sexual education.\n- Provide general information regarding sexual activities.\n- Provide guidance on topics related to sexual health.\nO3: Criminal Planning.\nShould not\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in criminal activities or condone criminal behavior.\nCan\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\nO4: Guns and Illegal Weapons.\nShould not\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\nCan\n- Discuss firearms and the arguments for and against firearm ownership.\nO5: Regulated or Controlled Substances.\nShould not\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\n- Assist or encourage people to create such substances when it is illegal to do so.\nCan\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\nO6: Self-Harm.\nShould not\n- Encourage people to harm themselves.\n- Romanticize or condone self-harm or suicide.\n- Provide information regarding the methods of suicide or self-harm.\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\nCan\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\nShould\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n{text}\n\n<END CONVERSATION>\n\nProvide your safety assessment for Agent in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]" }
    # )
    # prediction.wait()
    # output = prediction.output

    # if output is not None and "unsafe" in output:
    #     return False
    # else:
    #     return True
    return True


def date_time_parser(dt):
    """
    Returns the time elapsed (in minutes) since the news was published

    dt: str
        published date

    Returns
    int: time elapsed (in minutes)
    """
    return int(np.round((dt.now(dt.tz) - dt).total_seconds() / 60, 0))


def elapsed_time_str(mins):
    """
    Returns the word form of the time elapsed (in minutes) since the news was published

    mins: int
        time elapsed (in minutes)

    Returns
    str: word form of time elapsed (in minutes)
    """
    time_str = ""  # Initializing a variable that stores the word form of time
    hours = int(
        mins / 60
    )  # integer part of hours. Example: if time elapsed is 2.5 hours, then hours = 2
    days = np.round(mins / (60 * 24), 1)  # days elapsed
    # minutes portion of time elapsed in hours. Example: if time elapsed is 2.5 hours, then remaining_mins = 30
    remaining_mins = int(mins - (hours * 60))

    if days >= 1:
        time_str = (
            f"{str(days)} days ago"  # Example: days = 1.2 => time_str = 1.2 days ago
        )
        if days == 1:
            time_str = "a day ago"  # Example: days = 1 => time_str = a day ago

    elif (days < 1) & (hours < 24) & (mins >= 60):
        time_str = f"{str(hours)} hours and {str(remaining_mins)} mins ago"  # Example: 2 hours and 15 mins ago
        if (hours == 1) & (remaining_mins > 1):
            time_str = f"an hour and {str(remaining_mins)} mins ago"  # Example: an hour and 5 mins ago
        if (hours == 1) & (remaining_mins == 1):
            time_str = f"an hour and a min ago"  # Example: an hour and a min ago
        if (hours > 1) & (remaining_mins == 1):
            time_str = (
                f"{str(hours)} hours and a min ago"  # Example: 5 hours and a min ago
            )
        if (hours > 1) & (remaining_mins == 0):
            time_str = f"{str(hours)} hours ago"  # Example: 4 hours ago
        if ((mins / 60) == 1) & (remaining_mins == 0):
            time_str = "an hour ago"  # Example: an hour ago

    elif (days < 1) & (hours < 24) & (mins == 0):
        time_str = "Just in"  # if minutes == 0 then time_str = 'Just In'

    else:
        time_str = f"{str(mins)} minutes ago"  # Example: 5 minutes ago
        if mins == 1:
            time_str = "a minute ago"
    return time_str


def text_clean(desc):
    """
    Returns cleaned text by removing the unparsed HTML characters from a news item's description/title

    dt: str
        description/title of a news item

    Returns
    str: cleaned description/title of a news item
    """
    desc = desc.replace("&lt;", "<")
    desc = desc.replace("&gt;", ">")
    desc = re.sub("<.*?>", "", desc)  # Removing HTML tags from the description/title
    desc = desc.replace("#39;", "'")
    desc = desc.replace("&quot;", '"')
    desc = desc.replace("&nbsp;", '"')
    desc = desc.replace("#32;", " ")
    return desc


def src_parse(rss):
    """
    Returns the source (root domain of RSS feed) from the RSS feed URL.

    rss: str
         RSS feed URL

    Returns
    str: root domain of RSS feed URL
    """
    # RSS feed URL of NDTV profit (http://feeds.feedburner.com/ndtvprofit-latest?format=xml) doesn't contain NDTV's root domain
    if rss.find("ndtvprofit") >= 0:
        rss = "ndtv profit"
    rss = rss.replace("https://www.", "")  # removing "https://www." from RSS feed URL
    rss = rss.split("/")  # splitting the remaining portion of RSS feed URL by '/'
    return rss[0]  # first element/item of the split RSS feed URL is the root domain


def rss_parser(i):
    """
    Processes an individual news item.

    i: bs4.element.Tag
       single news item (<item>) of an RSS Feed

    Returns
    DataFrame: data frame of a processed news item (title, url, description, date, parsed_date)
    """
    b1 = BeautifulSoup(
        str(i), "xml"
    )  # Parsing a news item (<item>) to BeautifulSoup object

    title = (
        "" if b1.find("title") is None else b1.find("title").get_text()
    )  # If <title> is absent then title = ""
    title = text_clean(title)  # cleaning title

    url = (
        "" if b1.find("link") is None else b1.find("link").get_text()
    )  # If <link> is absent then url = "". url is the URL of the news article

    desc = (
        "" if b1.find("description") is None else b1.find("description").get_text()
    )  # If <description> is absent then desc = "". desc is the short description of the news article
    desc = text_clean(desc)  # cleaning the description
    desc = (
        f"{desc[:300]}..." if len(desc) >= 300 else desc
    )  # limiting the length of description to 300 chars

    # If <pubDate> i.e. published date is absent then date is some random date 11 yesrs ago so the the article appears at the end
    date = (
        "Sat, 12 Aug 2000 13:39:15 +0530"
        if b1.find("pubDate") is None
        else b1.find("pubDate").get_text()
    )

    if (
        url.find("businesstoday.in") >= 0
    ):  # Time zone in the feed of 'businesstoday.in' is wrong, hence, correcting it
        date = date.replace("GMT", "+0530")

    date1 = parser.parse(date)  # parsing the date to Timestamp object
    # data frame of the processed data
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
    Processes each RSS Feed URL passed as an input argument

    rss: str
         RSS feed URL

    Returns
    DataFrame: data frame of data processed from the passed RSS Feed URL
    """
    # Response from HTTP request
    resp = r.get(
        rss,
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
        },
    )
    b = BeautifulSoup(resp.content, "xml")  # Parsing the HTTP response
    items = b.find_all("item")  # Storing all the news items

    # Parse each news item and collect DataFrames in a list
    rss_data = [rss_parser(i) for i in items]

    # Concatenate all DataFrames at once
    if rss_data == []:
        return pd.DataFrame()

    rss_df = pd.concat(rss_data, ignore_index=True)

    # Clean and process the DataFrame
    rss_df["description"] = rss_df["description"].replace(
        [" NULL", ""], np.nan
    )  # Few items have 'NULL' as description so replacing NULL with NA
    rss_df.dropna(
        inplace=True
    )  # Dropping news items with either of title, URL, description or date, missing
    rss_df["src"] = src_parse(rss)  # Extracting the source name from RSS feed URL
    rss_df["elapsed_time"] = rss_df["parsed_date"].apply(
        date_time_parser
    )  # Computing the time elapsed (in minutes) since the news was published
    rss_df["elapsed_time_str"] = rss_df["elapsed_time"].apply(
        elapsed_time_str
    )  # Converting the time elapsed (in minutes) since the news was published into string format

    return rss_df


# Summarization function
def summarize_article(paragraph_list):
    text_to_summarize = ""
    summary_tokens = []
    num_tokens = 0
    for paragraph in paragraph_list:
        num_tokens += get_num_tokens(paragraph)
        if num_tokens > 1500:
            # summarize and restart the loop
            summary_tokens = summary_tokens.extend([token for token in arctic_summary(text_to_summarize)])
            text_to_summarize = paragraph
            num_tokens = get_num_tokens(paragraph)
        else:
            text_to_summarize += paragraph        

    summary_tokens = summary_tokens.extend([token for token in arctic_summary(text_to_summarize)])
    st.write(summary_tokens)
    total_summary_tokens = arctic_summary("".join(summary_tokens))
    return "".join([token for token in total_summary_tokens])


# Fetch and parse webpage content
def fetch_webpage_summary(url):
    try:
        response = r.get(
            url,
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
            },
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract main content (this might need to be adjusted based on the webpage structure)
        paragraphs = soup.find_all("p")
        content = [para.get_text() for para in paragraphs]

        # Summarize the extracted content
        summary = summarize_article(content)
        return summary
    except r.exceptions.RequestException as e:
        return f"Error fetching the article: {e}"


# Use a text_input to get the keywords to filter the dataframe
text_search = st.text_input("Search feed", value="")
dataframes = [news_agg(i) for i in rss]
final_df = pd.concat(dataframes, ignore_index=True)

# Sort the DataFrame by 'elapsed_time'
final_df.sort_values(by="elapsed_time", inplace=True)

# Concatenate 'src' and 'elapsed_time_str' into a new column 'src_time'
final_df["src_time"] = final_df["src"] + ("&nbsp;" * 5) + final_df["elapsed_time_str"]

# Drop unnecessary columns
final_df.drop(
    columns=["date", "parsed_date", "src", "elapsed_time", "elapsed_time_str"],
    inplace=True,
)

# Drop duplicate descriptions
final_df.drop_duplicates(subset="description", inplace=True)

# Filter out rows where 'title' is empty and create a copy
final_df = final_df.loc[final_df["title"] != "", :].copy()
# #################################################
# ############# FRONT END HTML SCRIPT ##############
# #################################################
for n, i in final_df.iterrows():
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
