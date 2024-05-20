import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests as r
import regex as re
from dateutil import parser
import streamlit as st

rss = [
    "https://www.economictimes.indiatimes.com/rssfeedstopstories.cms",
    "http://feeds.feedburner.com/ndtvprofit-latest?format=xml",
    # "https://www.thehindubusinessline.com/news/feeder/default.rss",
    # "https://www.moneycontrol.com/rss/latestnews.xml",
    # "https://www.livemint.com/rss/news",
    # "https://www.financialexpress.com/feed/",
    # "https://www.business-standard.com/rss/latest.rss",
    # "https://www.businesstoday.in/rssfeeds/?id=225346",
    # "https://www.zeebiz.com/latest.xml/feed",
]


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
    # st.write(pd.DataFrame(
    #     {
    #         "title": title,
    #         "url": url,
    #         "description": desc,
    #         "date": date,
    #         "parsed_date": date1,
    #     },
    #     index=[0],
    # ))
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

#################################################
############# FRONT END HTML SCRIPT ##############
#################################################
result_str = '<html><table style="border: none;"><tr style="border: none;"><td style="border: none; height: 10px;"></td></tr>'
for n, i in final_df.iterrows():  # iterating through the search results
    href = i["url"]
    description = i["description"]
    url_txt = i["title"]
    src_time = i["src_time"]

    result_str += (
        f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: black; line-height: 1.2;">'
        + f'<tr style="align:justify; border-left: 5px solid transparent; border-top: 5px solid transparent; border-bottom: 5px solid transparent; font-weight: bold; font-size: 18px; background-color: whitesmoke;">{url_txt}</tr></a>'
        + f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: dimgray; line-height: 1.25;">'
        + f'<tr style="align:justify; border-left: 5px solid transparent; border-top: 0px; border-bottom: 5px solid transparent; font-size: 14px; padding-bottom:5px;">{description}</tr></a>'
        + f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: black;">'
        + f'<tr style="border-left: 5px solid transparent; border-top: 0px; border-bottom: 5px solid transparent; color: green; font-size: 11px;">{src_time}</tr></a>'
        + f'<tr style="border: none;"><td style="border: none; height: 10px;"></td></tr>'
    )

result_str += "</table></html>"

# HTML Script to hide Streamlit menu
# Reference: https://discuss.streamlit.io/t/how-do-i-hide-remove-the-menu-in-production/362/8
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .css-hi6a2p {padding-top: 0rem;}
            .css-1moshnm {visibility: hidden;}
            .css-kywgdc {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(result_str, unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
