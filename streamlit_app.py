import os
import requests as r
import pandas as pd
import numpy as np
import regex as re
import streamlit as st
import logging
import pandas as pd
import plotly.graph_objects as go
logger = logging.getLogger(__name__)

# Set Replicate API token
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]


def display_sidebar_ui():
    with st.sidebar:
        st.title("Configuration")
        # colors = {
        #     "https://www.snowflake.com/feed/": "Snowflake Official",
        #     "https://rss.aws-news.com/custom_feeds/FEzdG/rss": "AWS Snowflake News",
        # }

        # # Get the list of feed URLs and labels
        # rss_feeds = list(rss_feed_labels.keys())
        # feed_labels = list(rss_feed_labels.values())

        # # Create a dictionary to map labels back to their URLs for later use
        # label_to_url = {label: url for url, label in rss_feed_labels.items()}

        # Get HEX color
        selected_color = st.color_picker("Select color")

        # # Convert selected labels back to their URLs
        st.session_state.color = selected_color

        # st.subheader("About")
        # st.caption(
        #     "Hi there! I hope this app helps you catch up with the latest news of snowflake. Note that performance can be greatly improved still, but I consider this the MVP. Have fun!"
        # )


def generate_sankey(df):
    
    # Prepare unique values and mapping
    unique_cols = [df[col].unique() for col in df.columns]
    all_values = [item for sublist in unique_cols for item in sublist]

    node_labels = list(pd.unique(all_values))
    label_to_index = {label: i for i, label in enumerate(node_labels)}
    
    # Create links
    links = []
    for _, row in df.iterrows():
        for i in range(len(row) - 1):
            links.append({'source': label_to_index[row[i]],
                          'target': label_to_index[row[i + 1]],
                          'value': 1})
    
    # Prepare data for the Sankey diagram
    sankey_data = {
        'node': {
            'label': node_labels
        },
        'link': {
            'source': [link['source'] for link in links],
            'target': [link['target'] for link in links],
            'value': [link['value'] for link in links],
        }
    }
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['node']['label']
        ),
        link=dict(
            source=sankey_data['link']['source'],
            target=sankey_data['link']['target'],
            value=sankey_data['link']['value']
        )
    ))
    
    # Update layout and show figure
    fig.update_layout(title_text="Sankey Diagram", font_size=10)

    # output the figure to streamlit
    st.plotly_chart(fig, use_container_width=True)



def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Generate sankey for Milou!")
    st.title("Generate sankey!")
    display_sidebar_ui()

    st.session_state.sankey_data = pd.DataFrame()

    file_uploaded = st.file_uploader('Upload csv', type=['csv'], accept_multiple_files=False)
    if file_uploaded is not None:
        st.write("Found columns: ")
        for col in df.columns:
            st.write(col)
        df = pd.read_csv(file_uploaded)
        # Initialize empty dataframe to store all news
        st.session_state.sankey_data = df


    if not st.session_state.sankey_data.empty:
        # show button to generate sankey
        if st.button("Generate sankey"):
            generate_sankey(st.session_state.data)


if __name__ == "__main__":
    main()
