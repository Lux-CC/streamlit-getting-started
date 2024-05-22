# What's New in Snowflake? - README

## Goal of the App

The "What's New in Snowflake?" app is designed to help users stay updated with the latest news and developments related to Snowflake. By aggregating news from various RSS feeds, the app provides a consolidated view of recent updates and allows users to interactively summarize articles and ask questions about the news content.

![Gif of usage](./docs/how-to.gif)



## High-Level Functionalities

### 1. News Aggregation
The app aggregates news from multiple RSS feeds related to Snowflake, including official Snowflake announcements and relevant AWS news. Users can select which feeds to include in their news feed.

### 2. Summarization
Users can summarize individual news articles to quickly grasp the essential information. This feature leverages advanced natural language processing (NLP) models to generate concise summaries.

### 3. Question Answering
The app allows users to ask specific questions about the news articles. Using a retrieval-augmented generation (RAG) approach, the app retrieves relevant documents and generates informative answers to user queries.

## How It Is Built

### Technology Stack

- **Streamlit**: The app is built using Streamlit, a popular framework for creating interactive web applications in Python. Streamlit provides a user-friendly interface for displaying news articles, summaries, and answers.
- **Natural Language Processing (NLP)**: The app uses several NLP libraries and models, including:
  - **Sentence Transformers**: For embedding news articles and queries to facilitate document retrieval.
  - **Replicate**: For generating summaries and answering questions using large language models (LLMs).
  - **Hugging Face Transformers**: For tokenizing text inputs to ensure compatibility with the models.

### Key Components

- **RSS Feed Aggregation**: The app fetches news articles from specified RSS feeds, parses the XML content, and processes the articles to clean and extract relevant information.
- **Document Embedding and Retrieval**: Using Sentence Transformers, the app embeds news articles and user queries into vector space, allowing for efficient retrieval of relevant documents based on semantic similarity.
- **Summarization and Answer Generation**: The app employs the Replicate API to generate summaries and answers. It uses a structured prompt to ensure that the model provides comprehensive and relevant responses.

### Caching
To enhance performance, the app utilizes Streamlit's caching mechanisms for storing transformer models, tokenizers, and fetched data. This reduces the need for redundant computations and speeds up the user experience.

## Usage

### Sidebar Configuration
Users can configure the app using the sidebar:
- Select the RSS feeds to include in the news feed.
- View additional information about the app.

### Main Interface
- **News Feed**: Displays a list of aggregated news articles, sorted by recency. Users can click on articles to view more details or summarize the content.
- **Ask a Question**: Users can input a question related to the news articles. The app retrieves relevant documents and generates an answer based on the content.

## Conclusion
The "What's New in Snowflake?" app is a powerful tool for staying informed about Snowflake's latest developments. By combining news aggregation, summarization, and question answering, the app offers a comprehensive and interactive way to keep up with the fast-paced world of Snowflake technology.