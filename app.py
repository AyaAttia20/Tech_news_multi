import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import requests
import pandas as pd
import re
import plotly.express as px

from crewai import Agent, Task, Crew
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

# Streamlit UI setup
st.set_page_config(page_title="📊 Tech Trend Analyzer", layout="wide")
st.title("🧠 Tech News Trend Analyzer with Agents")

topic = st.text_input("🎯 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ NewsAPI Key", type="password")
run_button = st.button("🚀 Run Analysis")

# Define the tool as a dict, NOT a Tool class (to avoid import errors)
def fetch_tech_news(topic: str) -> str:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": news_api_key,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return "Failed to fetch news."
    articles = response.json().get("articles", [])
    results = []
    for art in articles:
        title = art.get('title', 'No title')
        desc = art.get('description', 'No description')
        results.append(f"{title} - {desc}")
    return "\n".join(results)

fetch_news_tool = {
    "name": "fetch_tech_news",
    "func": fetch_tech_news,
    "description": "Fetch recent tech news by topic"
}

if run_button:
    if not hf_token or not news_api_key:
        st.error("❌ Please enter both Hugging Face and NewsAPI keys.")
        st.stop()

    with st.spinner("Running agents..."):
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_token,
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        fetcher = Agent(
            role="News Fetcher",
            goal="Get recent news about a topic",
            tools=[fetch_news_tool],  # list of dict tools
            verbose=True,
            llm=llm,
            memory=memory
        )

        summarizer = Agent(
            role="News Summarizer",
            goal="Summarize the key points of tech news",
            verbose=True,
            llm=llm,
            memory=memory
        )

        trend_agent = Agent(
            role="Trend Extractor",
            goal="Extract trending keywords from news content",
            verbose=True,
            llm=llm,
            memory=memory
        )

        task1 = Task(
            description=f"Fetch recent news about {topic}",
            expected_output="List of news headlines and descriptions",
            agent=fetcher
        )

        task2 = Task(
            description="Summarize the main points from the news.",
            expected_output="Concise summary paragraph",
            agent=summarizer,
            context=[task1]
        )

        task3 = Task(
            description="Extract the top trending keywords from the news headlines.",
            expected_output="A list of top 10 trending keywords",
            agent=trend_agent,
            context=[task1]
        )

        crew = Crew(
            agents=[fetcher, summarizer, trend_agent],
            tasks=[task1, task2, task3],
            verbose=True
        )

        result = crew.kickoff(inputs={"topic": topic})

        st.success("✅ Agents finished analysis!")

        st.subheader("🧠 Summary")
        st.write(task2.output)

        st.subheader("📈 Trending Keywords")
        keywords = [re.sub(r"[-•]\s*", "", line.strip()) for line in str(task3.output).split("\n") if line.strip()]
        df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
        fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🗞 Raw News Fetched")
        st.code(task1.output, language="markdown")
