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
from langchain.tools import tool

# ===============================
# 🔧 Streamlit Config
# ===============================
st.set_page_config(page_title="📊 Tech Trend Analyzer", layout="wide")
st.title("🧠 Tech News Trend Analyzer with Agents")

topic = st.text_input("🎯 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ NewsAPI Key", type="password")
run_button = st.button("🚀 Run Analysis")

# ===============================
# 🧠 News Fetcher Tool (expects a dict input)
# ===============================
@tool
def fetch_tech_news(args: dict) -> str:
    topic = args.get("topic")
    api_key = args.get("api_key")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": api_key,
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
        results.append(f"{art.get('title')} - {art.get('description')}")
    return "\n".join(results)

# ===============================
# 🚀 App Logic
# ===============================
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

        fetcher = Agent(
            role="News Fetcher",
            goal="Get recent news about a topic",
            tools=[fetch_tech_news],
            verbose=True,
            llm=llm
        )

        summarizer = Agent(
            role="News Summarizer",
            goal="Summarize the key points of tech news",
            verbose=True,
            llm=llm
        )

        trend_agent = Agent(
            role="Trend Extractor",
            goal="Extract trending keywords from news content",
            verbose=True,
            llm=llm
        )

        # Task 1: fetch news with dict input for fetch_tech_news
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

        # kickoff with proper input dictionary matching fetch_tech_news args
        result = crew.kickoff(inputs={"topic": topic, "api_key": news_api_key})

        st.success("✅ Agents finished analysis!")

        # 🧠 Summary
        st.subheader("🧠 Summary")
        st.write(task2.output)

        # 📈 Keywords
        st.subheader("📈 Trending Keywords")
        keywords = [re.sub(r"[-•]\s*", "", line.strip()) for line in str(task3.output).split("\n") if line.strip()]
        df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
        fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
        st.plotly_chart(fig, use_container_width=True)

        # 🗞 Raw News
        st.subheader("🗞 Raw News Fetched")
        st.code(task1.output, language="markdown")
