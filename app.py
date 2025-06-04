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

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(page_title="📊 Tech Trend Analyzer", layout="wide")
st.title("🧠 Tech News Trend Analyzer with Agents")

topic = st.text_input("🎯 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ NewsAPI Key", type="password")
run_button = st.button("🚀 Run Analysis")

# ===============================
# News fetch function (no @tool)
# ===============================
def fetch_tech_news(topic: str, api_key: str) -> str:
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
        title = art.get('title', 'No title')
        desc = art.get('description', 'No description')
        results.append(f"{title} - {desc}")
    return "\n".join(results)

# ===============================
# Agents classes with internal calls
# ===============================
class FetcherAgent(Agent):
    def run(self, inputs):
        topic = inputs.get("topic")
        api_key = inputs.get("news_api_key")
        news = fetch_tech_news(topic, api_key)
        return news

class SummarizerAgent(Agent):
    def run(self, inputs):
        news = inputs.get("news")
        prompt = f"Please summarize the following tech news:\n{news}"
        summary = self.llm(prompt)
        return summary

class TrendAgent(Agent):
    def run(self, inputs):
        news = inputs.get("news")
        prompt = f"Extract top 10 trending keywords from this news:\n{news}"
        keywords = self.llm(prompt)
        return keywords

# ===============================
# Run app logic
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

        fetcher = FetcherAgent(role="News Fetcher", goal="Get recent news about a topic", verbose=True, llm=llm)
        summarizer = SummarizerAgent(role="News Summarizer", goal="Summarize the key points of tech news", verbose=True, llm=llm)
        trend_agent = TrendAgent(role="Trend Extractor", goal="Extract trending keywords from news content", verbose=True, llm=llm)

        # Step 1: fetch news
        news = fetcher.run({"topic": topic, "news_api_key": news_api_key})

        # Step 2: summarize news
        summary = summarizer.run({"news": news})

        # Step 3: extract trends
        keywords_raw = trend_agent.run({"news": news})

        st.success("✅ Agents finished analysis!")

        # Summary display
        st.subheader("🧠 Summary")
        st.write(summary)

        # Keywords display (حاول تنفذ تنظيف للkeywords)
        st.subheader("📈 Trending Keywords")
        keywords = [re.sub(r"[-•]\s*", "", line.strip()) for line in str(keywords_raw).split("\n") if line.strip()]
        df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
        fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
        st.plotly_chart(fig, use_container_width=True)

        # Raw news display
        st.subheader("🗞 Raw News Fetched")
        st.code(news, language="markdown")
