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
from langchain.agents import Tool

# Streamlit UI setup
st.set_page_config(page_title="📊 Tech Trend Analyzer", layout="wide")
st.title("🧠 Tech News Trend Analyzer with Agents")

topic = st.text_input("🎯 Enter a technology topic", "AI")
hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
news_api_key = st.text_input("🗝️ NewsAPI Key", type="password")
run_button = st.button("🚀 Run Analysis")

# Define the tool function
def fetch_tech_news(topic: str) -> str:
    """Fetch recent tech news by topic using NewsAPI."""
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

# Create the Tool object properly
fetch_news_tool = Tool(
    name="fetch_tech_news",
    func=fetch_tech_news,
    description="Fetch recent tech news by topic from NewsAPI"
)

if run_button:
    if not hf_token or not news_api_key:
        st.error("❌ Please enter both Hugging Face and NewsAPI keys.")
        st.stop()

    with st.spinner("Running agents..."):
        try:
            llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
            )

            # Initialize agents with all required parameters
            fetcher = Agent(
                role="Tech News Fetcher",
                goal=f"Fetch the latest news about {topic} from reliable sources",
                backstory="You are an expert at finding and collecting recent tech news from various sources.",
                tools=[fetch_news_tool],
                verbose=True,
                llm=llm,
                allow_delegation=False
            )

            summarizer = Agent(
                role="Tech News Summarizer",
                goal="Summarize tech news articles while maintaining key technical details",
                backstory="You have a talent for distilling complex tech news into concise, informative summaries.",
                verbose=True,
                llm=llm,
                allow_delegation=False
            )

            trend_agent = Agent(
                role="Tech Trend Analyst",
                goal="Identify emerging trends and patterns in technology news",
                backstory="You specialize in spotting technological trends before they become mainstream.",
                verbose=True,
                llm=llm,
                allow_delegation=False
            )

            # Define tasks
            task1 = Task(
                description=f"Fetch the latest news articles about {topic} using available tools.",
                expected_output=f"A list of at least 5 recent news articles about {topic} with titles and descriptions.",
                agent=fetcher
            )

            task2 = Task(
                description="Analyze the fetched news and create a comprehensive summary highlighting the main points.",
                expected_output="A well-structured paragraph summarizing the key points from the news articles.",
                agent=summarizer,
                context=[task1]
            )

            task3 = Task(
                description="Analyze the news content to identify trending keywords and technologies.",
                expected_output="A list of top 10 trending keywords with brief explanations of why they're trending.",
                agent=trend_agent,
                context=[task1]
            )

            crew = Crew(
                agents=[fetcher, summarizer, trend_agent],
                tasks=[task1, task2, task3],
                verbose=2
            )

            result = crew.kickoff()

            st.success("✅ Analysis Complete!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 News Summary")
                st.write(task2.output)
                
            with col2:
                st.subheader("📈 Trending Keywords")
                try:
                    keywords = [re.sub(r"[-•]\s*", "", line.strip()) 
                               for line in str(task3.output).split("\n") if line.strip()]
                    df_keywords = pd.DataFrame({'Keyword': keywords[:10]})
                    fig = px.bar(df_keywords, x='Keyword', title="Top Trending Keywords", color='Keyword')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying trends: {e}")

            st.subheader("📰 Raw News Data")
            st.code(task1.output)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
