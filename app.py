from crewai import CrewAgent  # Updated import

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

    
        fetcher = CrewAgent(
            role="News Fetcher",
            goal="Get recent news about a topic",
            tools=[fetch_news_tool],
            verbose=True,
            llm=llm
        )

        summarizer = CrewAgent(
            role="News Summarizer",
            goal="Summarize the key points of tech news",
            verbose=True,
            llm=llm
        )

        trend_agent = CrewAgent(
            role="Trend Extractor",
            goal="Extract trending keywords from news content",
            verbose=True,
            llm=llm
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
