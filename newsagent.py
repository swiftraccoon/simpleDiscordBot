from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv
import os
import logging

# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

load_dotenv(dotenv_path=".env", override=True)

# Use environment variable for model name
MODEL = os.environ.get("NEWS_MODEL", "mistral-nemo:12b-instruct-2407-fp16")

def create_news_agents(model=MODEL):
    """Create and return the news processing agents"""
    client = Swarm()
    
    search_agent = Agent(
        name="News Searcher",
        instructions="""
        You are a news search specialist. Your task is to:
        1. Search for the most relevant and recent news on the given topic
        2. Ensure the results are from reputable sources
        3. Return the raw search results in a structured format
        """,
        functions=[search_news],
        model=model
    )

    synthesis_agent = Agent(
        name="News Synthesizer",
        instructions="""
        You are a news synthesis expert. Your task is to:
        1. Analyze the raw news articles provided
        2. Identify the key themes and important information
        3. Combine information from multiple sources
        4. Create a comprehensive but concise synthesis
        5. Focus on facts and maintain journalistic objectivity
        6. Write in a clear, professional style
        Provide a 2-3 paragraph synthesis of the main points.
        """,
        model=model
    )

    summary_agent = Agent(
        name="News Summarizer",
        instructions="""
        You are an expert news summarizer combining AP and Reuters style clarity with digital-age brevity.

        Your task:
        1. Core Information:
           - Lead with the most newsworthy development
           - Include key stakeholders and their actions
           - Add critical numbers/data if relevant
           - Explain why this matters now
           - Mention immediate implications

        2. Style Guidelines:
           - Use strong, active verbs
           - Be specific, not general
           - Maintain journalistic objectivity
           - Make every word count
           - Explain technical terms if necessary

        Format: Create a single paragraph of 250-400 words that informs and engages.
        Pattern: [Major News] + [Key Details/Data] + [Why It Matters/What's Next]

        Focus on answering: What happened? Why is it significant? What's the impact?

        IMPORTANT: Provide ONLY the summary paragraph. Do not include any introductory phrases, 
        labels, or meta-text like "Here's a summary" or "In AP/Reuters style."
        Start directly with the news content.
        """,
        model=model
    )
    
    return client, search_agent, synthesis_agent, summary_agent

def search_news(topic):
    """Search for news articles using DuckDuckGo"""
    with DDGS() as ddg:
        results = ddg.text(f"{topic} news {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            news_results = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return news_results
        return f"No news found for {topic}."

def process_news(topic, progress_callback=None):
    """
    Process news on a given topic
    Args:
        topic (str): The news topic to process
        progress_callback (callable): Optional callback function to report progress
    Returns:
        tuple: (raw_news, synthesized_news, final_summary)
    """
    client, search_agent, synthesis_agent, summary_agent = create_news_agents()
    
    if progress_callback:
        progress_callback("🔍 Searching for news...")
    search_response = client.run(
        agent=search_agent,
        messages=[{"role": "user", "content": f"Find recent news about {topic}"}]
    )
    raw_news = search_response.messages[-1]["content"]
    
    if progress_callback:
        progress_callback("🔄 Synthesizing information...")
    synthesis_response = client.run(
        agent=synthesis_agent,
        messages=[{"role": "user", "content": f"Synthesize these news articles:\n{raw_news}"}]
    )
    synthesized_news = synthesis_response.messages[-1]["content"]
    
    LOGGER.info("Synthesized news length: %d characters.", len(synthesized_news))

    if progress_callback:
        progress_callback("📝 Creating summary...")
    summary_response = client.run(
        agent=summary_agent,
        messages=[{"role": "user", "content": f"Summarize this synthesis to under 600 characters:\n{synthesized_news}"}]
    )
    final_summary = summary_response.messages[-1]["content"]
    LOGGER.info("Final summary length: %d characters.", len(final_summary))
    return raw_news, synthesized_news, final_summary