# tarotagent.py

import random
from swarm import Swarm, Agent
from dotenv import load_dotenv
import os

# Load only the main .env file, not .env.example
load_dotenv(dotenv_path=".env", override=True)

# Use environment variable for model name
MODEL = os.environ.get("TAROT_MODEL", "phi4:14b-q8_0")

MAJOR_ARCANA = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World"
]

def create_tarot_agents(model=MODEL):
    """
    Create a Swarm client with two agents:
      1) 'Tarot Reader' to produce the main reading
      2) 'Tarot Summarizer' to ensure the reading is under 600 characters if needed.
    """
    client = Swarm()

    tarot_agent = Agent(
        name="Tarot Reader",
        instructions="""
        You are a Tarot reading specialist. Given the user has drawn a specific card
        from the Major Arcana, you will provide a 1 paragraph Tarot reading,
        focusing on the card's symbolism, meaning, guidance, and how it may apply
        to the user's life situation.

        Requirements:
        1. The reading must be under 600 characters.
        2. Emphasize unique insights about the drawn card.
        3. Maintain a clear, friendly tone without extraneous fluff.
        4. Provide genuine guidance based on the card's traditional interpretation.
        """,
        model=model
    )

    summarizer_agent = Agent(
        name="Tarot Summarizer",
        instructions="""
        You are a Tarot reading summarizer. The user has provided a reading that
        may exceed 600 characters. Your job is to refine or shorten the text so that 
        the final result is under 600 characters, while preserving the most important 
        insights and guidance. Maintain a friendly, clear tone and respect 
        traditional Tarot meanings. Omit any fluff or repetition.
        """,
        model=model
    )

    return client, tarot_agent, summarizer_agent

def generate_tarot_reading(progress_callback=None) -> str:
    """
    Select a random card from the Major Arcana, then run the Tarot Reader agent
    to produce a reading. If the reading is over 600 characters, pass it to the
    Tarot Summarizer agent for final compression. Optionally accepts a
    progress_callback to update logs about steps.
    """
    client, tarot_agent, summarizer_agent = create_tarot_agents()

    card = random.choice(MAJOR_ARCANA)
    if progress_callback:
        progress_callback(f"Shuffling deck... (Your card: {card})")

    # Send a user message instructing the agent to produce a reading under 600 characters:
    messages = [
        {
            "role": "user",
            "content": (
                f"You have drawn the Tarot card '{card}'. "
                "Please provide a short reading (under 600 characters total) "
                "focusing on symbolism, meaning, and guidance."
            )
        }
    ]

    # 1) Run the Tarot Reader agent
    reader_response = client.run(agent=tarot_agent, messages=messages)
    reading_text = reader_response.messages[-1]["content"]

    if progress_callback:
        progress_callback(f"Initial reading length: {len(reading_text)} characters.")

    # 2) Check character count
    if len(reading_text) > 600:
        if progress_callback:
            progress_callback(f"Reading is {len(reading_text)} characters; summarizing...")

        # Summarize with the Tarot Summarizer agent
        summarize_messages = [
            {
                "role": "user",
                "content": (
                    "Here is a Tarot reading that exceeds 600 characters:\n"
                    f"{reading_text}\n\n"
                    "Please shorten it to under 600 characters while preserving key points."
                )
            }
        ]
        summary_response = client.run(agent=summarizer_agent, messages=summarize_messages)
        reading_text = summary_response.messages[-1]["content"]

        if progress_callback:
            progress_callback(f"Summarized reading length: {len(reading_text)} characters.")

    return reading_text
