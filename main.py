# pyre-strict

"""
A Discord bot that can:
  1) Query a local LLM instance (Ollama) via `.llm`
  2) Generate images via ComfyUI using `.img`
  3) Provide news analysis via a multi-agent approach (newsagent.py) using `.news`
  4) Display voice channel join/leave notices in a dedicated notification channel, 
     excluding certain voice channels read from env
  5) Provide a Tarot reading (tarotagent.py) using `.tarot`
  6) Analyze images using Ollama's vision model via `.vision`
  7) Chat with conversation memory using `.chat`

Usage:
    1. Install dependencies:
       pip install discord.py requests websocket-client duckduckgo-search swarm python-dotenv ollama websockets

    2. Environment Setup:
       - Create a .env file with:
         DISCORD_BOT_TOKEN=your_token
         DISCORD_NOTIFICATION_CHANNEL_ID=your_channel_id
         DISCORD_EXCLUDED_VOICE_CHANNEL_IDS=comma,separated,channel,ids

    3. Service Requirements:
       - Ollama running locally (for LLM and vision features)
       - ComfyUI running and accepting POST to /prompt and websocket at /ws (for image generation)

    4. Run:
       python main.py --model="your-ollama-model"
       Optional: Use --no-img to disable image generation

Available Commands:
    .llm <prompt>     - Query the LLM with your text prompt
    .img <prompt>     - Generate an image from text description
    .news <topic>     - Get news analysis on a topic
    .tarot           - Get a one-card Tarot reading
    .vision <prompt>  - Analyze an attached image using AI vision
    .chat <message>  - Chat with the bot (maintains conversation history)
    .clearchat       - Clear chat history (requires manage_messages permission)
"""

import argparse
import logging
import os
import base64
from dataclasses import dataclass
from typing import Optional, Final, List
from datetime import datetime
import discord
from discord.ext import commands
import random
import asyncio
import json
import uuid
import urllib.request
import urllib.parse
import websocket
import requests

from comfyui_workflows import stoiqo_comfyui_workflow, sd35medium_comfyui_worfklow
from newsagent import process_news
from tarotagent import generate_tarot_reading
from chat_db import ChatDatabase, ChatMessage
from dotenv import load_dotenv

# Load only the main .env file, not .env.example
load_dotenv(dotenv_path=".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args():
    """
    Parse command-line arguments to specify the Ollama model name.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Discord bot for LLM queries (Ollama), ComfyUI image generation, "
            "news summarization, tarot readings, and voice notifications."
        )
    )
    parser.add_argument(
        "--model",
        default="phi4:14b-fp16",
        help="Which Ollama model name to use."
    )
    parser.add_argument("--no-img", action="store_true", help="Disable image generation.")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data classes for Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscordConfig:
    """
    Holds all Discord-related configuration, including Bot Token.
    """
    bot_token: str
    command_prefix: str
    enable_message_content_intent: bool

@dataclass(frozen=True)
class LLMConfig:
    """
    LLM Config for Ollama usage.
    """
    url: str
    model_name: str
    stream: bool = False

# ---------------------------------------------------------------------------
# LLM Client (via requests) for quick commands
# ---------------------------------------------------------------------------
class LLMClient:
    """
    A client for interacting with a local Ollama instance.
    Handles both text and vision queries through the appropriate Ollama endpoints.

    Attributes:
        config (LLMConfig): Configuration for the LLM, including model name and endpoint URL.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the LLM client with the given configuration.

        Args:
            config (LLMConfig): Configuration object containing model name and endpoint URL.
        """
        self.config: Final[LLMConfig] = config

    def query(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Send a text prompt to the Ollama backend and return the response.

        Args:
            prompt (str): The text prompt to send to the model.
            model (Optional[str]): Override the default model name if provided.

        Returns:
            str: The model's response text, or an error message if the request fails.
        """
        model_to_use = model or self.config.model_name
        LOGGER.info("Using model: %s", model_to_use)
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": self.config.stream,
        }
        try:
            LOGGER.info("Sending prompt to Ollama: %s", prompt)
            response = requests.post(self.config.url, json=payload, timeout=300)
            if response.status_code != 200:
                err_msg = f"Ollama error: {response.status_code} {response.text}"
                LOGGER.error(err_msg)
                return err_msg
            data = response.json()
            result: str = data.get("response", "")
            return result.strip()
        except Exception as exc:
            err_msg = f"Request to Ollama failed: {exc}"
            LOGGER.exception(err_msg)
            return err_msg

    def query_vision(self, prompt: str, image_path: str) -> str:
        """
        Send a vision query to the Ollama backend with both text and image.

        Args:
            prompt (str): The text prompt describing what to analyze in the image.
            image_path (str): Path to the image file to analyze.

        Returns:
            str: The model's analysis of the image, or an error message if the request fails.
        """
        # Read and encode the image
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
        except Exception as exc:
            err_msg = f"Failed to read or encode image: {exc}"
            LOGGER.exception(err_msg)
            return err_msg

        vision_model = os.environ.get("VISION_MODEL", "llama3.2-vision:11b-instruct-q8_0")
        LOGGER.info("Using vision model: %s", vision_model)

        prompt = "Keep the response to 600 characters or less. " + prompt
        payload = {
            "model": vision_model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }],
            "stream": False
        }
        try:
            LOGGER.info("Sending vision prompt to Ollama: %s with image: %s", prompt, image_path)
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=300
            )
            if response.status_code != 200:
                err_msg = f"Ollama error: {response.status_code} {response.text}"
                LOGGER.error(err_msg)
                return err_msg

            # Handle the response content line by line
            response_text = ""
            for line in response.text.strip().split('\n'):
                try:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        response_text += data['message']['content']
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines

            return response_text.strip()
        except Exception as exc:
            err_msg = f"Request to Ollama failed: {exc}"
            LOGGER.exception(err_msg)
            return err_msg

# ---------------------------------------------------------------------------
# Moderation Helper
# ---------------------------------------------------------------------------
def ollama_moderation_check(llm_client: LLMClient, prompt: str, disallowed_categories: Optional[List[str]] = None) -> bool:
    """
    Check if the given prompt violates any disallowed content categories using the Llama Guard 3 model.

    Args:
        llm_client (LLMClient): The client to interact with the LLM.
        prompt (str): The text prompt to be checked.
        disallowed_categories (Optional[List[str]]): List of disallowed hazard categories (e.g., ['S1', 'S10']).

    Returns:
        bool: True if the prompt is considered unsafe, False otherwise.
    """
    if disallowed_categories is None:
        disallowed_categories = ['S4']

    # For Llama Guard 3, we just send the prompt directly
    payload = {
        "model": "llama-guard3:8b-q8_0",
        "prompt": prompt,
        "stream": False
    }
    try:
        LOGGER.info("Sending prompt to Llama Guard: %s", prompt)
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        if response.status_code != 200:
            err_msg = f"Llama Guard error: {response.status_code} {response.text}"
            LOGGER.error(err_msg)
            return False  # Fail safe: if we can't check, assume it's safe
        data = response.json()
        response_text = data.get("response", "").strip()
        
        LOGGER.info("Llama Guard raw response: %s", response_text)
        
        # Split response into lines
        response_lines = response_text.splitlines()
        if not response_lines:
            LOGGER.info("Llama Guard decision: SAFE (no response lines)")
            return False

        # Check if the response indicates unsafe content
        is_unsafe = response_lines[0].strip().lower() == "unsafe"
        if not is_unsafe:
            LOGGER.info("Llama Guard decision: SAFE")
            return False

        # Extract categories from the response
        violated_categories = response_lines[1:]
        is_violation = any(category in disallowed_categories for category in violated_categories)
        
        LOGGER.info(
            "Llama Guard decision: UNSAFE. Categories: %s. Matches disallowed categories? %s", 
            violated_categories, 
            is_violation
        )
        return is_violation

    except Exception as exc:
        LOGGER.exception("Request to Llama Guard failed: %s", exc)
        return True  # Fail safe: if we can't check, assume it's unsafe

# ---------------------------------------------------------------------------
# Summarization / Truncation
# ---------------------------------------------------------------------------
def condense_text(text: str, max_chars: int, llm_client: LLMClient) -> str:
    """
    Condense a text to fit within a maximum character limit, using LLM summarization if needed.

    Args:
        text (str): The text to condense.
        max_chars (int): Maximum number of characters allowed.
        llm_client (LLMClient): LLM client to use for summarization if needed.

    Returns:
        str: The condensed text, either truncated or summarized to fit within max_chars.
    """
    if len(text) <= max_chars:
        return text

    LOGGER.info("Original text length: %d characters.", len(text))

    LOGGER.info("Response is over %s characters; attempting summarization.", max_chars)
    summarization_prompt = (
        f"Condense the following text to fewer than {max_chars} characters, ensuring clarity and completeness:\n\n{text}"
    )
    summarized_text = llm_client.query(summarization_prompt).strip()
    LOGGER.info("Summarized text length: %d characters.", len(summarized_text))

    if len(summarized_text) <= max_chars:
        return summarized_text

    LOGGER.warning("Summarization result still exceeds %s characters. Truncating.", max_chars)
    return summarized_text[: (max_chars - 1)] + "…"

# ---------------------------------------------------------------------------
# ComfyUI Helpers
# ---------------------------------------------------------------------------
def get_image_from_comfyui(server_address: str, filename: str, subfolder: str, folder_type: str) -> bytes:
    """
    Retrieve an image from the ComfyUI server.

    Args:
        server_address (str): The address of the ComfyUI server.
        filename (str): Name of the image file to retrieve.
        subfolder (str): Subfolder containing the image.
        folder_type (str): Type of folder (e.g., 'output').

    Returns:
        bytes: The raw image data.
    """
    data = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    url_values = urllib.parse.urlencode(data)
    url = f"http://{server_address}/view?{url_values}"
    with urllib.request.urlopen(url) as response:
        return response.read()

def get_history(server_address: str, prompt_id: str) -> dict:
    """
    Retrieve the generation history for a specific prompt from ComfyUI.

    Args:
        server_address (str): The address of the ComfyUI server.
        prompt_id (str): The ID of the prompt to get history for.

    Returns:
        dict: The generation history data.
    """
    url = f"http://{server_address}/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def queue_prompt(server_address: str, workflow_dict: dict, client_id: str) -> dict:
    """
    Queue a new image generation prompt on the ComfyUI server.

    Args:
        server_address (str): The address of the ComfyUI server.
        workflow_dict (dict): The workflow configuration for image generation.
        client_id (str): Unique identifier for the client session.

    Returns:
        dict: The server's response containing the prompt ID.
    """
    data_dict = {
        "prompt": workflow_dict,
        "client_id": client_id
    }
    data_json = json.dumps(data_dict).encode("utf-8")

    req = urllib.request.Request(f"http://{server_address}/prompt", data=data_json)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def wait_for_prompt_completion(ws: websocket.WebSocket, prompt_id: str):
    """
    Wait for a ComfyUI prompt to complete processing.

    Args:
        ws (websocket.WebSocket): The websocket connection to ComfyUI.
        prompt_id (str): The ID of the prompt to wait for.
    """
    while True:
        msg = ws.recv()
        if not isinstance(msg, str):
            continue
        parsed = json.loads(msg)
        if parsed.get("type") == "executing":
            data = parsed.get("data", {})
            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                break

async def generate_comfyui(ctx: commands.Context, prompt_str: str) -> None:
    """
    Offload the entire ComfyUI generation to a separate thread so
    the bot remains responsive.
    """
    import copy

    def do_generation_logic():
        server_address = os.environ.get("COMFYUI_SERVER_ADDRESS", "localhost:8188")

        comfy_workflow_template = stoiqo_comfyui_workflow()
        comfy_payload = copy.deepcopy(comfy_workflow_template)

        random_seed = random.randint(0, 2**32 - 1)
        comfy_payload["6"]["inputs"]["text"] = prompt_str
        comfy_payload["294"]["inputs"]["seed"] = random_seed

        client_id = str(uuid.uuid4())
        ws_url = f"ws://{server_address}/ws?clientId={client_id}"

        try:
            ws = websocket.create_connection(ws_url, timeout=30)
            queue_result = queue_prompt(server_address, comfy_payload, client_id)
            prompt_id = queue_result.get("prompt_id")
            if not prompt_id:
                return (None, "Error: No prompt_id returned from ComfyUI /prompt.")

            wait_for_prompt_completion(ws, prompt_id)
            hist_data = get_history(server_address, prompt_id)
            hist = hist_data.get(prompt_id, {})
            outputs = hist.get("outputs", {})

            all_image_data = []
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img_info in node_output["images"]:
                        filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder", "")
                        folder_type = img_info.get("type", "")
                        if filename:
                            raw_bytes = get_image_from_comfyui(server_address, filename, subfolder, folder_type)
                            all_image_data.append(raw_bytes)

            ws.close()

            if not all_image_data:
                return (None, "No images returned from the ComfyUI workflow.")

            return (all_image_data[0], None)

        except Exception as exc:
            LOGGER.exception("ComfyUI generation (websocket) failed: %s", exc)
            return (None, f"Error generating image with ComfyUI via websockets: {exc}")

    image_bytes, error_msg = await asyncio.to_thread(do_generation_logic)

    if error_msg:
        await ctx.send(f"{ctx.author.mention} {error_msg}")
        return

    if image_bytes is None:
        await ctx.send(f"{ctx.author.mention} No image data returned.")
        return

    output_path = f"./comfyui_image_{ctx.author.id}.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    await ctx.send(f"{ctx.author.mention}", file=discord.File(output_path))

# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------
class DiscordLLMBot(commands.Bot):
    """
    A Discord Bot that provides AI-powered features through various commands.

    Features:
        - LLM queries via `.llm`
        - Image generation via `.img`
        - News analysis via `.news`
        - Tarot readings via `.tarot`
        - Image analysis via `.vision`
        - Chat with memory via `.chat`
        - Voice channel activity notifications

    Attributes:
        discord_config (DiscordConfig): Configuration for Discord connection and behavior.
        llm_client (LLMClient): Client for interacting with the Ollama LLM.
        no_img (bool): Whether image generation is disabled.
        chat_db (ChatDatabase): Database for storing chat history.
    """

    def __init__(
        self,
        discord_config: DiscordConfig,
        llm_client: LLMClient,
        no_img: bool
    ) -> None:
        """
        Initialize the Discord bot with the given configuration.

        Args:
            discord_config (DiscordConfig): Configuration for Discord connection and behavior.
            llm_client (LLMClient): Client for interacting with the Ollama LLM.
            no_img (bool): Whether to disable image generation.
        """
        if discord_config.enable_message_content_intent:
            intents = discord.Intents.default()
            intents.message_content = True
        else:
            intents = discord.Intents.default()

        intents.voice_states = True
        intents.guilds = True

        super().__init__(command_prefix=discord_config.command_prefix, intents=intents)
        self.discord_config: Final[DiscordConfig] = discord_config
        self.llm_client: Final[LLMClient] = llm_client
        self.no_img: Final[bool] = no_img
        self.chat_db: Final[ChatDatabase] = ChatDatabase()

    async def on_ready(self) -> None:
        """
        Event handler called when the bot has successfully connected to Discord.
        Logs the bot's login status.
        """
        LOGGER.info("[Bot] Logged in as: %s", self.user)

    async def setup_hook(self) -> None:
        """
        Set up all bot commands and their handlers.
        This is called automatically by discord.py during bot initialization.
        """
        @commands.command(
            name="llm",
            brief="Query our local LLM with your text prompt.",
            help=(
                "Query our local LLM with your text prompt.\n\n"
                "The response will be automatically condensed if it exceeds Discord's "
                "character limit.\n\n"
                "Usage: .llm <your question or prompt>\n"
                "Example: .llm What is the capital of France?"
            ),
            description="Query our local LLM with your text prompt."
        )
        async def llm_cmd(
            ctx: commands.Context, 
            *, 
            prompt: str = commands.parameter(
                default="Hello from Discord to LLM!",
                description="The question or prompt to send to the LLM"
            )
        ) -> None:
            """
            Handle the .llm command by sending the prompt to the LLM and returning its response.

            Args:
                ctx (commands.Context): The command context.
                prompt (str): The prompt to send to the LLM.
            """
            prefix = "Please respond succinctly. Your answer must be under 800 characters total.\n"
            full_prompt = f"{prefix}{prompt}"

            def do_llm_logic():
                raw_response = self.llm_client.query(full_prompt)
                return condense_text(raw_response, 800, self.llm_client)

            final_text = await asyncio.to_thread(do_llm_logic)
            await ctx.send(f"{ctx.author.mention} {final_text}")

        self.add_command(llm_cmd)

        @commands.command(
            name="img",
            brief="Generate an image using our local ComfyUI workflow.",
            help=(
                "Generate an image using our local ComfyUI workflow.\n\n"
                "The prompt will be checked for safety using Llama Guard before generation.\n"
                "Usage: .img <your image description>\n"
                "Example: .img a beautiful sunset over mountains"
            ),
            description="Generate an image using our local ComfyUI workflow."
        )
        async def img_cmd(
            ctx: commands.Context, 
            *, 
            prompt: str = commands.parameter(
                description="A detailed description of the image you want to generate"
            )
        ) -> None:
            """
            Handle the .img command by generating an image using ComfyUI.

            Args:
                ctx (commands.Context): The command context.
                prompt (str): Description of the image to generate.
            """
            if self.no_img:
                await ctx.send(f"{ctx.author.mention} Image generation is disabled presently.")
                return

            banned_users = [
                "badUser#1234",
                "someoneElse#9876",
            ]
            user_str = str(ctx.author)
            if user_str in banned_users:
                await ctx.send(f"{ctx.author.mention} You are banned from using this command.")
                return

            if not prompt:
                await ctx.send(f"{ctx.author.mention} Please provide a prompt for image generation.")
                return

            restricted_keywords = [
                "restricted", "restricted content", "restricted material",
            ]
            found_keyword = next(
                (kw for kw in restricted_keywords if kw.lower() in prompt.lower()),
                None
            )
            if found_keyword:
                flagged = await asyncio.to_thread(ollama_moderation_check, self.llm_client, prompt, ['S4'])
                if flagged:
                    await ctx.send(
                        f"{ctx.author.mention} Your prompt was flagged as unsafe content by Llama Guard."
                    )
                    return

            await generate_comfyui(ctx, prompt)

        self.add_command(img_cmd)

        @commands.command(
            name="news",
            brief="Summarize and analyze the latest news on a given topic.",
            help=(
                "Summarize and analyze the latest news on a given topic.\n\n"
                "Uses a multi-agent approach to gather, analyze, and synthesize news.\n"
                "The response is automatically condensed to fit Discord's limits.\n\n"
                "Usage: .news <topic>\n"
                "Example: .news artificial intelligence"
            ),
            description="Get a summary of the latest news on a specific topic."
        )
        async def news_cmd(
            ctx: commands.Context, 
            *, 
            topic: str = commands.parameter(
                description="The news topic you want to learn about"
            )
        ) -> None:
            """
            Handle the .news command by analyzing recent news on the given topic.

            Args:
                ctx (commands.Context): The command context.
                topic (str): The topic to analyze news about.
            """
            if not topic:
                await ctx.send(f"{ctx.author.mention} Please provide a topic, e.g. `.news inflation`")
                return

            def do_news_logic():
                try:
                    raw_news, synthesized_news, final_summary = process_news(topic)
                    return final_summary
                except Exception as exc:
                    LOGGER.exception("Error during news processing: %s", exc)
                    return f"Error during news processing: {exc}"

            message_text = await asyncio.to_thread(do_news_logic)
            await ctx.send(f"{ctx.author.mention} {message_text}")

        self.add_command(news_cmd)

        @commands.command(
            name="tarot",
            brief="Get a one card Tarot reading.",
            help=(
                "Get a one card Tarot reading about your current situation.\n\n"
                "Draws a single card and provides an interpretation.\n"
                "The reading is automatically condensed if too long.\n\n"
                "Usage: .tarot\n"
                "No additional arguments needed."
            ),
            description="Get a one card Tarot reading about your current situation."
        )
        async def tarot_cmd(ctx: commands.Context) -> None:
            """
            Handle the .tarot command by providing a one-card Tarot reading.

            Args:
                ctx (commands.Context): The command context.
            """
            def do_tarot_logic():
                reading = generate_tarot_reading()
                words = reading.split()
                if len(words) > 600:
                    reading = " ".join(words[:600])
                return reading

            final_reading = await asyncio.to_thread(do_tarot_logic)
            await ctx.send(f"{ctx.author.mention} {final_reading}")

        self.add_command(tarot_cmd)

        @commands.command(
            name="chat",
            brief="Chat with the bot with conversation memory.",
            help=(
                "Chat with the bot while maintaining conversation history.\n\n"
                "The bot remembers the conversation history for each server until cleared.\n"
                "Use .clearchat to clear the history.\n\n"
                "Usage: .chat <your message>\n"
                "Example: .chat What's the weather like?"
            ),
            description="Chat with the bot while maintaining conversation history."
        )
        async def chat_cmd(
            ctx: commands.Context,
            *,
            message: str = commands.parameter(
                description="Your message to the bot"
            )
        ) -> None:
            """
            Handle the .chat command by maintaining a conversation history.

            Args:
                ctx (commands.Context): The command context.
                message (str): The user's message.
            """
            if not message:
                await ctx.send(f"{ctx.author.mention} Please provide a message to chat about.")
                return

            # Get chat history
            history = self.chat_db.get_chat_history(ctx.guild.id)
            
            # Format history for the LLM
            system_prompt = (
                "You are a helpful AI assistant in a Discord chat. "
                "Be concise, friendly, and engaging. "
                f"Keep responses under {os.environ.get('CHAT_RESPONSE_CHAR_LIMIT', '800')} characters."
            )
            
            conversation = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add history
            for msg in history:
                role = "assistant" if msg.is_bot else "user"
                conversation.append({
                    "role": role,
                    "content": f"{msg.username}: {msg.message}"
                })
            
            # Add current message
            conversation.append({
                "role": "user",
                "content": f"{ctx.author.name}: {message}"
            })
            
            # Store user's message
            self.chat_db.add_message(ChatMessage(
                server_id=ctx.guild.id,
                user_id=ctx.author.id,
                username=ctx.author.name,
                message=message,
                is_bot=False,
                timestamp=datetime.now()
            ))
            
            # Get LLM response
            def do_chat_logic():
                prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation)
                chat_model = os.environ.get("CHAT_MODEL", "phi4:14b-fp16")
                LOGGER.info("Using chat model: %s", chat_model)
                response = self.llm_client.query(prompt, model=chat_model)
                return condense_text(response, int(os.environ.get("CHAT_RESPONSE_CHAR_LIMIT", "800")), self.llm_client)
            
            response_text = await asyncio.to_thread(do_chat_logic)
            
            # Store bot's response
            self.chat_db.add_message(ChatMessage(
                server_id=ctx.guild.id,
                user_id=self.user.id,
                username=self.user.name,
                message=response_text,
                is_bot=True,
                timestamp=datetime.now()
            ))
            
            await ctx.send(f"{ctx.author.mention} {response_text}")

        self.add_command(chat_cmd)

        @commands.command(
            name="clearchat",
            brief="Clear the chat history for this server.",
            help=(
                "Clear the entire chat history for the current server.\n\n"
                "This command can only be used by users with the 'manage_messages' permission.\n\n"
                "Usage: .clearchat\n"
                "No additional arguments needed."
            ),
            description="Clear the chat history for this server."
        )
        @commands.has_permissions(manage_messages=True)
        async def clearchat_cmd(ctx: commands.Context) -> None:
            """
            Handle the .clearchat command by clearing the server's chat history.

            Args:
                ctx (commands.Context): The command context.
            """
            self.chat_db.clear_chat_history(ctx.guild.id)
            await ctx.send(f"{ctx.author.mention} Chat history has been cleared.")

        self.add_command(clearchat_cmd)

        @commands.command(
            name="vision",
            brief="Ask llama3.2-vision to analyze an image.",
            help=(
                "Ask llama3.2-vision to analyze an image.\n\n"
                "You must attach an image to your message along with your prompt.\n"
                "The model will analyze the image and answer your prompt about it.\n\n"
                "Usage: .vision <your prompt> [attach an image]\n"
                "Example: .vision What is in this image? [with attached image]"
            ),
            description="Analyze an attached image using AI vision model."
        )
        async def vision_cmd(
            ctx: commands.Context, 
            *, 
            prompt: str = commands.parameter(
                description="Your question about the image (must also attach an image to the message)"
            )
        ) -> None:
            """
            Handle the .vision command by analyzing an attached image.

            Args:
                ctx (commands.Context): The command context.
                prompt (str): Question about the attached image.
            """
            if not ctx.message.attachments:
                await ctx.send(f"{ctx.author.mention} Please attach an image for the vision command.")
                return

            attachment = ctx.message.attachments[0]
            if not prompt:
                await ctx.send(f"{ctx.author.mention} Please provide a prompt along with the image.")
                return

            # Download the image using Discord's methods
            image_path = f"./temp_image_{ctx.author.id}.jpg"
            try:
                await attachment.save(image_path)
            except Exception as exc:
                await ctx.send(f"{ctx.author.mention} Failed to download the image: {exc}")
                return

            # Query the Ollama model asynchronously using LLMClient
            def do_vision_logic():
                try:
                    response = self.llm_client.query_vision(prompt, image_path)
                    return condense_text(response, int(os.environ.get("VISION_RESPONSE_CHAR_LIMIT", "600")), self.llm_client)
                except Exception as exc:
                    LOGGER.exception("Error querying the vision model: %s", exc)
                    return f"Error querying the vision model: {exc}"

            response_text = await asyncio.to_thread(do_vision_logic)
            await ctx.send(f"{ctx.author.mention} {response_text}")

            # Clean up the downloaded image
            if os.path.exists(image_path):
                os.remove(image_path)

        self.add_command(vision_cmd)

    async def on_voice_state_update(self, member, before, after):
        """
        Event handler for voice channel state changes.
        Posts join/leave/move messages to a configured text channel.

        Args:
            member (discord.Member): The member whose voice state changed.
            before (discord.VoiceState): The previous voice state.
            after (discord.VoiceState): The new voice state.
        """
        channel_id_str = os.environ.get("DISCORD_NOTIFICATION_CHANNEL_ID")
        if not channel_id_str:
            LOGGER.warning("DISCORD_NOTIFICATION_CHANNEL_ID not configured")
            return

        try:
            channel_id = int(channel_id_str)
        except ValueError:
            LOGGER.warning("DISCORD_NOTIFICATION_CHANNEL_ID invalid: %r", channel_id_str)
            return

        text_channel = self.get_channel(channel_id)
        if not text_channel:
            LOGGER.warning("Notification text channel not found. Check the channel ID.")
            return

        excluded_channels_str = os.environ.get("DISCORD_EXCLUDED_VOICE_CHANNEL_IDS", "")
        excluded_channels = []
        if excluded_channels_str:
            for ch in excluded_channels_str.split(","):
                ch = ch.strip()
                if ch.isdigit():
                    excluded_channels.append(int(ch))

        # If the event involves any excluded channel, skip
        if (before.channel and before.channel.id in excluded_channels) or \
           (after.channel and after.channel.id in excluded_channels):
            return

        # User joined a voice channel
        if before.channel is None and after.channel is not None:
            message = f"{member.display_name} has joined."
            await text_channel.send(message)
            LOGGER.info(message)
        # User left a voice channel
        elif before.channel is not None and after.channel is None:
            message = f"{member.display_name} has left."
            await text_channel.send(message)
            LOGGER.info(message)
        # User moved between voice channels
        elif before.channel != after.channel:
            message = (
                f"{member.display_name} moved from {before.channel.name} "
                f"to {after.channel.name}."
            )
            await text_channel.send(message)
            LOGGER.info(message)

    def run_bot(self) -> None:
        super().run(self.discord_config.bot_token, reconnect=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ollama_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")

    # Log environment variables for debugging
    LOGGER.info("Environment variables:")
    LOGGER.info("  OLLAMA_API_URL: %s", ollama_url)
    LOGGER.info("  DEFAULT_LLM_MODEL: %s", os.environ.get("DEFAULT_LLM_MODEL"))
    LOGGER.info("  VISION_MODEL: %s", os.environ.get("VISION_MODEL"))
    LOGGER.info("  CHAT_MODEL: %s", os.environ.get("CHAT_MODEL"))
    LOGGER.info("  TAROT_MODEL: %s", os.environ.get("TAROT_MODEL"))
    LOGGER.info("  NEWS_MODEL: %s", os.environ.get("NEWS_MODEL"))

    llm_config = LLMConfig(
        url=ollama_url,
        model_name=os.environ.get("DEFAULT_LLM_MODEL", args.model),
        stream=False,
    )

    llm_client = LLMClient(llm_config)

    discord_token = os.environ.get("DISCORD_BOT_TOKEN")
    if not discord_token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable is required")

    discord_config = DiscordConfig(
        bot_token=discord_token,
        command_prefix=os.environ.get("DISCORD_COMMAND_PREFIX", "."),
        enable_message_content_intent=True,
    )

    bot = DiscordLLMBot(discord_config, llm_client, args.no_img)
    bot.run_bot()

if __name__ == "__main__":
    main()
