# Discord Bot with Ollama & ComfyUI Integration

A simple Python-based Discord bot that combines a local LLM (via [Ollama](https://github.com/ollama/ollama)) with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation. It also features news analysis, tarot readings, voice channel notifications, and an interactive chat command with persistent conversation history.

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Commands](#available-commands)
- [File Structure](#file-structure)
- [Contributing](#contributing)

---

## Features

1. **LLM Queries (`.llm`)**  
   - Query a local Ollama instance for text-based responses.

2. **Image Generation (`.img`)**  
   - Generate images via ComfyUI based on custom workflow templates.

3. **Chat with Memory (`.chat` / `.clearchat`)**  
   - Engage in a multi-turn conversation with the bot, which stores history in a local SQLite database.

4. **News Analysis (`.news`)**  
   - A multi-agent system retrieves the latest news on a topic, synthesizes the key points, and summarizes them.

5. **Tarot Readings (`.tarot`)**  
   - Get a one-card Tarot reading with interpretation and guidance.

6. **Vision Model (`.vision`)**  
   - Analyze images using Ollama’s vision model by attaching an image and providing a prompt.

7. **Voice Channel Notifications**  
   - Posts notifications in a specific Discord channel whenever someone joins, leaves, or switches voice channels (excluding any channels you mark as excluded in the `.env`).

---

## Architecture Overview

```
Discord Bot        Ollama (Local LLM)        ComfyUI (Image Gen)
   |                       |                         |
   | Discord.py           | requests                | WebSocket + REST
   +----------------------|------------------------- |-------------------> 
   | Chat commands        |.llm, .vision            | .img generation
   | newsagent,           |                         | 
   | tarotagent           |                         |
```

- **Discord.py**: Manages bot commands, events, and server interactions.  
- **Ollama**: Serves as the LLM backend for text queries and image analysis (vision model).  
- **ComfyUI**: Workflow-driven image generation connected via a combination of REST/POST and WebSocket for job queueing.

---

## Prerequisites

1. **Python 3.8+**  
2. **[Ollama](https://github.com/ollama/ollama) running locally** (for text and vision queries).  
3. **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** running and reachable via HTTP/WS.  
4. **Discord Bot Token**: Create an application and bot on [Discord's Developer Portal](https://discord.com/developers/applications).  

---

## Installation

1. **Clone this repository** (or copy the files into a local folder):
   ```bash
   git clone https://github.com/swiftraccoon/simpleDiscordBot.git
   cd simpleDiscordBot
   ```

2. **Install Python dependencies**:
   ```bash
   pip install discord.py requests websocket-client duckduckgo-search swarm python-dotenv ollama websockets
   ```
   > *Tip:* You may want to use a virtual environment:
   > ```bash
   > python -m venv venv
   > source venv/bin/activate
   > pip install ...
   > ```

3. **Set up Ollama**:  
   Follow [Ollama’s installation instructions](https://github.com/ollama/ollama) to ensure it is running locally (default port `11434`).

4. **Set up ComfyUI**:  
   Ensure [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is installed and running (default port `8188`).

---

## Configuration

1. **Copy the example env file**:
   ```bash
   cp .env.example .env
   ```
2. **Edit your `.env`** to add your own credentials and settings:
   - `DISCORD_BOT_TOKEN` = Your Discord bot token.
   - `DISCORD_NOTIFICATION_CHANNEL_ID` = The channel ID where voice join/leave notifications should appear.
   - `DISCORD_EXCLUDED_VOICE_CHANNEL_IDS` = A comma-separated list of channel IDs you don’t want to track.
   - `OLLAMA_API_URL` = URL for Ollama’s `/api/generate` endpoint. (Default: `http://localhost:11434/api/generate`)
   - `COMFYUI_SERVER_ADDRESS` = Address of your ComfyUI instance, e.g. `localhost:8188`.
   - Additional model names (`DEFAULT_LLM_MODEL`, `CHAT_MODEL`, `VISION_MODEL`, etc.) can be overridden as needed.

   For example:
   ```env
   DISCORD_BOT_TOKEN=your_discord_bot_token
   DISCORD_NOTIFICATION_CHANNEL_ID=123456789012345678
   DISCORD_EXCLUDED_VOICE_CHANNEL_IDS=987654321098765432,123456789012345679
   OLLAMA_API_URL=http://localhost:11434/api/generate
   COMFYUI_SERVER_ADDRESS=localhost:8188
   ...
   ```

3. **(Optional) Adjust Bot Prefix**:  
   Change `DISCORD_COMMAND_PREFIX` in your `.env` if you want something other than `.`.

---

## Usage

1. **Run the Bot**:
   ```bash
   python main.py
   ```
   - By default, the bot uses the `--model` specified in `.env` or you can override it:
     ```bash
     python main.py --model "some-other-model"
     ```

2. **Disable Image Generation**:  
   If you do not want to generate images, add `--no-img`:
   ```bash
   python main.py --no-img
   ```

3. **Invite the Bot to Your Server**:  
   - Go to the [Discord Developer Portal](https://discord.com/developers/applications) → Your Application → **OAuth2** → **URL Generator**.
   - Select **bot** scope and any necessary **permissions** (e.g., for voice states, message content, manage messages, etc.).
   - Paste the generated invite link into your browser and select the server to add it to.

4. **Confirm It’s Online**:  
   - You should see a log message: `"Logged in as: YourBotName"`.
   - In your server, type `.help` (or your custom prefix + `help`) to see available commands.

---

## Available Commands

Below are the default commands (`.` is the default prefix; adjust if you changed `DISCORD_COMMAND_PREFIX`):

- **`.llm <prompt>`**  
  Query the local LLM (Ollama). Example:  
  ```
  .llm What is the capital of France?
  ```

- **`.img <prompt>`**  
  Generate an image via ComfyUI using a preconfigured workflow. Example:  
  ```
  .img a vivid painting of a dragon made of stardust
  ```
  *This requires ComfyUI to be running.*

- **`.vision <prompt>`** (attach an image)  
  Send an image prompt to Ollama’s vision model. Example:  
  ```
  .vision What is the primary object in this photo?
  ```
  *Make sure to attach an image in Discord along with your text prompt.*

- **`.news <topic>`**  
  Get a summary of the latest news on a topic. Example:  
  ```
  .news AI breakthroughs
  ```

- **`.tarot`**  
  Draw a single Tarot card for a quick reading. Example:  
  ```
  .tarot
  ```

- **`.chat <message>`**  
  Start or continue a chat conversation with the bot. It maintains context across messages in a server-specific database. Example:  
  ```
  .chat Hey, how’s it going?
  ```

- **`.clearchat`**  
  Clear the stored chat history for the current server. This requires the **Manage Messages** permission. Example:  
  ```
  .clearchat
  ```

---

## File Structure

```
.
├── .env.example             # Environment variable template
├── chat_db.py              # Chat database interactions (SQLite)
├── comfyui_workflows.py    # Workflow templates for image generation in ComfyUI
├── main.py                 # The main entry point (Discord bot logic)
├── newsagent.py            # News agent multi-stage processing
├── tarotagent.py           # Tarot reading agent
└── README.md               # This file
```

- **`.env.example`**  
  Contains sample environment variables. Copy and rename to `.env`, then fill in your own details.
- **`main.py`**  
  The primary script that starts the Discord bot, loads environment variables, and registers commands.
- **`chat_db.py`**  
  Handles saving and retrieving conversation history from a SQLite database.
- **`comfyui_workflows.py`**  
  Provides predefined JSON-based workflows for ComfyUI image generation.
- **`newsagent.py`**  
  Uses a multi-agent approach for searching and synthesizing recent news.
- **`tarotagent.py`**  
  Randomly draws a Tarot card and provides a short reading using an LLM agent approach.

---

## Contributing

1. **Fork or Clone** this repository.
2. Create a new branch for your feature or bugfix.
3. Submit a Pull Request with a clear description of your changes.

All contributions are welcome—whether it’s improving documentation, fixing bugs, or adding new features!

*Enjoy building your AI-powered Discord experience!*