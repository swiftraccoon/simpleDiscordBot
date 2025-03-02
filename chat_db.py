import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load only the main .env file, not .env.example
load_dotenv(dotenv_path=".env", override=True)

# Configuration constants
MAX_MESSAGES = 100  # Maximum number of messages to keep in history per server
MAX_CONTEXT_SIZE = 32000  # Maximum context size for the LLM (in characters)
                         # Conservative estimate to leave room for system prompts and new messages

@dataclass
class ChatMessage:
    """Represents a single message in the chat history."""
    server_id: int
    user_id: int
    username: str
    message: str
    is_bot: bool
    timestamp: datetime

class ChatDatabase:
    """Handles all database operations for the chat feature."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database connection and create tables if they don't exist."""
        if db_path is None:
            db_path = os.environ.get("CHAT_DB_PATH", "chat_history.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create the chat_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                is_bot BOOLEAN NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a new message to the chat history."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history 
            (server_id, user_id, username, message, is_bot, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message.server_id,
            message.user_id,
            message.username,
            message.message,
            message.is_bot,
            message.timestamp
        ))
        self.conn.commit()
    
    def get_chat_history(self, server_id: int) -> List[ChatMessage]:
        """Get the chat history for a specific server."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT server_id, user_id, username, message, is_bot, timestamp
            FROM chat_history
            WHERE server_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (server_id, MAX_MESSAGES))
        
        messages = []
        for row in cursor.fetchall():
            messages.append(ChatMessage(
                server_id=row[0],
                user_id=row[1],
                username=row[2],
                message=row[3],
                is_bot=row[4],
                timestamp=datetime.fromisoformat(row[5])
            ))
        
        return list(reversed(messages))  # Return in chronological order
    
    def clear_chat_history(self, server_id: int) -> None:
        """Clear the chat history for a specific server."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE server_id = ?", (server_id,))
        self.conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close() 