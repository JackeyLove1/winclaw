"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    content: str  # Message text
    session_id: str  # Unique id
    timestamp: datetime = field(default_factory=datetime.now)
    media: List[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    sender_id: Optional[str] = None  # User identifier
    channel: Optional[str] = None  # telegram, discord, slack, whatsapp
    chat_id: Optional[str] = None  # Chat/channel identifier

@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    content: str
    session_id: str
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    channel: Optional[str] = None
    chat_id: Optional[str] = None
    reply_to: Optional[str] = None

