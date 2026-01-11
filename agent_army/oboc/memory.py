"""
OBOC Memory Hierarchy

Implements the three-layer memory architecture:

SHORT-TERM MEMORY:
    - Conversation context (current session)
    - Active context window
    - Lifetime: Single session

WORKING MEMORY:
    - Intermediate task state
    - Research progress, to-do lists
    - Structured notes (NOTES.md pattern)
    - Lifetime: Duration of task

LONG-TERM MEMORY:
    - User preferences, learned patterns
    - Historical interactions
    - Semantic extraction and consolidation
    - Lifetime: Persistent (external storage)
"""

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class MemoryType(Enum):
    """Types of memory in the OBOC hierarchy."""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    importance: float = 0.5  # 0.0 to 1.0
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "importance": self.importance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            importance=data.get("importance", 0.5),
        )


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNote:
    """Structured note following the NOTES.md pattern."""
    task_id: str
    current_phase: str = ""
    completed: List[str] = field(default_factory=list)
    key_decisions: List[Dict[str, str]] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert to NOTES.md format."""
        md = f"# Task Notes: {self.task_id}\n\n"
        
        md += f"## Current Phase\n- {self.current_phase}\n\n"
        
        if self.completed:
            md += "## Completed\n"
            for item in self.completed:
                md += f"- [x] {item}\n"
            md += "\n"
        
        if self.key_decisions:
            md += "## Key Decisions\n"
            for decision in self.key_decisions:
                md += f"- {decision.get('decision', '')} (reason: {decision.get('reason', '')})\n"
            md += "\n"
        
        if self.unresolved_issues:
            md += "## Unresolved Issues\n"
            for issue in self.unresolved_issues:
                md += f"- {issue}\n"
            md += "\n"
        
        if self.next_steps:
            md += "## Next Steps\n"
            for i, step in enumerate(self.next_steps, 1):
                md += f"{i}. {step}\n"
            md += "\n"
        
        return md


class MemoryStore(ABC):
    """Abstract base class for memory stores."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search for relevant memories."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def consolidate(self) -> int:
        """Consolidate and compress memories. Returns count of consolidated."""
        pass


class ShortTermMemory(MemoryStore):
    """
    SHORT-TERM MEMORY
    
    - Conversation context for current session
    - Active context window
    - Automatically expires at session end
    """
    
    def __init__(self, max_messages: int = 50, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[ConversationMessage] = []
        self._entries: Dict[str, MemoryEntry] = {}
        self._session_id: str = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        return hashlib.sha256(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
    
    @property
    def session_id(self) -> str:
        return self._session_id
    
    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata or {},
        )
        self._messages.append(message)
        
        # Trim if exceeds limit
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)
    
    def get_messages(
        self,
        limit: Optional[int] = None,
        roles: Optional[List[str]] = None,
    ) -> List[ConversationMessage]:
        """Get conversation messages."""
        messages = self._messages
        
        if roles:
            messages = [m for m in messages if m.role in roles]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_context_window(self) -> str:
        """Get the current context window as a string."""
        context_parts = []
        for msg in self._messages:
            prefix = f"[{msg.agent_name}] " if msg.agent_name else ""
            context_parts.append(f"{prefix}{msg.role.upper()}: {msg.content}")
        return "\n\n".join(context_parts)
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a short-term memory entry."""
        entry.memory_type = MemoryType.SHORT_TERM
        self._entries[entry.id] = entry
        return entry.id
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        return self._entries.get(entry_id)
    
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search short-term memories (simple substring match)."""
        results = []
        for entry in self._entries.values():
            if query.lower() in entry.content.lower():
                results.append(entry)
        return results[:limit]
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False
    
    async def consolidate(self) -> int:
        """Short-term memory doesn't consolidate - it expires."""
        return 0
    
    def clear(self):
        """Clear all short-term memory (session end)."""
        self._messages.clear()
        self._entries.clear()
        self._session_id = self._generate_session_id()


class WorkingMemory(MemoryStore):
    """
    WORKING MEMORY
    
    - Intermediate task state
    - Research progress, to-do lists
    - Structured notes (NOTES.md pattern)
    - Lifetime: Duration of task
    """
    
    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}
        self._notes: Dict[str, TaskNote] = {}
        self._task_artifacts: Dict[str, Dict[str, Any]] = {}
    
    def create_task_notes(self, task_id: str) -> TaskNote:
        """Create a new task note."""
        note = TaskNote(task_id=task_id)
        self._notes[task_id] = note
        return note
    
    def get_task_notes(self, task_id: str) -> Optional[TaskNote]:
        """Get task notes by ID."""
        return self._notes.get(task_id)
    
    def update_task_notes(
        self,
        task_id: str,
        current_phase: Optional[str] = None,
        completed_item: Optional[str] = None,
        key_decision: Optional[Dict[str, str]] = None,
        unresolved_issue: Optional[str] = None,
        next_step: Optional[str] = None,
    ):
        """Update task notes incrementally."""
        note = self._notes.get(task_id)
        if not note:
            note = self.create_task_notes(task_id)
        
        if current_phase:
            note.current_phase = current_phase
        if completed_item:
            note.completed.append(completed_item)
        if key_decision:
            note.key_decisions.append(key_decision)
        if unresolved_issue:
            note.unresolved_issues.append(unresolved_issue)
        if next_step:
            note.next_steps.append(next_step)
    
    def store_artifact(self, task_id: str, name: str, content: Any):
        """Store a task artifact (code, file, data)."""
        if task_id not in self._task_artifacts:
            self._task_artifacts[task_id] = {}
        self._task_artifacts[task_id][name] = content
    
    def get_artifact(self, task_id: str, name: str) -> Optional[Any]:
        """Get a task artifact."""
        return self._task_artifacts.get(task_id, {}).get(name)
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a working memory entry."""
        entry.memory_type = MemoryType.WORKING
        self._entries[entry.id] = entry
        return entry.id
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        return self._entries.get(entry_id)
    
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search working memories."""
        results = []
        for entry in self._entries.values():
            if query.lower() in entry.content.lower():
                results.append(entry)
        return results[:limit]
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False
    
    async def consolidate(self) -> int:
        """Consolidate working memory - summarize and compress."""
        # This would use LLM to summarize verbose entries
        return 0
    
    def clear_task(self, task_id: str):
        """Clear all working memory for a specific task."""
        if task_id in self._notes:
            del self._notes[task_id]
        if task_id in self._task_artifacts:
            del self._task_artifacts[task_id]
        
        # Remove entries associated with the task
        to_remove = [
            eid for eid, entry in self._entries.items()
            if entry.metadata.get("task_id") == task_id
        ]
        for eid in to_remove:
            del self._entries[eid]


class LongTermMemory(MemoryStore):
    """
    LONG-TERM MEMORY
    
    - User preferences, learned patterns
    - Historical interactions
    - Semantic extraction and consolidation
    - Lifetime: Persistent (external storage)
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".agent_army" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, MemoryEntry] = {}
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load memories from persistent storage."""
        memory_file = self.storage_path / "long_term.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    entry = MemoryEntry.from_dict(entry_data)
                    self._entries[entry.id] = entry
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load long-term memory: {e}")
    
    def _save_to_disk(self):
        """Save memories to persistent storage."""
        memory_file = self.storage_path / "long_term.json"
        data = {
            "entries": [entry.to_dict() for entry in self._entries.values()],
            "last_saved": datetime.now().isoformat(),
        }
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a long-term memory entry."""
        entry.memory_type = MemoryType.LONG_TERM
        self._entries[entry.id] = entry
        self._save_to_disk()
        return entry.id
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        return self._entries.get(entry_id)
    
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search long-term memories."""
        # In production, this would use vector similarity search
        results = []
        for entry in self._entries.values():
            if query.lower() in entry.content.lower():
                results.append(entry)
        
        # Sort by importance
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save_to_disk()
            return True
        return False
    
    async def consolidate(self) -> int:
        """
        Consolidate long-term memories.
        
        - Merge similar entries
        - Summarize verbose content
        - Update importance scores based on access patterns
        """
        # This would use LLM for semantic consolidation
        # For now, just return 0
        return 0
    
    async def extract_learnings(
        self,
        source_entries: List[MemoryEntry],
    ) -> List[MemoryEntry]:
        """
        Extract learnings from working/short-term memories
        and store them in long-term memory.
        """
        learnings = []
        # This would use LLM to extract key learnings
        # from the source entries
        return learnings


class MemoryManager:
    """
    Unified memory manager for OBOC.
    
    Coordinates all three memory tiers and handles:
    - Memory retrieval across tiers
    - Consolidation and compression
    - Learning extraction
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(storage_path)
    
    async def retrieve_context(
        self,
        query: str,
        include_short_term: bool = True,
        include_working: bool = True,
        include_long_term: bool = True,
        limit_per_tier: int = 5,
    ) -> Dict[str, List[MemoryEntry]]:
        """Retrieve relevant context from all memory tiers."""
        context = {}
        
        if include_short_term:
            context["short_term"] = await self.short_term.search(query, limit=limit_per_tier)
        
        if include_working:
            context["working"] = await self.working.search(query, limit=limit_per_tier)
        
        if include_long_term:
            context["long_term"] = await self.long_term.search(query, limit=limit_per_tier)
        
        return context
    
    async def promote_to_long_term(
        self,
        entry_id: str,
        source: Union[ShortTermMemory, WorkingMemory],
    ) -> bool:
        """Promote a memory entry to long-term storage."""
        entry = await source.retrieve(entry_id)
        if entry:
            entry.importance = max(entry.importance, 0.7)  # Boost importance
            await self.long_term.store(entry)
            return True
        return False
    
    async def session_end(self):
        """Handle session end - consolidate and extract learnings."""
        # Extract learnings from working memory
        working_entries = list(self.working._entries.values())
        await self.long_term.extract_learnings(working_entries)
        
        # Clear short-term memory
        self.short_term.clear()
    
    async def task_end(self, task_id: str):
        """Handle task end - consolidate working memory."""
        # Get task notes for potential learning extraction
        notes = self.working.get_task_notes(task_id)
        if notes and notes.key_decisions:
            # Store key decisions as long-term learnings
            for decision in notes.key_decisions:
                entry = MemoryEntry(
                    id=f"decision-{task_id}-{len(notes.key_decisions)}",
                    content=f"Decision: {decision.get('decision', '')}. Reason: {decision.get('reason', '')}",
                    memory_type=MemoryType.LONG_TERM,
                    importance=0.7,
                    metadata={"task_id": task_id, "type": "key_decision"},
                )
                await self.long_term.store(entry)
        
        # Clear task-specific working memory
        self.working.clear_task(task_id)
