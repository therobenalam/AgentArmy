"""
OBOC Knowledge Graph

Implements knowledge graph infrastructure for agent orchestration:

Benefits of Knowledge Graphs for Agents:
- Explicit relationship encoding (not just similarity)
- Dependency tracing (causal reasoning)
- Semantic understanding of context
- Explainable decision trails

Use Cases:
- Incident response (trace service dependencies)
- Project management (track task dependencies)
- Code understanding (module relationships)
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Iterator


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    # Project entities
    PROJECT = "project"
    TASK = "task"
    FEATURE = "feature"
    BUG = "bug"
    
    # Code entities
    FILE = "file"
    MODULE = "module"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    
    # Agent entities
    AGENT = "agent"
    EXECUTION = "execution"
    DECISION = "decision"
    
    # Knowledge entities
    CONCEPT = "concept"
    PATTERN = "pattern"
    LEARNING = "learning"
    
    # People/Resources
    USER = "user"
    RESOURCE = "resource"
    DOCUMENTATION = "documentation"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Structural relationships
    CONTAINS = "contains"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    IMPORTS = "imports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CREATED_BY = "created_by"
    MODIFIED_BY = "modified_by"
    
    # Semantic relationships
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    EXAMPLE_OF = "example_of"
    
    # Action relationships
    EXECUTED_BY = "executed_by"
    ASSIGNED_TO = "assigned_to"
    BLOCKED_BY = "blocked_by"
    RESOLVES = "resolves"
    
    # Knowledge relationships
    LEARNED_FROM = "learned_from"
    APPLIES_TO = "applies_to"
    DERIVED_FROM = "derived_from"


@dataclass
class Entity:
    """
    A node in the knowledge graph.
    
    Represents any identifiable concept, object, or abstraction
    that agents need to reason about.
    """
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 to 1.0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False
    
    def add_observation(self, observation: str):
        """Add an observation about this entity."""
        self.observations.append(observation)
        self.updated_at = datetime.now()
    
    def update_property(self, key: str, value: Any):
        """Update a property."""
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "observations": self.observations,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "importance": self.importance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
            observations=data.get("observations", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            importance=data.get("importance", 0.5),
        )


@dataclass
class Relationship:
    """
    An edge in the knowledge graph.
    
    Represents a directed relationship between two entities.
    """
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Relationship strength
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Relationship):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class GraphQuery:
    """A query against the knowledge graph."""
    entity_types: Optional[List[EntityType]] = None
    relation_types: Optional[List[RelationType]] = None
    property_filters: Dict[str, Any] = field(default_factory=dict)
    text_search: Optional[str] = None
    max_depth: int = 3
    limit: int = 100


@dataclass
class TraversalResult:
    """Result of a graph traversal."""
    path: List[Entity]
    relationships: List[Relationship]
    total_weight: float = 0.0
    
    def __len__(self):
        return len(self.path)


class KnowledgeGraph:
    """
    In-memory knowledge graph for agent reasoning.
    
    Provides:
    - Entity and relationship CRUD
    - Graph traversal and queries
    - Dependency tracing
    - Pattern matching
    - Persistence to disk
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".agent_army" / "knowledge"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        
        # Indexes for fast lookup
        self._entities_by_type: Dict[EntityType, Set[str]] = {}
        self._outgoing_relations: Dict[str, Set[str]] = {}  # entity_id -> set of relationship_ids
        self._incoming_relations: Dict[str, Set[str]] = {}  # entity_id -> set of relationship_ids
        
        self._load_from_disk()
    
    # ==================== Entity Operations ====================
    
    def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
    ) -> Entity:
        """Create a new entity in the graph."""
        if entity_id is None:
            entity_id = self._generate_id(f"{entity_type.value}:{name}")
        
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        
        self._entities[entity.id] = entity
        
        # Update type index
        if entity_type not in self._entities_by_type:
            self._entities_by_type[entity_type] = set()
        self._entities_by_type[entity_type].add(entity.id)
        
        # Initialize relation indexes
        self._outgoing_relations[entity.id] = set()
        self._incoming_relations[entity.id] = set()
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self._entities_by_type.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
    ) -> Optional[Entity]:
        """Update an existing entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        
        if name:
            entity.name = name
        if properties:
            entity.properties.update(properties)
        if observation:
            entity.add_observation(observation)
        
        entity.updated_at = datetime.now()
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        
        # Remove all relationships involving this entity
        for rel_id in list(self._outgoing_relations.get(entity_id, set())):
            self._delete_relationship_internal(rel_id)
        for rel_id in list(self._incoming_relations.get(entity_id, set())):
            self._delete_relationship_internal(rel_id)
        
        # Remove from indexes
        if entity.entity_type in self._entities_by_type:
            self._entities_by_type[entity.entity_type].discard(entity_id)
        
        del self._entities[entity_id]
        self._outgoing_relations.pop(entity_id, None)
        self._incoming_relations.pop(entity_id, None)
        
        return True
    
    # ==================== Relationship Operations ====================
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> Optional[Relationship]:
        """Create a relationship between two entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        rel_id = self._generate_id(f"{source_id}:{relation_type.value}:{target_id}")
        
        relationship = Relationship(
            id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
        )
        
        self._relationships[rel_id] = relationship
        self._outgoing_relations[source_id].add(rel_id)
        self._incoming_relations[target_id].add(rel_id)
        
        return relationship
    
    def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        return self._relationships.get(rel_id)
    
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        rel_ids = set()
        
        if direction in ("outgoing", "both"):
            rel_ids.update(self._outgoing_relations.get(entity_id, set()))
        if direction in ("incoming", "both"):
            rel_ids.update(self._incoming_relations.get(entity_id, set()))
        
        relationships = [self._relationships[rid] for rid in rel_ids if rid in self._relationships]
        
        if relation_types:
            relationships = [r for r in relationships if r.relation_type in relation_types]
        
        return relationships
    
    def _delete_relationship_internal(self, rel_id: str):
        """Internal method to delete a relationship."""
        rel = self._relationships.get(rel_id)
        if rel:
            self._outgoing_relations.get(rel.source_id, set()).discard(rel_id)
            self._incoming_relations.get(rel.target_id, set()).discard(rel_id)
            del self._relationships[rel_id]
    
    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship."""
        if rel_id in self._relationships:
            self._delete_relationship_internal(rel_id)
            return True
        return False
    
    # ==================== Graph Traversal ====================
    
    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Entity]:
        """Get neighboring entities."""
        relationships = self.get_relationships(entity_id, direction, relation_types)
        neighbor_ids = set()
        
        for rel in relationships:
            if rel.source_id == entity_id:
                neighbor_ids.add(rel.target_id)
            else:
                neighbor_ids.add(rel.source_id)
        
        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[TraversalResult]:
        """Find a path between two entities (BFS)."""
        if start_id not in self._entities or end_id not in self._entities:
            return None
        
        if start_id == end_id:
            return TraversalResult(
                path=[self._entities[start_id]],
                relationships=[],
                total_weight=0,
            )
        
        # BFS
        visited = {start_id}
        queue: List[Tuple[str, List[str], List[str], float]] = [(start_id, [start_id], [], 0.0)]
        
        while queue:
            current_id, path, rel_path, weight = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for rel in self.get_relationships(current_id, "outgoing", relation_types):
                next_id = rel.target_id
                
                if next_id == end_id:
                    # Found!
                    return TraversalResult(
                        path=[self._entities[eid] for eid in path + [next_id]],
                        relationships=[self._relationships[rid] for rid in rel_path + [rel.id]],
                        total_weight=weight + rel.weight,
                    )
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((
                        next_id,
                        path + [next_id],
                        rel_path + [rel.id],
                        weight + rel.weight,
                    ))
        
        return None
    
    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        relation_types: Optional[List[RelationType]] = None,
        entity_types: Optional[List[EntityType]] = None,
    ) -> List[Entity]:
        """Traverse the graph from a starting entity."""
        if start_id not in self._entities:
            return []
        
        visited = {start_id}
        result = [self._entities[start_id]]
        current_level = {start_id}
        
        for _ in range(max_depth):
            next_level = set()
            
            for entity_id in current_level:
                neighbors = self.get_neighbors(entity_id, "outgoing", relation_types)
                
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        if entity_types is None or neighbor.entity_type in entity_types:
                            visited.add(neighbor.id)
                            result.append(neighbor)
                            next_level.add(neighbor.id)
            
            current_level = next_level
            if not current_level:
                break
        
        return result
    
    def get_dependencies(
        self,
        entity_id: str,
        max_depth: int = 5,
    ) -> List[Entity]:
        """Get all dependencies of an entity (transitive closure)."""
        return self.traverse(
            entity_id,
            max_depth=max_depth,
            relation_types=[RelationType.DEPENDS_ON, RelationType.IMPORTS],
        )
    
    def get_dependents(
        self,
        entity_id: str,
        max_depth: int = 5,
    ) -> List[Entity]:
        """Get all entities that depend on this entity."""
        if entity_id not in self._entities:
            return []
        
        dependents = []
        visited = {entity_id}
        current_level = {entity_id}
        
        for _ in range(max_depth):
            next_level = set()
            
            for eid in current_level:
                # Look at incoming DEPENDS_ON relationships
                for rel in self.get_relationships(eid, "incoming", [RelationType.DEPENDS_ON]):
                    source_id = rel.source_id
                    if source_id not in visited:
                        visited.add(source_id)
                        dependents.append(self._entities[source_id])
                        next_level.add(source_id)
            
            current_level = next_level
            if not current_level:
                break
        
        return dependents
    
    # ==================== Search & Query ====================
    
    def search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 20,
    ) -> List[Entity]:
        """Search entities by name or observations."""
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Check name
            if query_lower in entity.name.lower():
                results.append((entity, 1.0))
                continue
            
            # Check observations
            for obs in entity.observations:
                if query_lower in obs.lower():
                    results.append((entity, 0.7))
                    break
            
            # Check properties
            for key, value in entity.properties.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append((entity, 0.5))
                    break
        
        # Sort by score and importance
        results.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [r[0] for r in results[:limit]]
    
    def query(self, graph_query: GraphQuery) -> List[Entity]:
        """Execute a structured query against the graph."""
        results = list(self._entities.values())
        
        # Filter by entity type
        if graph_query.entity_types:
            results = [e for e in results if e.entity_type in graph_query.entity_types]
        
        # Filter by properties
        for key, value in graph_query.property_filters.items():
            results = [e for e in results if e.properties.get(key) == value]
        
        # Text search
        if graph_query.text_search:
            query_lower = graph_query.text_search.lower()
            results = [
                e for e in results
                if query_lower in e.name.lower() or
                any(query_lower in obs.lower() for obs in e.observations)
            ]
        
        return results[:graph_query.limit]
    
    # ==================== Analytics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_counts = {}
        for entity_type in EntityType:
            count = len(self._entities_by_type.get(entity_type, set()))
            if count > 0:
                entity_counts[entity_type.value] = count
        
        relation_counts = {}
        for rel in self._relationships.values():
            rel_type = rel.relation_type.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "entities_by_type": entity_counts,
            "relationships_by_type": relation_counts,
        }
    
    def get_most_connected(self, limit: int = 10) -> List[Tuple[Entity, int]]:
        """Get most connected entities (highest degree)."""
        degrees = []
        for entity_id, entity in self._entities.items():
            degree = (
                len(self._outgoing_relations.get(entity_id, set())) +
                len(self._incoming_relations.get(entity_id, set()))
            )
            degrees.append((entity, degree))
        
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:limit]
    
    # ==================== Persistence ====================
    
    def _generate_id(self, seed: str) -> str:
        """Generate a consistent ID from a seed string."""
        return hashlib.sha256(seed.encode()).hexdigest()[:12]
    
    def _load_from_disk(self):
        """Load graph from disk."""
        graph_file = self.storage_path / "knowledge_graph.json"
        if graph_file.exists():
            try:
                with open(graph_file, 'r') as f:
                    data = json.load(f)
                
                # Load entities
                for entity_data in data.get("entities", []):
                    entity = Entity.from_dict(entity_data)
                    self._entities[entity.id] = entity
                    
                    if entity.entity_type not in self._entities_by_type:
                        self._entities_by_type[entity.entity_type] = set()
                    self._entities_by_type[entity.entity_type].add(entity.id)
                    
                    self._outgoing_relations[entity.id] = set()
                    self._incoming_relations[entity.id] = set()
                
                # Load relationships
                for rel_data in data.get("relationships", []):
                    rel = Relationship.from_dict(rel_data)
                    self._relationships[rel.id] = rel
                    self._outgoing_relations[rel.source_id].add(rel.id)
                    self._incoming_relations[rel.target_id].add(rel.id)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load knowledge graph: {e}")
    
    def save(self):
        """Save graph to disk."""
        graph_file = self.storage_path / "knowledge_graph.json"
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relationships": [r.to_dict() for r in self._relationships.values()],
            "saved_at": datetime.now().isoformat(),
        }
        with open(graph_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear the entire graph."""
        self._entities.clear()
        self._relationships.clear()
        self._entities_by_type.clear()
        self._outgoing_relations.clear()
        self._incoming_relations.clear()
    
    # ==================== Convenience Methods ====================
    
    def add_task(
        self,
        name: str,
        description: str = "",
        depends_on: Optional[List[str]] = None,
    ) -> Entity:
        """Convenience method to add a task entity."""
        entity = self.create_entity(
            name=name,
            entity_type=EntityType.TASK,
            properties={"description": description, "status": "pending"},
        )
        
        if depends_on:
            for dep_id in depends_on:
                self.create_relationship(
                    entity.id,
                    dep_id,
                    RelationType.DEPENDS_ON,
                )
        
        return entity
    
    def add_learning(
        self,
        content: str,
        source: str = "",
        applies_to: Optional[List[str]] = None,
    ) -> Entity:
        """Convenience method to add a learning/pattern."""
        entity = self.create_entity(
            name=content[:50] + "..." if len(content) > 50 else content,
            entity_type=EntityType.LEARNING,
            properties={"content": content, "source": source},
        )
        
        if applies_to:
            for target_id in applies_to:
                self.create_relationship(
                    entity.id,
                    target_id,
                    RelationType.APPLIES_TO,
                )
        
        return entity
    
    def record_decision(
        self,
        decision: str,
        reason: str,
        agent_id: Optional[str] = None,
        context_entities: Optional[List[str]] = None,
    ) -> Entity:
        """Record an agent decision for explainability."""
        entity = self.create_entity(
            name=decision[:50] + "..." if len(decision) > 50 else decision,
            entity_type=EntityType.DECISION,
            properties={
                "decision": decision,
                "reason": reason,
                "agent_id": agent_id,
            },
        )
        
        if context_entities:
            for ctx_id in context_entities:
                self.create_relationship(
                    entity.id,
                    ctx_id,
                    RelationType.RELATED_TO,
                )
        
        return entity
