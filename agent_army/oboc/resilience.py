"""
OBOC Resilience Module

Implements fault tolerance patterns for production multi-agent systems:

1. CIRCUIT BREAKER: Stop calling failing services
2. RETRY POLICIES: Exponential backoff with jitter
3. GRACEFUL DEGRADATION: Fallback behaviors
4. STATE PRESERVATION: Separate permanent from temporary state

Error Handling Best Practices:
- Transient network errors: Retry with exponential backoff
- Rate limiting (429): Respect Retry-After header
- Authentication failure: Fail immediately
- Invalid input: Return error to caller
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from functools import wraps


class CircuitState(Enum):
    """States of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Block calls, fail fast
    HALF_OPEN = "half_open"  # Allow limited calls to test recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing from half-open
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 3
    initial_backoff: float = 0.1  # 100ms
    max_backoff: float = 10.0  # 10 seconds
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.2
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])
    non_retryable_exceptions: List[type] = field(default_factory=lambda: [
        ValueError, TypeError, KeyError
    ])


class CircuitBreaker:
    """
    CIRCUIT BREAKER PATTERN
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    Circuit Breaker                           │
    │                                                              │
    │   CLOSED ──(failures exceed threshold)──► OPEN              │
    │      │                                       │               │
    │      │                              (timeout expires)        │
    │      │                                       │               │
    │   ◄──────────(success)────── HALF-OPEN ◄────┘               │
    │                                                              │
    │   CLOSED: Normal operation                                   │
    │   OPEN: Block calls, fail fast                              │
    │   HALF-OPEN: Allow limited calls to test recovery           │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current state, handling timeout transitions."""
        if self._state == CircuitState.OPEN:
            if self._should_try_reset():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state
    
    def _should_try_reset(self) -> bool:
        """Check if timeout has expired for OPEN state."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            return self._half_open_calls < self.config.half_open_max_calls
    
    def record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._reset()
        else:
            self._failure_count = 0
    
    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._failure_count >= self.config.failure_threshold:
            self._trip()
    
    def _trip(self):
        """Trip the circuit breaker to OPEN state."""
        self._state = CircuitState.OPEN
        self._success_count = 0
    
    def _reset(self):
        """Reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    async def execute(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute a function with circuit breaker protection."""
        if not self.can_execute():
            if fallback:
                return await self._call_if_async(fallback, *args, **kwargs)
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
        
        try:
            result = await self._call_if_async(func, *args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            if fallback:
                return await self._call_if_async(fallback, *args, **kwargs)
            raise
    
    async def _call_if_async(self, func: Callable, *args, **kwargs) -> Any:
        """Call function, handling both sync and async."""
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryPolicy:
    """
    RETRY POLICY with Exponential Backoff and Jitter
    
    wait_time = min(base_delay × 2^attempt, max_delay) + random_jitter
    
    Retry Decision Matrix:
    - Transient network errors: ✅ Retry
    - Rate limiting (429): ✅ Retry (respect Retry-After)
    - Authentication failure: ❌ No retry
    - Invalid input: ❌ No retry
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base_delay = self.config.initial_backoff * (self.config.backoff_multiplier ** attempt)
        delay = min(base_delay, self.config.max_backoff)
        
        # Add jitter
        jitter = delay * self.config.jitter_factor * random.random()
        
        return delay + jitter
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception type."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with retry policy."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt + 1):
                    raise
                
                delay = self.calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        raise last_exception


T = TypeVar('T')


@dataclass
class DegradedResult(Generic[T]):
    """Result from a potentially degraded operation."""
    value: T
    is_degraded: bool = False
    degradation_level: str = "full"  # full, partial, minimal
    message: Optional[str] = None


class GracefulDegradation:
    """
    GRACEFUL DEGRADATION
    
    Full Functionality → Partial Functionality → Core Only → Safe Mode
    
    Example (Document Processing):
    - Full: Extract → Validate → Enrich → Route
    - Degraded: Extract → Validate → Route (skip enrichment if agent fails)
    - Core: Extract → Route (mark as "reduced confidence")
    """
    
    def __init__(
        self,
        levels: Optional[List[str]] = None,
    ):
        self.levels = levels or ["full", "partial", "core", "minimal"]
        self._current_level = "full"
        self._fallbacks: Dict[str, Callable] = {}
    
    @property
    def current_level(self) -> str:
        return self._current_level
    
    def register_fallback(
        self,
        level: str,
        fallback: Callable,
    ):
        """Register a fallback for a degradation level."""
        if level not in self.levels:
            raise ValueError(f"Unknown degradation level: {level}")
        self._fallbacks[level] = fallback
    
    def degrade(self) -> str:
        """Move to the next degradation level."""
        current_idx = self.levels.index(self._current_level)
        if current_idx < len(self.levels) - 1:
            self._current_level = self.levels[current_idx + 1]
        return self._current_level
    
    def recover(self) -> str:
        """Try to recover to a better level."""
        current_idx = self.levels.index(self._current_level)
        if current_idx > 0:
            self._current_level = self.levels[current_idx - 1]
        return self._current_level
    
    def reset(self):
        """Reset to full functionality."""
        self._current_level = "full"
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> DegradedResult:
        """Execute with graceful degradation."""
        current_level = self._current_level
        
        while current_level in self.levels:
            try:
                if current_level == "full":
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return DegradedResult(
                        value=result,
                        is_degraded=False,
                        degradation_level="full",
                    )
                else:
                    # Use fallback
                    fallback = self._fallbacks.get(current_level)
                    if fallback:
                        result = fallback(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return DegradedResult(
                            value=result,
                            is_degraded=True,
                            degradation_level=current_level,
                            message=f"Degraded to {current_level} functionality",
                        )
                    else:
                        # No fallback, try next level
                        current_level = self.degrade()
                        
            except Exception as e:
                current_level = self.degrade()
        
        # Minimal level failed
        return DegradedResult(
            value=None,
            is_degraded=True,
            degradation_level="failed",
            message="All degradation levels exhausted",
        )


@dataclass
class StateCheckpoint:
    """A checkpoint of execution state."""
    checkpoint_id: str
    task_id: str
    step_index: int
    state_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    is_recoverable: bool = True


class StatePreservation:
    """
    STATE PRESERVATION STRATEGY
    
    Principle: Separate permanent learning from temporary task state
    
    Permanent State (Preserve):
    - Document patterns learned
    - User preferences discovered
    - Successful solution approaches
    
    Temporary State (Discard on failure):
    - Current task variables
    - Intermediate computations
    - Transient working memory
    """
    
    def __init__(self):
        self._checkpoints: Dict[str, StateCheckpoint] = {}
        self._permanent_state: Dict[str, Any] = {}
        self._temporary_state: Dict[str, Any] = {}
    
    def create_checkpoint(
        self,
        task_id: str,
        step_index: int,
        state_data: Dict[str, Any],
    ) -> StateCheckpoint:
        """Create a checkpoint for recovery."""
        checkpoint = StateCheckpoint(
            checkpoint_id=f"{task_id}_{step_index}_{datetime.now().timestamp()}",
            task_id=task_id,
            step_index=step_index,
            state_data=state_data.copy(),
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint
    
    def get_latest_checkpoint(
        self,
        task_id: str,
    ) -> Optional[StateCheckpoint]:
        """Get the latest checkpoint for a task."""
        checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.task_id == task_id and cp.is_recoverable
        ]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: cp.created_at)
    
    def save_permanent(self, key: str, value: Any):
        """Save to permanent state (survives failures)."""
        self._permanent_state[key] = value
    
    def get_permanent(self, key: str, default: Any = None) -> Any:
        """Get from permanent state."""
        return self._permanent_state.get(key, default)
    
    def save_temporary(self, key: str, value: Any):
        """Save to temporary state (discarded on failure)."""
        self._temporary_state[key] = value
    
    def get_temporary(self, key: str, default: Any = None) -> Any:
        """Get from temporary state."""
        return self._temporary_state.get(key, default)
    
    def discard_temporary(self):
        """Discard all temporary state (on failure)."""
        self._temporary_state.clear()
    
    def promote_to_permanent(self, key: str):
        """Promote a temporary value to permanent."""
        if key in self._temporary_state:
            self._permanent_state[key] = self._temporary_state.pop(key)


class ResilienceManager:
    """
    Unified resilience manager combining all patterns.
    
    Provides:
    - Circuit breakers per service
    - Retry policies
    - Graceful degradation
    - State preservation
    """
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retry_policy = RetryPolicy()
        self._degradation = GracefulDegradation()
        self._state = StatePreservation()
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
    
    async def execute_with_resilience(
        self,
        name: str,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        use_retry: bool = True,
        use_circuit_breaker: bool = True,
        **kwargs,
    ) -> Any:
        """Execute a function with full resilience stack."""
        circuit = self.get_circuit_breaker(name) if use_circuit_breaker else None
        
        async def wrapped():
            if use_retry:
                return await self._retry_policy.execute(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
        
        if circuit:
            return await circuit.execute(wrapped, fallback=fallback)
        else:
            try:
                return await wrapped()
            except Exception:
                if fallback:
                    result = fallback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall resilience status."""
        return {
            "circuit_breakers": {
                name: cb.get_status()
                for name, cb in self._circuit_breakers.items()
            },
            "degradation_level": self._degradation.current_level,
            "checkpoints": len(self._state._checkpoints),
        }
