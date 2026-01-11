"""
OBOC Observability Layer

Implements monitoring and observability for multi-agent systems:

┌────────────────────────────────────────────────────────────────┐
│                    TRACE LAYER                                  │
│   • Complete execution journeys                                 │
│   • LLM queries, tool calls, agent communications              │
│   • Context propagation across agents                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    METRIC LAYER                                 │
│   • Task completion rates                                       │
│   • Latency distributions                                       │
│   • Token consumption                                           │
│   • Error rates by agent/tool                                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                             │
│   • LLM-as-a-judge scoring                                      │
│   • Human feedback loops                                        │
│   • A/B testing frameworks                                      │
│   • Drift detection                                             │
└────────────────────────────────────────────────────────────────┘
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
import statistics


class SpanType(Enum):
    """Types of spans in a trace."""
    ORCHESTRATION = "orchestration"
    AGENT_EXECUTION = "agent_execution"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_OP = "memory_operation"
    KNOWLEDGE_OP = "knowledge_operation"


@dataclass
class Span:
    """A single span in an execution trace."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: SpanType
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, success, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return 0
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })
    
    def end(self, status: str = "success", error: Optional[str] = None):
        """End the span."""
        self.ended_at = datetime.now()
        self.status = status
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


@dataclass
class Trace:
    """A complete execution trace."""
    trace_id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str = "in_progress"
    spans: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return 0
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)
    
    def end(self, status: str = "success"):
        """End the trace."""
        self.ended_at = datetime.now()
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
            "attributes": self.attributes,
        }


class Tracer:
    """
    Distributed tracing for OBOC orchestration.
    
    Provides:
    - Span creation and management
    - Context propagation
    - Trace export
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".agent_army" / "traces"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._traces: Dict[str, Trace] = {}
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None
        self._span_counter = 0
    
    def start_trace(
        self,
        trace_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """Start a new trace."""
        trace = Trace(
            trace_id=trace_id,
            name=name,
            started_at=datetime.now(),
            attributes=attributes or {},
        )
        self._traces[trace_id] = trace
        self._current_trace_id = trace_id
        return trace
    
    def start_span(
        self,
        name: str,
        span_type: SpanType,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Start a new span within the current trace."""
        if not self._current_trace_id:
            raise ValueError("No active trace")
        
        self._span_counter += 1
        span_id = f"span_{self._span_counter}"
        
        span = Span(
            span_id=span_id,
            trace_id=self._current_trace_id,
            parent_span_id=parent_span_id or self._current_span_id,
            name=name,
            span_type=span_type,
            started_at=datetime.now(),
            attributes=attributes or {},
        )
        
        self._traces[self._current_trace_id].add_span(span)
        self._current_span_id = span_id
        
        return span
    
    def end_span(
        self,
        span: Span,
        status: str = "success",
        error: Optional[str] = None,
    ):
        """End a span."""
        span.end(status, error)
        
        # Reset current span to parent
        if span.parent_span_id:
            self._current_span_id = span.parent_span_id
    
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        status: str = "success",
    ):
        """End a trace."""
        tid = trace_id or self._current_trace_id
        if tid and tid in self._traces:
            self._traces[tid].end(status)
            self._save_trace(self._traces[tid])
        
        if tid == self._current_trace_id:
            self._current_trace_id = None
            self._current_span_id = None
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self._traces.get(trace_id)
    
    def _save_trace(self, trace: Trace):
        """Save trace to disk."""
        trace_file = self.storage_path / f"{trace.trace_id}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics collection for OBOC orchestration.
    
    Key Metrics to Track:
    - Performance: Task completion rate, latency P50/P95/P99
    - Quality: Accuracy, relevance scores, hallucination rate
    - Cost: Tokens per task, cost per completion
    - Reliability: Error rate, retry rate, circuit breaker trips
    - Tool Usage: Tool selection accuracy, parameter validity
    """
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._metric_history: List[Metric] = []
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        self._record(name, MetricType.COUNTER, self._counters[key], labels)
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._record(name, MetricType.GAUGE, value, labels)
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        self._record(name, MetricType.HISTOGRAM, value, labels)
    
    def timer(
        self,
        name: str,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a timer observation."""
        key = self._make_key(name, labels)
        self._timers[key].append(duration_ms)
        self._record(name, MetricType.TIMER, duration_ms, labels)
    
    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]],
    ) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _record(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]],
    ):
        """Record a metric observation."""
        self._metric_history.append(Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
        ))
        
        # Keep only last 10000 records
        if len(self._metric_history) > 10000:
            self._metric_history = self._metric_history[-10000:]
    
    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)
    
    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }
    
    def get_timer_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get timer statistics."""
        key = self._make_key(name, labels)
        values = self._timers.get(key, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": statistics.mean(values),
            "p50_ms": statistics.median(values),
            "p95_ms": self._percentile(values, 95),
            "p99_ms": self._percentile(values, 99),
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: self.get_histogram_stats(k)
                for k in self._histograms.keys()
            },
            "timers": {
                k: self.get_timer_stats(k)
                for k in self._timers.keys()
            },
        }


@dataclass
class EvaluationResult:
    """Result of an LLM evaluation."""
    evaluator: str
    dimension: str
    score: float  # 0.0 to 1.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """
    LLM-as-a-Judge Pattern for quality evaluation.
    
    Workflow:
    1. Create "golden dataset" with human expert labels
    2. Train LLM evaluator to match human judgments
    3. Deploy evaluator for automated scoring
    4. Continuous calibration against human samples
    
    Evaluation Dimensions:
    - Coherence
    - Relevance
    - Accuracy
    - Appropriateness
    - Task completion
    """
    
    DIMENSIONS = [
        "coherence",
        "relevance",
        "accuracy",
        "appropriateness",
        "task_completion",
    ]
    
    def __init__(self):
        self._golden_dataset: List[Dict[str, Any]] = []
        self._evaluation_history: List[EvaluationResult] = []
        self._human_feedback: List[Dict[str, Any]] = []
    
    async def evaluate(
        self,
        task: str,
        response: str,
        dimension: str = "task_completion",
        context: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a response using LLM-as-a-judge.
        
        In production, this would call an LLM to evaluate.
        For now, returns a placeholder result.
        """
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        # Placeholder evaluation
        # In production, this would use an LLM
        result = EvaluationResult(
            evaluator="placeholder",
            dimension=dimension,
            score=0.8,  # Placeholder score
            reasoning="Placeholder evaluation",
        )
        
        self._evaluation_history.append(result)
        return result
    
    def add_golden_example(
        self,
        task: str,
        response: str,
        human_score: float,
        dimension: str,
    ):
        """Add a golden example for calibration."""
        self._golden_dataset.append({
            "task": task,
            "response": response,
            "human_score": human_score,
            "dimension": dimension,
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_human_feedback(
        self,
        trace_id: str,
        rating: float,
        comment: Optional[str] = None,
    ):
        """Add human feedback for a specific execution."""
        self._human_feedback.append({
            "trace_id": trace_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        if not self._evaluation_history:
            return {"evaluations": 0}
        
        scores = [e.score for e in self._evaluation_history]
        
        return {
            "evaluations": len(self._evaluation_history),
            "golden_examples": len(self._golden_dataset),
            "human_feedback": len(self._human_feedback),
            "mean_score": statistics.mean(scores) if scores else 0,
            "score_distribution": {
                "low": sum(1 for s in scores if s < 0.3),
                "medium": sum(1 for s in scores if 0.3 <= s < 0.7),
                "high": sum(1 for s in scores if s >= 0.7),
            },
        }


class ObservabilityManager:
    """
    Unified observability manager for OBOC.
    
    Combines:
    - Tracing (execution journeys)
    - Metrics (performance, quality, cost)
    - Evaluation (LLM-as-judge)
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.tracer = Tracer(storage_path)
        self.metrics = MetricsCollector()
        self.evaluator = Evaluator()
    
    def start_execution(
        self,
        trace_id: str,
        task: str,
        pattern: str,
    ) -> Trace:
        """Start tracking an execution."""
        trace = self.tracer.start_trace(
            trace_id,
            name=task[:50],
            attributes={"pattern": pattern},
        )
        
        self.metrics.increment("executions_total", labels={"pattern": pattern})
        self.metrics.gauge("active_executions", 1)
        
        return trace
    
    def end_execution(
        self,
        trace_id: str,
        success: bool,
        duration_ms: float,
        cost: float = 0.0,
        tokens: int = 0,
    ):
        """End tracking an execution."""
        status = "success" if success else "error"
        self.tracer.end_trace(trace_id, status)
        
        self.metrics.increment(f"executions_{status}")
        self.metrics.timer("execution_duration", duration_ms)
        self.metrics.histogram("execution_cost", cost)
        self.metrics.histogram("execution_tokens", tokens)
        self.metrics.gauge("active_executions", 0)
    
    def record_agent_execution(
        self,
        agent_name: str,
        tier: str,
        duration_ms: float,
        success: bool,
        tokens: int = 0,
    ):
        """Record an agent execution."""
        labels = {"agent": agent_name, "tier": tier}
        
        self.metrics.increment("agent_executions_total", labels=labels)
        self.metrics.timer("agent_duration", duration_ms, labels=labels)
        
        if not success:
            self.metrics.increment("agent_errors_total", labels=labels)
    
    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost: float,
    ):
        """Record an LLM API call."""
        labels = {"model": model}
        
        self.metrics.increment("llm_calls_total", labels=labels)
        self.metrics.histogram("llm_input_tokens", input_tokens, labels=labels)
        self.metrics.histogram("llm_output_tokens", output_tokens, labels=labels)
        self.metrics.timer("llm_latency", duration_ms, labels=labels)
        self.metrics.histogram("llm_cost", cost, labels=labels)
    
    def record_circuit_breaker_event(
        self,
        name: str,
        event: str,  # trip, reset, half_open
    ):
        """Record a circuit breaker state change."""
        self.metrics.increment(
            f"circuit_breaker_{event}",
            labels={"name": name},
        )
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get a dashboard view of all observability data."""
        return {
            "traces": len(self.tracer._traces),
            "metrics": self.metrics.get_summary(),
            "evaluation": self.evaluator.get_calibration_stats(),
        }
