from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(50))
    doc_tier: Mapped[int] = mapped_column(Integer, default=1)
    category: Mapped[Optional[str]] = mapped_column(String(100))
    source_id: Mapped[Optional[str]] = mapped_column(String(200))
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSONB,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    intent: Mapped[Optional[str]] = mapped_column(String(50))
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    csat_score: Mapped[Optional[int]] = mapped_column(Integer)
    nps_score: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    agent_spans: Mapped[List["AgentSpan"]] = relationship(back_populates="session")
    outcomes: Mapped[List["Outcome"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    evaluation_scores: Mapped[List["EvaluationScore"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSONB,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    session: Mapped[Session] = relationship(back_populates="messages")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    order_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[Optional[str]] = mapped_column(String(50))
    items: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    total_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    estimated_delivery: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="SET NULL"),
    )
    issue_type: Mapped[Optional[str]] = mapped_column(String(100))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[Optional[str]] = mapped_column(String(50), default="open")
    priority: Mapped[Optional[str]] = mapped_column(String(20), default="normal")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class AgentSpan(Base):
    __tablename__ = "agent_spans"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="SET NULL"),
    )
    trace_id: Mapped[Optional[str]] = mapped_column(String(100))
    span_name: Mapped[str] = mapped_column(String(100), nullable=False)
    attributes: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    session: Mapped[Optional[Session]] = relationship(back_populates="agent_spans")


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="CASCADE"),
    )
    task: Mapped[str] = mapped_column(String(100), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    session: Mapped[Optional[Session]] = relationship(back_populates="outcomes")


class EvaluationScore(Base):
    __tablename__ = "evaluation_scores"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="CASCADE"),
    )
    groundedness: Mapped[Optional[float]] = mapped_column(Float)
    hallucination: Mapped[Optional[bool]] = mapped_column(Boolean)
    helpfulness: Mapped[Optional[float]] = mapped_column(Float)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    session: Mapped[Optional[Session]] = relationship(back_populates="evaluation_scores")


class IntentEvalRun(Base):
    __tablename__ = "intent_eval_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    experiment_name: Mapped[Optional[str]] = mapped_column(String(200))
    dataset_key: Mapped[Optional[str]] = mapped_column(String(50))
    model_provider: Mapped[Optional[str]] = mapped_column(String(50))
    model_name: Mapped[Optional[str]] = mapped_column(String(200))
    prompt_version: Mapped[Optional[str]] = mapped_column(String(100))
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    macro_precision: Mapped[Optional[float]] = mapped_column(Float)
    macro_recall: Mapped[Optional[float]] = mapped_column(Float)
    macro_f1: Mapped[Optional[float]] = mapped_column(Float)
    total_examples: Mapped[int] = mapped_column(Integer)
    correct_examples: Mapped[int] = mapped_column(Integer)
    failed_examples: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    predictions: Mapped[List["IntentEvalPrediction"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class IntentEvalPrediction(Base):
    __tablename__ = "intent_eval_predictions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("intent_eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    test_id: Mapped[Optional[str]] = mapped_column(String(200))
    split: Mapped[Optional[str]] = mapped_column(String(100))
    question: Mapped[Optional[str]] = mapped_column(Text)
    expected_intent: Mapped[Optional[str]] = mapped_column(String(100))
    predicted_intent: Mapped[Optional[str]] = mapped_column(String(100))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    run: Mapped[IntentEvalRun] = relationship(back_populates="predictions")
    session: Mapped[Optional[Session]] = relationship()


__all__ = [
    "Base",
    "Document",
    "Session",
    "Message",
    "Order",
    "Ticket",
    "AgentSpan",
    "Outcome",
    "EvaluationScore",
    "IntentEvalRun",
    "IntentEvalPrediction",
]

