from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship


# Keep models compatible with SQLAlchemy 1.4 and 2.x.
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    content = Column(Text, nullable=False)
    parent_id = Column(UUID(as_uuid=False), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    embedding = Column(Vector(768), nullable=True)
    source = Column(String(50))
    doc_tier = Column(Integer, default=1)
    category = Column(String(100))
    source_id = Column(String(200))
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(String(100))
    intent = Column(String(50))
    escalated = Column(Boolean, default=False, server_default=text("false"))
    resolved_at = Column(DateTime(timezone=True))
    csat_score = Column(Integer)
    nps_score = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    agent_spans = relationship("AgentSpan", back_populates="session")
    outcomes = relationship("Outcome", back_populates="session", cascade="all, delete-orphan")
    evaluation_scores = relationship("EvaluationScore", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="messages")


class Order(Base):
    __tablename__ = "orders"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    order_number = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(100))
    status = Column(String(50))
    items = Column(JSONB)
    total_amount = Column(Numeric(10, 2))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    estimated_delivery = Column(DateTime(timezone=True), nullable=True)


class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="SET NULL"))
    issue_type = Column(String(100))
    summary = Column(Text)
    status = Column(String(50), default="open")
    priority = Column(String(20), default="normal")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentSpan(Base):
    __tablename__ = "agent_spans"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="SET NULL"))
    trace_id = Column(String(100))
    span_name = Column(String(100), nullable=False)
    attributes = Column(JSONB)
    latency_ms = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="agent_spans")


class Outcome(Base):
    __tablename__ = "outcomes"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"))
    task = Column(String(100), nullable=False)
    completed = Column(Boolean, nullable=False, default=False)
    escalated = Column(Boolean, nullable=False, default=False)
    verified = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="outcomes")


class EvaluationScore(Base):
    __tablename__ = "evaluation_scores"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"))
    groundedness = Column(Float)
    hallucination = Column(Boolean)
    helpfulness = Column(Float)
    metadata_ = Column("metadata", JSONB)
    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="evaluation_scores")


class IntentEvalRun(Base):
    __tablename__ = "intent_eval_runs"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    experiment_name = Column(String(200))
    dataset_key = Column(String(50))
    model_provider = Column(String(50))
    model_name = Column(String(200))
    prompt_version = Column(String(100))
    metadata_ = Column("metadata", JSONB)
    accuracy = Column(Float)
    macro_precision = Column(Float)
    macro_recall = Column(Float)
    macro_f1 = Column(Float)
    total_examples = Column(Integer)
    correct_examples = Column(Integer)
    failed_examples = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("IntentEvalPrediction", back_populates="run", cascade="all, delete-orphan")


class IntentEvalPrediction(Base):
    __tablename__ = "intent_eval_predictions"
    id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    run_id = Column(UUID(as_uuid=False), ForeignKey("intent_eval_runs.id", ondelete="CASCADE"), nullable=False)
    test_id = Column(String(200))
    split = Column(String(100))
    question = Column(Text)
    expected_intent = Column(String(100))
    predicted_intent = Column(String(100))
    is_correct = Column(Boolean)
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("IntentEvalRun", back_populates="predictions")
    session = relationship("Session")


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

