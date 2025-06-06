"""
Database Models for OmniBio
SQLAlchemy models for PostgreSQL database tables.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, 
    DECIMAL, BIGINT, JSON, ForeignKey, Table, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

# Junction table for job-file many-to-many relationship
job_files_table = Table(
    'job_files',
    Base.metadata,
    Column('job_id', Integer, ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True),
    Column('file_id', Integer, ForeignKey('files.id', ondelete='CASCADE'), primary_key=True)
)


class User(Base):
    """User model for API access management."""
    
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    permissions = Column(JSONB, default={"read": True, "write": True, "admin": False})
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="created_by_user")
    files = relationship("File", back_populates="uploaded_by_user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class ApiKey(Base):
    """API Key model for managing multiple keys per user."""
    
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(255), unique=True, nullable=False)
    key_name = Column(String(100), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    permissions = Column(JSONB, default={"read": True, "write": True})
    expires_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(name='{self.key_name}', user_id='{self.user_id}')>"


class Job(Base):
    """Job model for tracking analysis jobs."""
    
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    project_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default='pending')
    analysis_types = Column(ARRAY(String), nullable=False)
    file_ids = Column(ARRAY(String), nullable=False)
    group_column = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    progress = Column(DECIMAL(5, 2), default=0.0)
    message = Column(Text, default='Queued')
    error_message = Column(Text)
    results_path = Column(Text)
    results_data = Column(JSONB)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    created_by_user = relationship("User", back_populates="jobs")
    files = relationship("File", secondary=job_files_table, back_populates="jobs")
    
    def __repr__(self):
        return f"<Job(analysis_id='{self.analysis_id}', status='{self.status}', project='{self.project_name}')>"


class File(Base):
    """File model for tracking uploaded files."""
    
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(255), unique=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(Text, nullable=False)
    size_bytes = Column(BIGINT, nullable=False)
    checksum = Column(String(64))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    status = Column(String(50), default='uploaded')
    file_metadata = Column(JSONB)
    
    # Relationships
    uploaded_by_user = relationship("User", back_populates="files")
    jobs = relationship("Job", secondary=job_files_table, back_populates="files")
    
    def __repr__(self):
        return f"<File(file_id='{self.file_id}', filename='{self.original_filename}', type='{self.file_type}')>"


# Utility functions for database operations
def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all tables in the database."""
    Base.metadata.drop_all(bind=engine) 