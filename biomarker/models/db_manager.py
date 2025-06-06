"""
Database Connection Manager for OmniBio
Handles database connections, sessions, and CRUD operations.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from databases import Database
import hashlib
from datetime import datetime

from .database import Base, User, ApiKey, Job, File
from .config import get_settings


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database.url
        
        # Create engines
        self.engine = create_async_engine(self.database_url, echo=self.settings.debug)
        self.async_session_maker = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Database instance for raw queries
        self.database = Database(self.database_url)
    
    async def connect(self):
        """Connect to the database."""
        await self.database.connect()
        
    async def disconnect(self):
        """Disconnect from the database."""
        await self.database.disconnect()
        await self.engine.dispose()
    
    async def create_tables(self):
        """Create all tables in the database."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all tables in the database."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self):
        """Get an async database session."""
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    # User operations
    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE api_key = :api_key AND is_active = true"),
                {"api_key": api_key}
            )
            row = result.fetchone()
            if row:
                return User(**dict(row._mapping))
            return None
    
    async def create_user(self, username: str, email: str, api_key: str, permissions: Dict[str, Any] = None) -> User:
        """Create a new user."""
        if permissions is None:
            permissions = {"read": True, "write": True, "admin": False}
            
        async with self.get_session() as session:
            user = User(
                username=username,
                email=email,
                api_key=api_key,
                permissions=permissions
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        async with self.get_session() as session:
            await session.execute(
                text("UPDATE users SET last_login = :now WHERE id = :user_id"),
                {"now": datetime.utcnow(), "user_id": user_id}
            )
            await session.commit()
    
    # File operations
    async def create_file_record(self, file_info: Dict[str, Any], uploaded_by: Optional[str] = None) -> File:
        """Create a file record in the database."""
        async with self.get_session() as session:
            file_record = File(
                file_id=file_info["file_id"],
                original_filename=file_info["filename"],
                file_type=file_info["file_type"],
                file_path=file_info.get("file_path", ""),
                size_bytes=file_info["size_bytes"],
                uploaded_by=uploaded_by,
                file_metadata=file_info.get("metadata", {})
            )
            session.add(file_record)
            await session.commit()
            await session.refresh(file_record)
            return file_record
    
    async def get_file_by_id(self, file_id: str) -> Optional[File]:
        """Get file record by file ID."""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM files WHERE file_id = :file_id"),
                {"file_id": file_id}
            )
            row = result.fetchone()
            if row:
                return File(**dict(row._mapping))
            return None
    
    async def list_files(self, uploaded_by: Optional[str] = None, limit: int = 100) -> List[File]:
        """List files with optional filtering."""
        async with self.get_session() as session:
            if uploaded_by:
                result = await session.execute(
                    text("SELECT * FROM files WHERE uploaded_by = :uploaded_by ORDER BY uploaded_at DESC LIMIT :limit"),
                    {"uploaded_by": uploaded_by, "limit": limit}
                )
            else:
                result = await session.execute(
                    text("SELECT * FROM files ORDER BY uploaded_at DESC LIMIT :limit"),
                    {"limit": limit}
                )
            
            rows = result.fetchall()
            return [File(**dict(row._mapping)) for row in rows]
    
    # Job operations
    async def create_job(self, job_data: Dict[str, Any], created_by: Optional[str] = None) -> Job:
        """Create a new analysis job."""
        async with self.get_session() as session:
            job = Job(
                analysis_id=job_data["analysis_id"],
                project_name=job_data["project_name"],
                analysis_types=job_data["analysis_types"],
                file_ids=job_data["file_ids"],
                group_column=job_data.get("group_column"),
                created_by=created_by
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job
    
    async def get_job_by_analysis_id(self, analysis_id: str) -> Optional[Job]:
        """Get job by analysis ID."""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM jobs WHERE analysis_id = :analysis_id"),
                {"analysis_id": analysis_id}
            )
            row = result.fetchone()
            if row:
                return Job(**dict(row._mapping))
            return None
    
    async def update_job_status(self, analysis_id: str, status: str, progress: float = None, 
                              message: str = None, error_message: str = None, 
                              results_data: Dict[str, Any] = None):
        """Update job status and progress."""
        updates = {"status": status}
        
        if progress is not None:
            updates["progress"] = progress
        if message is not None:
            updates["message"] = message
        if error_message is not None:
            updates["error_message"] = error_message
        if results_data is not None:
            updates["results_data"] = results_data
        
        # Set timestamps based on status
        if status == "running" and "started_at" not in updates:
            updates["started_at"] = datetime.utcnow()
        elif status in ["completed", "failed"] and "completed_at" not in updates:
            updates["completed_at"] = datetime.utcnow()
        
        # Build dynamic query
        set_clauses = []
        params = {"analysis_id": analysis_id}
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = :{key}")
            params[key] = value
        
        query = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE analysis_id = :analysis_id"
        
        async with self.get_session() as session:
            await session.execute(text(query), params)
            await session.commit()
    
    async def list_jobs(self, created_by: Optional[str] = None, status: Optional[str] = None, 
                       limit: int = 100) -> List[Job]:
        """List jobs with optional filtering."""
        conditions = []
        params = {"limit": limit}
        
        if created_by:
            conditions.append("created_by = :created_by")
            params["created_by"] = created_by
        
        if status:
            conditions.append("status = :status")
            params["status"] = status
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM jobs {where_clause} ORDER BY created_at DESC LIMIT :limit"
        
        async with self.get_session() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()
            return [Job(**dict(row._mapping)) for row in rows]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Test connection
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            
            # Get some stats
            async with self.get_session() as session:
                # Count tables
                users_count = await session.execute(text("SELECT COUNT(*) FROM users"))
                jobs_count = await session.execute(text("SELECT COUNT(*) FROM jobs"))
                files_count = await session.execute(text("SELECT COUNT(*) FROM files"))
                
                return {
                    "status": "healthy",
                    "connected": True,
                    "users_count": users_count.scalar(),
                    "jobs_count": jobs_count.scalar(),
                    "files_count": files_count.scalar()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def init_database():
    """Initialize the database connection."""
    db_manager = get_database_manager()
    await db_manager.connect()
    return db_manager


async def close_database():
    """Close the database connection."""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None 