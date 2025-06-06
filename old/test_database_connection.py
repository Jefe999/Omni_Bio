#!/usr/bin/env python3
"""
Test Database Connection and Setup
Verifies that PostgreSQL connection works and creates initial data.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from biomarker.models.config import get_settings
from biomarker.models.db_manager import DatabaseManager
from biomarker.models.database import User
from biomarker.api.auth import generate_api_key


async def test_database_connection():
    """Test database connection and basic operations."""
    print("🔄 Testing database connection...")
    
    # Get settings
    settings = get_settings()
    print(f"Database URL: {settings.database.url}")
    
    # Create database manager
    db_manager = DatabaseManager()
    
    try:
        # Connect to database
        await db_manager.connect()
        print("✅ Database connected successfully!")
        
        # Create tables
        await db_manager.create_tables()
        print("✅ Database tables created!")
        
        # Test health check
        health = await db_manager.health_check()
        print(f"🏥 Database health: {health}")
        
        # Create a test user
        api_key = generate_api_key()
        user = await db_manager.create_user(
            username="test_user",
            email="test@omnibio.com",
            api_key=api_key,
            permissions={"read": True, "write": True, "admin": False}
        )
        print(f"👤 Created test user: {user.username} with API key: {api_key}")
        
        # Test user lookup
        found_user = await db_manager.get_user_by_api_key(api_key)
        if found_user:
            print(f"✅ User lookup successful: {found_user.username}")
        else:
            print("❌ User lookup failed")
        
        # List users
        print("\n📊 Database Statistics:")
        health = await db_manager.health_check()
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await db_manager.disconnect()


if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    sys.exit(0 if success else 1) 