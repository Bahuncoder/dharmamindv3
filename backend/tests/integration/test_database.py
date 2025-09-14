"""
🗄️ Database Integration Tests
============================

Comprehensive database testing for DharmaMind:
- Database operations and transactions
- Model relationships and integrity
- Database performance testing
- Migration testing
- Connection pooling tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import text, select, func
from sqlalchemy.exc import IntegrityError
from unittest.mock import patch

from app.models.user import User
from app.models.chat import ChatSession, ChatMessage
from app.models.subscription import UserSubscription
from app.db.database import DatabaseManager


@pytest.mark.database
class TestDatabaseConnections:
    """Test database connection management."""
    
    async def test_database_connection(self, db_session):
        """Test basic database connection."""
        # Simple query to test connection
        result = await db_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        
        assert row is not None
        assert row.test == 1
    
    async def test_database_transaction_commit(self, db_session):
        """Test database transaction commit."""
        # Create a test user
        user = User(
            email="transaction_test@example.com",
            username="transaction_test",
            hashed_password="test_password",
            full_name="Transaction Test User"
        )
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        assert user.id is not None
        assert user.created_at is not None
    
    async def test_database_transaction_rollback(self, db_session):
        """Test database transaction rollback."""
        # Create user within transaction
        user = User(
            email="rollback_test@example.com",
            username="rollback_test", 
            hashed_password="test_password",
            full_name="Rollback Test User"
        )
        
        db_session.add(user)
        
        # Rollback before commit
        await db_session.rollback()
        
        # User should not exist after rollback
        result = await db_session.execute(
            select(User).where(User.email == "rollback_test@example.com")
        )
        assert result.scalar_one_or_none() is None
    
    async def test_concurrent_database_access(self, test_db_engine):
        """Test concurrent database access."""
        async def create_user(session, user_id):
            user = User(
                email=f"concurrent_{user_id}@example.com",
                username=f"concurrent_{user_id}",
                hashed_password="test_password",
                full_name=f"Concurrent User {user_id}"
            )
            session.add(user)
            await session.commit()
            return user.id
        
        # Create multiple concurrent sessions
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        async_session = async_sessionmaker(test_db_engine, class_=AsyncSession)
        
        tasks = []
        for i in range(10):
            session = async_session()
            task = create_user(session, i)
            tasks.append(task)
        
        # Execute concurrently
        user_ids = await asyncio.gather(*tasks)
        
        # All users should be created successfully
        assert len(user_ids) == 10
        assert all(uid is not None for uid in user_ids)


@pytest.mark.database
class TestUserModel:
    """Test User model operations."""
    
    async def test_user_creation(self, db_session):
        """Test basic user creation."""
        user = User(
            email="test_user@example.com",
            username="test_user",
            hashed_password="hashed_password",
            full_name="Test User",
            spiritual_level="beginner",
            dharmic_path="mindfulness"
        )
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "test_user@example.com"
        assert user.username == "test_user"
        assert user.is_active is True
        assert user.is_verified is False
        assert user.created_at is not None
        assert user.spiritual_level == "beginner"
    
    async def test_user_email_uniqueness(self, db_session):
        """Test email uniqueness constraint."""
        # Create first user
        user1 = User(
            email="unique@example.com",
            username="user1",
            hashed_password="password1",
            full_name="User One"
        )
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create second user with same email
        user2 = User(
            email="unique@example.com",  # Same email
            username="user2", 
            hashed_password="password2",
            full_name="User Two"
        )
        db_session.add(user2)
        
        # Should raise integrity error
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    async def test_user_username_uniqueness(self, db_session):
        """Test username uniqueness constraint."""
        # Create first user
        user1 = User(
            email="user1@example.com",
            username="unique_username",
            hashed_password="password1",
            full_name="User One"
        )
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create second user with same username
        user2 = User(
            email="user2@example.com",
            username="unique_username",  # Same username
            hashed_password="password2",
            full_name="User Two"
        )
        db_session.add(user2)
        
        # Should raise integrity error
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    async def test_user_update(self, db_session, test_user):
        """Test user update operations."""
        original_updated_at = test_user.updated_at
        
        # Update user data
        test_user.spiritual_level = "advanced"
        test_user.meditation_experience = 1000
        test_user.bio = "Updated bio"
        
        await db_session.commit()
        await db_session.refresh(test_user)
        
        assert test_user.spiritual_level == "advanced"
        assert test_user.meditation_experience == 1000
        assert test_user.bio == "Updated bio"
        # updated_at should change
        assert test_user.updated_at != original_updated_at
    
    async def test_user_soft_delete(self, db_session, test_user):
        """Test user soft delete (deactivation)."""
        user_id = test_user.id
        
        # Deactivate user instead of deleting
        test_user.is_active = False
        await db_session.commit()
        
        # User should still exist but be inactive
        result = await db_session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        assert user is not None
        assert user.is_active is False


@pytest.mark.database
class TestChatModel:
    """Test chat session and message models."""
    
    async def test_chat_session_creation(self, db_session, test_user):
        """Test chat session creation.""" 
        session = ChatSession(
            user_id=test_user.id,
            title="Test Meditation Session",
            spiritual_focus="mindfulness",
            is_active=True
        )
        
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.title == "Test Meditation Session"
        assert session.spiritual_focus == "mindfulness"
        assert session.is_active is True
        assert session.created_at is not None
    
    async def test_chat_message_creation(self, db_session, test_user):
        """Test chat message creation."""
        # Create session first
        session = ChatSession(
            user_id=test_user.id,
            title="Test Session",
            is_active=True
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Create message
        message = ChatMessage(
            session_id=session.id,
            user_id=test_user.id,
            role="user",
            content="How should I meditate?",
            metadata={"context": "beginner_question"}
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.id is not None
        assert message.session_id == session.id
        assert message.user_id == test_user.id
        assert message.role == "user"
        assert message.content == "How should I meditate?"
        assert message.metadata["context"] == "beginner_question"
    
    async def test_chat_session_messages_relationship(self, db_session, test_user):
        """Test relationship between chat session and messages."""
        # Create session
        session = ChatSession(
            user_id=test_user.id,
            title="Relationship Test Session",
            is_active=True
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Create multiple messages
        messages = []
        for i in range(3):
            message = ChatMessage(
                session_id=session.id,
                user_id=test_user.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                metadata={"sequence": i}
            )
            messages.append(message)
            db_session.add(message)
        
        await db_session.commit()
        
        # Query session with messages
        result = await db_session.execute(
            select(ChatSession)
            .where(ChatSession.id == session.id)
        )
        session_with_messages = result.scalar_one()
        
        # Should have relationships loaded
        message_count = await db_session.execute(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.session_id == session.id)
        )
        count = message_count.scalar()
        
        assert count == 3


@pytest.mark.database
class TestSubscriptionModel:
    """Test subscription model operations."""
    
    async def test_user_subscription_creation(self, db_session, test_user):
        """Test user subscription creation."""
        subscription = UserSubscription(
            user_id=test_user.id,
            plan_type="premium",
            status="active",
            billing_cycle="monthly",
            price=29.99,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30)
        )
        
        db_session.add(subscription)
        await db_session.commit()
        await db_session.refresh(subscription)
        
        assert subscription.id is not None
        assert subscription.user_id == test_user.id
        assert subscription.plan_type == "premium"
        assert subscription.status == "active"
        assert subscription.price == 29.99
    
    async def test_subscription_status_updates(self, db_session, test_user):
        """Test subscription status updates."""
        # Create active subscription
        subscription = UserSubscription(
            user_id=test_user.id,
            plan_type="premium",
            status="active",
            billing_cycle="monthly",
            price=29.99,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30)
        )
        
        db_session.add(subscription)
        await db_session.commit()
        await db_session.refresh(subscription)
        
        # Update to cancelled
        subscription.status = "cancelled"
        subscription.cancelled_at = datetime.utcnow()
        
        await db_session.commit()
        await db_session.refresh(subscription)
        
        assert subscription.status == "cancelled"
        assert subscription.cancelled_at is not None


@pytest.mark.database  
class TestDatabaseQueries:
    """Test complex database queries."""
    
    async def test_user_chat_history_query(self, db_session, test_user):
        """Test querying user chat history."""
        # Create chat sessions and messages
        sessions = []
        for i in range(3):
            session = ChatSession(
                user_id=test_user.id,
                title=f"Session {i}",
                spiritual_focus="mindfulness",
                is_active=True
            )
            db_session.add(session)
            sessions.append(session)
        
        await db_session.commit()
        
        # Add messages to sessions
        for i, session in enumerate(sessions):
            await db_session.refresh(session)
            for j in range(2):  # 2 messages per session
                message = ChatMessage(
                    session_id=session.id,
                    user_id=test_user.id,
                    role="user",
                    content=f"Message {j} in session {i}",
                    created_at=datetime.utcnow() + timedelta(minutes=i*10 + j)
                )
                db_session.add(message)
        
        await db_session.commit()
        
        # Query user's chat history
        result = await db_session.execute(
            select(ChatSession, func.count(ChatMessage.id).label('message_count'))
            .join(ChatMessage, ChatSession.id == ChatMessage.session_id)
            .where(ChatSession.user_id == test_user.id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.created_at.desc())
        )
        
        session_data = result.all()
        
        assert len(session_data) == 3
        for session, message_count in session_data:
            assert message_count == 2
            assert session.user_id == test_user.id
    
    async def test_spiritual_progress_query(self, db_session, test_user):
        """Test querying user's spiritual progress over time.""" 
        # Create chat sessions with different spiritual focuses
        spiritual_focuses = ["mindfulness", "compassion", "wisdom", "mindfulness"]
        sessions = []
        
        for i, focus in enumerate(spiritual_focuses):
            session = ChatSession(
                user_id=test_user.id,
                title=f"Session {i}",
                spiritual_focus=focus,
                is_active=True,
                created_at=datetime.utcnow() - timedelta(days=30-i*7)  # Spread over time
            )
            db_session.add(session)
            sessions.append(session)
        
        await db_session.commit()
        
        # Query spiritual focus distribution
        result = await db_session.execute(
            select(
                ChatSession.spiritual_focus,
                func.count(ChatSession.id).label('session_count')
            )
            .where(ChatSession.user_id == test_user.id)
            .group_by(ChatSession.spiritual_focus)
            .order_by(func.count(ChatSession.id).desc())
        )
        
        focus_distribution = result.all()
        
        # Should have mindfulness as most common (2 sessions)
        assert len(focus_distribution) == 3  # mindfulness, compassion, wisdom
        top_focus = focus_distribution[0]
        assert top_focus.spiritual_focus == "mindfulness"
        assert top_focus.session_count == 2
    
    async def test_user_statistics_query(self, db_session, test_user):
        """Test complex user statistics query."""
        # Create test data
        session = ChatSession(
            user_id=test_user.id,
            title="Stats Test Session",
            is_active=True
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Add messages with different roles
        message_data = [
            ("user", "Question 1"),
            ("assistant", "Answer 1"),
            ("user", "Question 2"), 
            ("assistant", "Answer 2"),
            ("user", "Question 3"),
            ("assistant", "Answer 3")
        ]
        
        for role, content in message_data:
            message = ChatMessage(
                session_id=session.id,
                user_id=test_user.id,
                role=role,
                content=content
            )
            db_session.add(message)
        
        await db_session.commit()
        
        # Query user statistics
        stats_query = await db_session.execute(
            select(
                func.count(ChatSession.id).label('total_sessions'),
                func.count(ChatMessage.id).filter(ChatMessage.role == 'user').label('user_messages'),
                func.count(ChatMessage.id).filter(ChatMessage.role == 'assistant').label('ai_responses'),
                func.max(ChatSession.created_at).label('last_activity')
            )
            .select_from(ChatSession)
            .join(ChatMessage, ChatSession.id == ChatMessage.session_id)
            .where(ChatSession.user_id == test_user.id)
        )
        
        stats = stats_query.first()
        
        assert stats.total_sessions == 1
        assert stats.user_messages == 3
        assert stats.ai_responses == 3
        assert stats.last_activity is not None


@pytest.mark.database
@pytest.mark.slow
class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    async def test_bulk_user_creation_performance(self, db_session, benchmark_timer):
        """Test performance of bulk user creation."""
        timer = benchmark_timer()
        
        # Create many users
        users = []
        for i in range(100):
            user = User(
                email=f"bulk_user_{i}@example.com",
                username=f"bulk_user_{i}",
                hashed_password="test_password",
                full_name=f"Bulk User {i}"
            )
            users.append(user)
        
        timer.start()
        
        # Bulk insert
        db_session.add_all(users)
        await db_session.commit()
        
        execution_time = timer.stop()
        
        # Should complete within reasonable time
        assert execution_time < 2.0  # 2 seconds for 100 users
        
        # Verify all users created
        count_result = await db_session.execute(
            select(func.count(User.id))
            .where(User.email.like("bulk_user_%"))
        )
        count = count_result.scalar()
        assert count == 100
    
    async def test_complex_query_performance(self, db_session, test_user, benchmark_timer):
        """Test performance of complex queries."""
        # Create substantial test data
        sessions = []
        for i in range(20):
            session = ChatSession(
                user_id=test_user.id,
                title=f"Performance Test Session {i}",
                spiritual_focus=f"focus_{i % 5}",
                is_active=True
            )
            db_session.add(session)
            sessions.append(session)
        
        await db_session.commit()
        
        # Add messages to each session
        for session in sessions:
            await db_session.refresh(session)
            for j in range(10):  # 10 messages per session
                message = ChatMessage(
                    session_id=session.id,
                    user_id=test_user.id,
                    role="user" if j % 2 == 0 else "assistant",
                    content=f"Performance test message {j}",
                    metadata={"test": True, "sequence": j}
                )
                db_session.add(message)
        
        await db_session.commit()
        
        # Execute complex query with timing
        timer = benchmark_timer()
        timer.start()
        
        complex_query = await db_session.execute(
            select(
                ChatSession.id,
                ChatSession.title,
                ChatSession.spiritual_focus,
                func.count(ChatMessage.id).label('message_count'),
                func.min(ChatMessage.created_at).label('first_message'),
                func.max(ChatMessage.created_at).label('last_message')
            )
            .join(ChatMessage, ChatSession.id == ChatMessage.session_id)
            .where(ChatSession.user_id == test_user.id)
            .group_by(ChatSession.id, ChatSession.title, ChatSession.spiritual_focus)
            .order_by(func.max(ChatMessage.created_at).desc())
        )
        
        results = complex_query.all()
        execution_time = timer.stop()
        
        # Should complete quickly even with complex joins
        assert execution_time < 1.0  # 1 second
        assert len(results) == 20  # All sessions returned
        
        # Verify result correctness
        for result in results:
            assert result.message_count == 10  # 10 messages per session
    
    async def test_database_connection_pooling(self, test_db_engine):
        """Test database connection pool performance."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        
        # Create session factory
        async_session = async_sessionmaker(test_db_engine, class_=AsyncSession)
        
        async def perform_query(session_id):
            """Perform a query in a separate session."""
            async with async_session() as session:
                result = await session.execute(
                    text("SELECT :session_id as id"), 
                    {"session_id": session_id}
                )
                return result.scalar()
        
        # Execute many concurrent queries
        tasks = [perform_query(i) for i in range(50)]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        
        # Should handle concurrent connections efficiently
        assert execution_time < 3.0  # 3 seconds for 50 concurrent queries
        assert len(results) == 50
        assert all(isinstance(r, int) for r in results)


@pytest.mark.database
class TestDatabaseMigrations:
    """Test database migrations and schema changes."""
    
    async def test_migration_state(self, test_db_engine):
        """Test current migration state."""
        # Check that all tables exist
        async with test_db_engine.connect() as conn:
            # Check for main tables
            tables_check = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """))
            
            table_names = [row[0] for row in tables_check.fetchall()]
            
            expected_tables = ['users', 'chat_sessions', 'chat_messages']
            for table in expected_tables:
                assert table in table_names, f"Missing table: {table}"
    
    async def test_database_constraints(self, db_session):
        """Test database constraints and indexes."""
        # Test that unique constraints work
        with pytest.raises(IntegrityError):
            # Try to create users with same email
            user1 = User(email="test@example.com", username="user1", hashed_password="pass")
            user2 = User(email="test@example.com", username="user2", hashed_password="pass")
            
            db_session.add_all([user1, user2])
            await db_session.commit()
    
    async def test_foreign_key_constraints(self, db_session, test_user):
        """Test foreign key constraints.""" 
        # Try to create chat message without valid session
        invalid_message = ChatMessage(
            session_id=99999,  # Non-existent session
            user_id=test_user.id,
            role="user",
            content="This should fail"
        )
        
        db_session.add(invalid_message)
        
        # Should raise integrity error due to foreign key constraint
        with pytest.raises(IntegrityError):
            await db_session.commit()
