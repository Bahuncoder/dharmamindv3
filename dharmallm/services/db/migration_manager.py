"""
Database Migration Manager

Provides utilities for managing database migrations with Alembic.

Usage:
    from services.db.migration_manager import MigrationManager
    
    # Initialize manager
    mgr = MigrationManager()
    
    # Get current version
    current = mgr.get_current_version()
    
    # Run migrations
    await mgr.upgrade()
    
    # Rollback
    await mgr.downgrade()
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations using Alembic"""
    
    def __init__(self, alembic_ini: str = "alembic.ini"):
        """
        Initialize migration manager
        
        Args:
            alembic_ini: Path to alembic.ini configuration file
        """
        self.alembic_ini = alembic_ini
        self.project_root = Path(__file__).parent.parent.parent
        
        if not (self.project_root / alembic_ini).exists():
            raise FileNotFoundError(
                f"Alembic configuration not found: {alembic_ini}"
            )
    
    def _run_alembic_command(
        self,
        command: List[str],
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run an Alembic command
        
        Args:
            command: Alembic command and arguments
            capture_output: Whether to capture output
        
        Returns:
            CompletedProcess object
        """
        cmd = ["alembic", "-c", self.alembic_ini] + command
        
        result = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            capture_output=capture_output,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Alembic command failed: {' '.join(cmd)}")
            logger.error(f"Error: {result.stderr}")
            raise RuntimeError(f"Migration command failed: {result.stderr}")
        
        return result
    
    def get_current_version(self) -> Optional[str]:
        """
        Get current database migration version
        
        Returns:
            Current revision ID or None if no migrations applied
        """
        try:
            result = self._run_alembic_command(["current"])
            output = result.stdout.strip()
            
            if not output or "no current revision" in output.lower():
                return None
            
            # Extract revision ID from output
            # Format: "revision_id (head)" or just "revision_id"
            if "(" in output:
                return output.split("(")[0].strip()
            return output.strip()
            
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None
    
    def get_migration_history(self) -> List[dict]:
        """
        Get migration history
        
        Returns:
            List of migration dictionaries
        """
        try:
            result = self._run_alembic_command(["history", "-v"])
            history = []
            
            for line in result.stdout.split("\n"):
                if "->" in line:
                    # Parse history line
                    parts = line.split("->")
                    if len(parts) == 2:
                        history.append({
                            "from": parts[0].strip(),
                            "to": parts[1].strip()
                        })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    def upgrade(self, revision: str = "head") -> bool:
        """
        Upgrade database to a specific revision
        
        Args:
            revision: Target revision (default: "head" for latest)
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Upgrading database to revision: {revision}")
            self._run_alembic_command(["upgrade", revision])
            logger.info("Database upgrade completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database upgrade failed: {e}")
            return False
    
    def downgrade(self, revision: str = "-1") -> bool:
        """
        Downgrade database to a specific revision
        
        Args:
            revision: Target revision (default: "-1" for one down)
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Downgrading database to revision: {revision}")
            self._run_alembic_command(["downgrade", revision])
            logger.info("Database downgrade completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database downgrade failed: {e}")
            return False
    
    def create_migration(
        self,
        message: str,
        autogenerate: bool = False
    ) -> bool:
        """
        Create a new migration
        
        Args:
            message: Migration description
            autogenerate: Whether to auto-generate from models
        
        Returns:
            True if successful
        """
        try:
            command = ["revision"]
            
            if autogenerate:
                command.append("--autogenerate")
            
            command.extend(["-m", message])
            
            logger.info(f"Creating migration: {message}")
            result = self._run_alembic_command(command)
            
            # Extract filename from output
            for line in result.stdout.split("\n"):
                if "Generating" in line:
                    logger.info(line)
            
            logger.info("Migration created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration creation failed: {e}")
            return False
    
    def stamp(self, revision: str) -> bool:
        """
        Stamp database with a specific revision without running migrations
        
        Args:
            revision: Revision to stamp
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Stamping database with revision: {revision}")
            self._run_alembic_command(["stamp", revision])
            logger.info("Database stamped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database stamping failed: {e}")
            return False
    
    def check_migrations_needed(self) -> bool:
        """
        Check if there are pending migrations
        
        Returns:
            True if migrations are needed
        """
        try:
            current = self.get_current_version()
            
            # Get head revision
            result = self._run_alembic_command(["heads"])
            head = result.stdout.strip().split()[0]
            
            return current != head
            
        except Exception as e:
            logger.error(f"Failed to check migrations: {e}")
            return False
    
    def validate_database(self) -> bool:
        """
        Validate that database matches current migration state
        
        Returns:
            True if database is valid
        """
        try:
            # Check if current revision matches head
            current = self.get_current_version()
            
            if current is None:
                logger.warning("No migrations applied to database")
                return False
            
            # Check if there are pending migrations
            if self.check_migrations_needed():
                logger.warning("Pending migrations detected")
                return False
            
            logger.info("Database is up to date")
            return True
            
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False


# CLI utility functions

def cli_upgrade():
    """CLI command: Upgrade database to latest"""
    mgr = MigrationManager()
    success = mgr.upgrade()
    
    if success:
        print("✅ Database upgraded successfully")
        print(f"Current version: {mgr.get_current_version()}")
    else:
        print("❌ Database upgrade failed")
        exit(1)


def cli_downgrade():
    """CLI command: Downgrade database by one revision"""
    mgr = MigrationManager()
    
    current = mgr.get_current_version()
    print(f"Current version: {current}")
    
    confirm = input("Downgrade database? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted")
        return
    
    success = mgr.downgrade()
    
    if success:
        print("✅ Database downgraded successfully")
        print(f"Current version: {mgr.get_current_version()}")
    else:
        print("❌ Database downgrade failed")
        exit(1)


def cli_current():
    """CLI command: Show current database version"""
    mgr = MigrationManager()
    
    current = mgr.get_current_version()
    
    if current:
        print(f"Current database version: {current}")
    else:
        print("No migrations applied")


def cli_history():
    """CLI command: Show migration history"""
    mgr = MigrationManager()
    
    history = mgr.get_migration_history()
    
    if history:
        print("Migration History:")
        for item in history:
            print(f"  {item['from']} -> {item['to']}")
    else:
        print("No migration history available")


def cli_validate():
    """CLI command: Validate database state"""
    mgr = MigrationManager()
    
    print("Validating database...")
    
    is_valid = mgr.validate_database()
    
    if is_valid:
        print("✅ Database is valid and up to date")
    else:
        print("⚠️  Database validation failed or migrations pending")
        
        if mgr.check_migrations_needed():
            print("\nPending migrations detected. Run:")
            print("  python -m services.db.migration_manager upgrade")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m services.db.migration_manager <command>")
        print("\nCommands:")
        print("  upgrade    - Upgrade to latest migration")
        print("  downgrade  - Downgrade by one migration")
        print("  current    - Show current version")
        print("  history    - Show migration history")
        print("  validate   - Validate database state")
        sys.exit(1)
    
    command = sys.argv[1]
    
    commands = {
        "upgrade": cli_upgrade,
        "downgrade": cli_downgrade,
        "current": cli_current,
        "history": cli_history,
        "validate": cli_validate
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
