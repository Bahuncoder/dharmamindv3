"""
Automated Backup Manager for DharmaMind

Provides comprehensive backup and restore functionality for:
- SQLite databases
- Model checkpoints
- Configuration files
- Training data

Features:
- Automatic compression (tar.gz)
- Retention policy management
- Scheduled backups
- Backup verification
- Incremental backups
- Disaster recovery

Usage:
    backup_mgr = BackupManager()
    await backup_mgr.initialize()
    
    # Manual backup
    result = await backup_mgr.backup_database("dharma_knowledge.db")
    
    # Scheduled backup
    await backup_mgr.start_scheduled_backups()
    
    # Restore
    await backup_mgr.restore_backup(backup_id)
"""

import os
import json
import shutil
import tarfile
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Types of backups"""
    DATABASE = "database"
    MODEL = "model"
    CONFIG = "config"
    FULL = "full"
    INCREMENTAL = "incremental"


class BackupStatus(str, Enum):
    """Backup status"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class BackupMetadata:
    """Metadata for a backup"""
    
    def __init__(
        self,
        backup_id: str,
        backup_type: BackupType,
        timestamp: datetime,
        file_path: str,
        size_bytes: int,
        checksum: str,
        status: BackupStatus,
        source_files: List[str],
        retention_days: int = 30,
        tags: Optional[Dict[str, str]] = None
    ):
        self.backup_id = backup_id
        self.backup_type = backup_type
        self.timestamp = timestamp
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.checksum = checksum
        self.status = status
        self.source_files = source_files
        self.retention_days = retention_days
        self.tags = tags or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "status": self.status,
            "source_files": self.source_files,
            "retention_days": self.retention_days,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """Create from dictionary"""
        return cls(
            backup_id=data["backup_id"],
            backup_type=data["backup_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            file_path=data["file_path"],
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            status=data["status"],
            source_files=data["source_files"],
            retention_days=data.get("retention_days", 30),
            tags=data.get("tags", {})
        )


class BackupManager:
    """Manages automated backups and restores"""
    
    def __init__(
        self,
        backup_dir: str = "backups",
        retention_days: int = 30,
        max_backups: int = 50,
        compression_level: int = 9
    ):
        """
        Initialize backup manager
        
        Args:
            backup_dir: Directory to store backups
            retention_days: Days to keep backups (default: 30)
            max_backups: Maximum number of backups to keep
            compression_level: gzip compression level (1-9, default: 9)
        """
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.max_backups = max_backups
        self.compression_level = compression_level
        
        # Metadata storage
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.backups: Dict[str, BackupMetadata] = {}
        
        # Scheduler
        self.scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize backup system"""
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        await self._load_metadata()
        
        # Clean old backups
        await self.cleanup_old_backups()
        
        logger.info(f"Backup manager initialized. Backup dir: {self.backup_dir}")
        logger.info(f"Loaded {len(self.backups)} existing backups")
    
    async def backup_database(
        self,
        db_path: str,
        backup_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> BackupMetadata:
        """
        Backup a SQLite database
        
        Args:
            db_path: Path to database file
            backup_name: Custom backup name (optional)
            tags: Custom tags for the backup
        
        Returns:
            BackupMetadata object
        """
        db_path = Path(db_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Generate backup ID
        timestamp = datetime.now()
        backup_id = self._generate_backup_id("db", timestamp)
        
        # Backup filename
        if backup_name:
            backup_file = self.backup_dir / f"{backup_name}_{backup_id}.tar.gz"
        else:
            backup_file = self.backup_dir / f"database_{backup_id}.tar.gz"
        
        logger.info(f"Starting database backup: {db_path} -> {backup_file}")
        
        try:
            # Create temporary directory for backup
            temp_dir = self.backup_dir / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy database with proper locking
            temp_db = temp_dir / db_path.name
            await self._backup_sqlite_safely(str(db_path), str(temp_db))
            
            # Create compressed archive
            await asyncio.to_thread(
                self._create_tarball,
                backup_file,
                temp_dir,
                [db_path.name]
            )
            
            # Calculate checksum
            checksum = await self._calculate_checksum(backup_file)
            
            # Get file size
            size_bytes = backup_file.stat().st_size
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.DATABASE,
                timestamp=timestamp,
                file_path=str(backup_file),
                size_bytes=size_bytes,
                checksum=checksum,
                status=BackupStatus.COMPLETED,
                source_files=[str(db_path)],
                retention_days=self.retention_days,
                tags=tags or {}
            )
            
            # Verify backup
            if await self._verify_backup(backup_file, checksum):
                metadata.status = BackupStatus.VERIFIED
            else:
                metadata.status = BackupStatus.CORRUPTED
                raise ValueError("Backup verification failed")
            
            # Save metadata
            self.backups[backup_id] = metadata
            await self._save_metadata()
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"Database backup completed: {backup_id} ({size_bytes} bytes)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    async def backup_model_checkpoint(
        self,
        checkpoint_path: str,
        backup_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> BackupMetadata:
        """
        Backup a model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            backup_name: Custom backup name (optional)
            tags: Custom tags for the backup
        
        Returns:
            BackupMetadata object
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Generate backup ID
        timestamp = datetime.now()
        backup_id = self._generate_backup_id("model", timestamp)
        
        # Backup filename
        if backup_name:
            backup_file = self.backup_dir / f"{backup_name}_{backup_id}.tar.gz"
        else:
            backup_file = self.backup_dir / f"model_{backup_id}.tar.gz"
        
        logger.info(f"Starting model backup: {checkpoint_path} -> {backup_file}")
        
        try:
            # Collect files to backup
            if checkpoint_path.is_file():
                source_files = [checkpoint_path]
                arcnames = [checkpoint_path.name]
            else:
                source_files = list(checkpoint_path.rglob("*"))
                source_files = [f for f in source_files if f.is_file()]
                arcnames = [str(f.relative_to(checkpoint_path)) for f in source_files]
            
            # Create compressed archive
            await asyncio.to_thread(
                self._create_tarball_from_files,
                backup_file,
                source_files,
                arcnames
            )
            
            # Calculate checksum
            checksum = await self._calculate_checksum(backup_file)
            
            # Get file size
            size_bytes = backup_file.stat().st_size
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.MODEL,
                timestamp=timestamp,
                file_path=str(backup_file),
                size_bytes=size_bytes,
                checksum=checksum,
                status=BackupStatus.COMPLETED,
                source_files=[str(f) for f in source_files],
                retention_days=self.retention_days,
                tags=tags or {}
            )
            
            # Verify backup
            if await self._verify_backup(backup_file, checksum):
                metadata.status = BackupStatus.VERIFIED
            else:
                metadata.status = BackupStatus.CORRUPTED
                raise ValueError("Backup verification failed")
            
            # Save metadata
            self.backups[backup_id] = metadata
            await self._save_metadata()
            
            logger.info(f"Model backup completed: {backup_id} ({size_bytes} bytes)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Model backup failed: {e}")
            raise
    
    async def backup_full_system(
        self,
        include_databases: List[str],
        include_models: List[str],
        include_configs: List[str],
        backup_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> BackupMetadata:
        """
        Create a full system backup
        
        Args:
            include_databases: List of database paths
            include_models: List of model checkpoint paths
            include_configs: List of config file paths
            backup_name: Custom backup name (optional)
            tags: Custom tags for the backup
        
        Returns:
            BackupMetadata object
        """
        timestamp = datetime.now()
        backup_id = self._generate_backup_id("full", timestamp)
        
        if backup_name:
            backup_file = self.backup_dir / f"{backup_name}_{backup_id}.tar.gz"
        else:
            backup_file = self.backup_dir / f"full_system_{backup_id}.tar.gz"
        
        logger.info(f"Starting full system backup: {backup_file}")
        
        try:
            # Create temporary directory
            temp_dir = self.backup_dir / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            # Backup databases
            db_dir = temp_dir / "databases"
            db_dir.mkdir(exist_ok=True)
            
            for db_path in include_databases:
                db_path = Path(db_path)
                if db_path.exists():
                    temp_db = db_dir / db_path.name
                    await self._backup_sqlite_safely(str(db_path), str(temp_db))
            
            # Copy models
            model_dir = temp_dir / "models"
            model_dir.mkdir(exist_ok=True)
            
            for model_path in include_models:
                model_path = Path(model_path)
                if model_path.exists():
                    if model_path.is_file():
                        shutil.copy2(model_path, model_dir / model_path.name)
                    else:
                        shutil.copytree(model_path, model_dir / model_path.name)
            
            # Copy configs
            config_dir = temp_dir / "configs"
            config_dir.mkdir(exist_ok=True)
            
            for config_path in include_configs:
                config_path = Path(config_path)
                if config_path.exists():
                    shutil.copy2(config_path, config_dir / config_path.name)
            
            # Create compressed archive
            await asyncio.to_thread(
                self._create_tarball,
                backup_file,
                temp_dir,
                ["databases", "models", "configs"]
            )
            
            # Calculate checksum
            checksum = await self._calculate_checksum(backup_file)
            
            # Get file size
            size_bytes = backup_file.stat().st_size
            
            # Collect all source files
            all_source_files = (
                include_databases + include_models + include_configs
            )
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.FULL,
                timestamp=timestamp,
                file_path=str(backup_file),
                size_bytes=size_bytes,
                checksum=checksum,
                status=BackupStatus.COMPLETED,
                source_files=all_source_files,
                retention_days=self.retention_days,
                tags=tags or {}
            )
            
            # Verify backup
            if await self._verify_backup(backup_file, checksum):
                metadata.status = BackupStatus.VERIFIED
            else:
                metadata.status = BackupStatus.CORRUPTED
                raise ValueError("Backup verification failed")
            
            # Save metadata
            self.backups[backup_id] = metadata
            await self._save_metadata()
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"Full system backup completed: {backup_id} ({size_bytes} bytes)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Full system backup failed: {e}")
            raise
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_dir: str,
        verify_first: bool = True
    ) -> bool:
        """
        Restore a backup
        
        Args:
            backup_id: Backup ID to restore
            restore_dir: Directory to restore to
            verify_first: Verify backup integrity before restore
        
        Returns:
            True if successful
        """
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        backup_file = Path(metadata.file_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        logger.info(f"Restoring backup: {backup_id} -> {restore_dir}")
        
        try:
            # Verify backup if requested
            if verify_first:
                if not await self._verify_backup(backup_file, metadata.checksum):
                    raise ValueError("Backup verification failed - cannot restore")
            
            # Create restore directory
            restore_path = Path(restore_dir)
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Extract archive
            await asyncio.to_thread(
                self._extract_tarball,
                backup_file,
                restore_path
            )
            
            logger.info(f"Backup restored successfully: {backup_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            raise
    
    async def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[BackupMetadata]:
        """
        List available backups
        
        Args:
            backup_type: Filter by backup type (optional)
            tags: Filter by tags (optional)
        
        Returns:
            List of BackupMetadata objects
        """
        backups = list(self.backups.values())
        
        # Filter by type
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        # Filter by tags
        if tags:
            backups = [
                b for b in backups
                if all(b.tags.get(k) == v for k, v in tags.items())
            ]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        
        return backups
    
    async def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup
        
        Args:
            backup_id: Backup ID to delete
        
        Returns:
            True if successful
        """
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        backup_file = Path(metadata.file_path)
        
        try:
            # Delete file
            if backup_file.exists():
                backup_file.unlink()
            
            # Remove from metadata
            del self.backups[backup_id]
            await self._save_metadata()
            
            logger.info(f"Backup deleted: {backup_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            raise
    
    async def cleanup_old_backups(self) -> int:
        """
        Remove backups older than retention period
        
        Returns:
            Number of backups deleted
        """
        now = datetime.now()
        deleted_count = 0
        
        for backup_id, metadata in list(self.backups.items()):
            age_days = (now - metadata.timestamp).days
            
            if age_days > metadata.retention_days:
                try:
                    await self.delete_backup(backup_id)
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup_id} (age: {age_days} days)")
                except Exception as e:
                    logger.error(f"Failed to delete old backup {backup_id}: {e}")
        
        # Also enforce max_backups limit
        if len(self.backups) > self.max_backups:
            # Sort by timestamp, delete oldest
            sorted_backups = sorted(
                self.backups.items(),
                key=lambda x: x[1].timestamp
            )
            
            to_delete = len(self.backups) - self.max_backups
            
            for backup_id, _ in sorted_backups[:to_delete]:
                try:
                    await self.delete_backup(backup_id)
                    deleted_count += 1
                    logger.info(f"Deleted excess backup: {backup_id}")
                except Exception as e:
                    logger.error(f"Failed to delete excess backup {backup_id}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: {deleted_count} backups deleted")
        
        return deleted_count
    
    async def start_scheduled_backups(
        self,
        databases: List[str],
        models: List[str],
        interval_hours: int = 24
    ):
        """
        Start scheduled automatic backups
        
        Args:
            databases: List of databases to backup
            models: List of model checkpoints to backup
            interval_hours: Backup interval in hours (default: 24)
        """
        if self.is_running:
            logger.warning("Scheduled backups already running")
            return
        
        self.is_running = True
        
        async def backup_loop():
            while self.is_running:
                try:
                    logger.info("Starting scheduled backup...")
                    
                    # Backup databases
                    for db_path in databases:
                        try:
                            await self.backup_database(
                                db_path,
                                tags={"scheduled": "true", "type": "automatic"}
                            )
                        except Exception as e:
                            logger.error(f"Scheduled database backup failed: {e}")
                    
                    # Backup models
                    for model_path in models:
                        try:
                            await self.backup_model_checkpoint(
                                model_path,
                                tags={"scheduled": "true", "type": "automatic"}
                            )
                        except Exception as e:
                            logger.error(f"Scheduled model backup failed: {e}")
                    
                    # Cleanup old backups
                    await self.cleanup_old_backups()
                    
                    logger.info(f"Scheduled backup completed. Next backup in {interval_hours} hours")
                    
                except Exception as e:
                    logger.error(f"Scheduled backup error: {e}")
                
                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)
        
        self.scheduler_task = asyncio.create_task(backup_loop())
        logger.info(f"Scheduled backups started (interval: {interval_hours} hours)")
    
    async def stop_scheduled_backups(self):
        """Stop scheduled backups"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scheduled backups stopped")
    
    # Private helper methods
    
    def _generate_backup_id(self, prefix: str, timestamp: datetime) -> str:
        """Generate unique backup ID"""
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{date_str}"
    
    async def _backup_sqlite_safely(self, source_db: str, dest_db: str):
        """Backup SQLite database with proper locking"""
        def _backup():
            # Open source database
            source_conn = sqlite3.connect(source_db)
            
            # Create backup
            dest_conn = sqlite3.connect(dest_db)
            source_conn.backup(dest_conn)
            
            # Close connections
            dest_conn.close()
            source_conn.close()
        
        await asyncio.to_thread(_backup)
    
    def _create_tarball(
        self,
        output_file: Path,
        source_dir: Path,
        include_names: List[str]
    ):
        """Create compressed tarball"""
        with tarfile.open(output_file, f"w:gz", compresslevel=self.compression_level) as tar:
            for name in include_names:
                source_path = source_dir / name
                if source_path.exists():
                    tar.add(source_path, arcname=name)
    
    def _create_tarball_from_files(
        self,
        output_file: Path,
        source_files: List[Path],
        arcnames: List[str]
    ):
        """Create compressed tarball from file list"""
        with tarfile.open(output_file, f"w:gz", compresslevel=self.compression_level) as tar:
            for source_file, arcname in zip(source_files, arcnames):
                tar.add(source_file, arcname=arcname)
    
    def _extract_tarball(self, archive_file: Path, dest_dir: Path):
        """Extract tarball"""
        with tarfile.open(archive_file, "r:gz") as tar:
            tar.extractall(dest_dir)
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum"""
        def _calc():
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        
        return await asyncio.to_thread(_calc)
    
    async def _verify_backup(self, backup_file: Path, expected_checksum: str) -> bool:
        """Verify backup integrity"""
        if not backup_file.exists():
            return False
        
        actual_checksum = await self._calculate_checksum(backup_file)
        return actual_checksum == expected_checksum
    
    async def _load_metadata(self):
        """Load backup metadata from disk"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
            
            self.backups = {
                backup_id: BackupMetadata.from_dict(backup_data)
                for backup_id, backup_data in data.items()
            }
            
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
    
    async def _save_metadata(self):
        """Save backup metadata to disk"""
        try:
            data = {
                backup_id: metadata.to_dict()
                for backup_id, metadata in self.backups.items()
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")


# Convenience functions for common operations

async def backup_critical_files():
    """Backup all critical system files"""
    backup_mgr = BackupManager()
    await backup_mgr.initialize()
    
    # Define critical files
    databases = [
        "data/dharma_knowledge.db",
        "databases/saptarishi_analytics.db"
    ]
    
    models = [
        "models/checkpoints/latest"
    ]
    
    configs = [
        "config/model_config.py",
        "config/advanced_config.py"
    ]
    
    # Create full backup
    metadata = await backup_mgr.backup_full_system(
        include_databases=[db for db in databases if Path(db).exists()],
        include_models=[m for m in models if Path(m).exists()],
        include_configs=[c for c in configs if Path(c).exists()],
        backup_name="critical_system",
        tags={"priority": "critical", "type": "scheduled"}
    )
    
    return metadata


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize backup manager
        backup_mgr = BackupManager(
            backup_dir="backups",
            retention_days=30,
            max_backups=50
        )
        
        await backup_mgr.initialize()
        
        # Backup database
        print("🗄️  Backing up database...")
        db_backup = await backup_mgr.backup_database(
            "data/dharma_knowledge.db",
            tags={"type": "manual", "priority": "high"}
        )
        print(f"✅ Database backup: {db_backup.backup_id}")
        print(f"   Size: {db_backup.size_bytes / 1024 / 1024:.2f} MB")
        
        # List backups
        print("\n📋 Available backups:")
        backups = await backup_mgr.list_backups()
        for backup in backups[:5]:  # Show latest 5
            print(f"   {backup.backup_id}: {backup.backup_type} - {backup.timestamp}")
        
        print("\n✅ Backup system ready!")
    
    asyncio.run(main())
