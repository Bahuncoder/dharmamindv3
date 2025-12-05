"""
Backup Module for DharmaMind

Provides comprehensive backup and restore functionality for databases,
model checkpoints, and configuration files.

Components:
- BackupManager: Core backup management system
- backup_routes: FastAPI endpoints for backup operations

Usage:
    # In FastAPI app
    from services.backup.backup_routes import router as backup_router
    app.include_router(backup_router, prefix="/api/v1")
    
    # Direct usage
    from services.backup import BackupManager
    
    backup_mgr = BackupManager()
    await backup_mgr.initialize()
    await backup_mgr.backup_database("data/my_db.db")
"""

from services.backup.backup_manager import (
    BackupManager,
    BackupMetadata,
    BackupType,
    BackupStatus,
    backup_critical_files
)

__all__ = [
    "BackupManager",
    "BackupMetadata",
    "BackupType",
    "BackupStatus",
    "backup_critical_files"
]
