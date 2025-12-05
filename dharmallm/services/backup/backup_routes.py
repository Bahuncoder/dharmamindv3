"""
FastAPI Integration for Backup Manager

Provides REST API endpoints for backup management:
- POST /backups/database - Backup a database
- POST /backups/model - Backup a model checkpoint
- POST /backups/full - Create full system backup
- GET /backups - List all backups
- GET /backups/{backup_id} - Get backup details
- POST /backups/{backup_id}/restore - Restore a backup
- DELETE /backups/{backup_id} - Delete a backup
- POST /backups/cleanup - Clean old backups
- POST /backups/schedule/start - Start scheduled backups
- POST /backups/schedule/stop - Stop scheduled backups

Usage:
    from services.backup.backup_routes import router as backup_router
    
    app.include_router(backup_router, prefix="/api/v1")
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from services.backup.backup_manager import (
    BackupManager,
    BackupMetadata,
    BackupType,
    BackupStatus
)

# Initialize router
router = APIRouter(prefix="/backups", tags=["backups"])

# Global backup manager instance
backup_manager: Optional[BackupManager] = None


# Request/Response models

class BackupDatabaseRequest(BaseModel):
    """Request to backup a database"""
    db_path: str = Field(..., description="Path to database file")
    backup_name: Optional[str] = Field(None, description="Custom backup name")
    tags: Optional[Dict[str, str]] = Field(None, description="Custom tags")


class BackupModelRequest(BaseModel):
    """Request to backup a model"""
    model_path: str = Field(..., description="Path to model checkpoint")
    backup_name: Optional[str] = Field(None, description="Custom backup name")
    tags: Optional[Dict[str, str]] = Field(None, description="Custom tags")


class BackupFullSystemRequest(BaseModel):
    """Request to create full system backup"""
    databases: List[str] = Field(..., description="Database paths")
    models: List[str] = Field(..., description="Model checkpoint paths")
    configs: List[str] = Field(..., description="Config file paths")
    backup_name: Optional[str] = Field(None, description="Custom backup name")
    tags: Optional[Dict[str, str]] = Field(None, description="Custom tags")


class RestoreBackupRequest(BaseModel):
    """Request to restore a backup"""
    restore_dir: str = Field(..., description="Directory to restore to")
    verify_first: bool = Field(True, description="Verify before restore")


class BackupResponse(BaseModel):
    """Backup metadata response"""
    backup_id: str
    backup_type: str
    timestamp: datetime
    file_path: str
    size_bytes: int
    size_mb: float
    checksum: str
    status: str
    source_files: List[str]
    retention_days: int
    tags: Dict[str, str]
    
    @classmethod
    def from_metadata(cls, metadata: BackupMetadata) -> "BackupResponse":
        """Create from BackupMetadata"""
        return cls(
            backup_id=metadata.backup_id,
            backup_type=metadata.backup_type,
            timestamp=metadata.timestamp,
            file_path=metadata.file_path,
            size_bytes=metadata.size_bytes,
            size_mb=round(metadata.size_bytes / 1024 / 1024, 2),
            checksum=metadata.checksum,
            status=metadata.status,
            source_files=metadata.source_files,
            retention_days=metadata.retention_days,
            tags=metadata.tags
        )


class BackupListResponse(BaseModel):
    """List of backups response"""
    total: int
    backups: List[BackupResponse]


class BackupStatsResponse(BaseModel):
    """Backup statistics response"""
    total_backups: int
    total_size_bytes: int
    total_size_gb: float
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    oldest_backup: Optional[datetime]
    newest_backup: Optional[datetime]


class ScheduleBackupsRequest(BaseModel):
    """Request to schedule automatic backups"""
    databases: List[str] = Field(..., description="Databases to backup")
    models: List[str] = Field(..., description="Models to backup")
    interval_hours: int = Field(24, description="Backup interval in hours")


# Initialize backup manager

async def get_backup_manager() -> BackupManager:
    """Get or initialize backup manager"""
    global backup_manager
    
    if backup_manager is None:
        backup_manager = BackupManager(
            backup_dir="backups",
            retention_days=30,
            max_backups=50
        )
        await backup_manager.initialize()
    
    return backup_manager


# API Endpoints

@router.post("/database", response_model=BackupResponse)
async def backup_database(
    request: BackupDatabaseRequest,
    background_tasks: BackgroundTasks
):
    """
    Backup a database
    
    Creates a compressed backup of the specified database with
    verification and metadata tracking.
    """
    mgr = await get_backup_manager()
    
    # Validate database path
    if not Path(request.db_path).exists():
        raise HTTPException(status_code=404, detail="Database not found")
    
    try:
        metadata = await mgr.backup_database(
            db_path=request.db_path,
            backup_name=request.backup_name,
            tags=request.tags
        )
        
        return BackupResponse.from_metadata(metadata)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model", response_model=BackupResponse)
async def backup_model(request: BackupModelRequest):
    """
    Backup a model checkpoint
    
    Creates a compressed backup of the specified model checkpoint
    or directory with verification.
    """
    mgr = await get_backup_manager()
    
    # Validate model path
    if not Path(request.model_path).exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        metadata = await mgr.backup_model_checkpoint(
            checkpoint_path=request.model_path,
            backup_name=request.backup_name,
            tags=request.tags
        )
        
        return BackupResponse.from_metadata(metadata)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full", response_model=BackupResponse)
async def backup_full_system(request: BackupFullSystemRequest):
    """
    Create a full system backup
    
    Backs up databases, models, and configuration files in a single
    compressed archive for disaster recovery.
    """
    mgr = await get_backup_manager()
    
    try:
        # Filter existing paths
        databases = [d for d in request.databases if Path(d).exists()]
        models = [m for m in request.models if Path(m).exists()]
        configs = [c for c in request.configs if Path(c).exists()]
        
        if not (databases or models or configs):
            raise HTTPException(
                status_code=400,
                detail="No valid files found to backup"
            )
        
        metadata = await mgr.backup_full_system(
            include_databases=databases,
            include_models=models,
            include_configs=configs,
            backup_name=request.backup_name,
            tags=request.tags
        )
        
        return BackupResponse.from_metadata(metadata)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=BackupListResponse)
async def list_backups(
    backup_type: Optional[BackupType] = None,
    limit: int = 50
):
    """
    List all backups
    
    Returns a list of all backups with optional filtering by type.
    Ordered by timestamp (newest first).
    """
    mgr = await get_backup_manager()
    
    try:
        backups = await mgr.list_backups(backup_type=backup_type)
        
        # Apply limit
        backups = backups[:limit]
        
        return BackupListResponse(
            total=len(backups),
            backups=[BackupResponse.from_metadata(b) for b in backups]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=BackupStatsResponse)
async def get_backup_stats():
    """
    Get backup statistics
    
    Returns comprehensive statistics about all backups including
    total size, count by type, and status distribution.
    """
    mgr = await get_backup_manager()
    
    try:
        backups = await mgr.list_backups()
        
        if not backups:
            return BackupStatsResponse(
                total_backups=0,
                total_size_bytes=0,
                total_size_gb=0.0,
                by_type={},
                by_status={},
                oldest_backup=None,
                newest_backup=None
            )
        
        # Calculate statistics
        total_size = sum(b.size_bytes for b in backups)
        
        by_type = {}
        for backup in backups:
            by_type[backup.backup_type] = by_type.get(backup.backup_type, 0) + 1
        
        by_status = {}
        for backup in backups:
            by_status[backup.status] = by_status.get(backup.status, 0) + 1
        
        timestamps = [b.timestamp for b in backups]
        
        return BackupStatsResponse(
            total_backups=len(backups),
            total_size_bytes=total_size,
            total_size_gb=round(total_size / 1024 / 1024 / 1024, 2),
            by_type=by_type,
            by_status=by_status,
            oldest_backup=min(timestamps),
            newest_backup=max(timestamps)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backup_id}", response_model=BackupResponse)
async def get_backup(backup_id: str):
    """
    Get backup details
    
    Returns detailed metadata for a specific backup.
    """
    mgr = await get_backup_manager()
    
    if backup_id not in mgr.backups:
        raise HTTPException(status_code=404, detail="Backup not found")
    
    metadata = mgr.backups[backup_id]
    return BackupResponse.from_metadata(metadata)


@router.post("/{backup_id}/restore")
async def restore_backup(backup_id: str, request: RestoreBackupRequest):
    """
    Restore a backup
    
    Extracts and restores the specified backup to the given directory.
    Optionally verifies backup integrity before restoration.
    """
    mgr = await get_backup_manager()
    
    if backup_id not in mgr.backups:
        raise HTTPException(status_code=404, detail="Backup not found")
    
    try:
        success = await mgr.restore_backup(
            backup_id=backup_id,
            restore_dir=request.restore_dir,
            verify_first=request.verify_first
        )
        
        return {
            "success": success,
            "backup_id": backup_id,
            "restore_dir": request.restore_dir,
            "message": "Backup restored successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{backup_id}")
async def delete_backup(backup_id: str):
    """
    Delete a backup
    
    Permanently deletes the specified backup and its metadata.
    This action cannot be undone.
    """
    mgr = await get_backup_manager()
    
    if backup_id not in mgr.backups:
        raise HTTPException(status_code=404, detail="Backup not found")
    
    try:
        success = await mgr.delete_backup(backup_id)
        
        return {
            "success": success,
            "backup_id": backup_id,
            "message": "Backup deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_backups():
    """
    Clean up old backups
    
    Removes backups older than the retention period and enforces
    the maximum backup limit.
    """
    mgr = await get_backup_manager()
    
    try:
        deleted_count = await mgr.cleanup_old_backups()
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Cleaned up {deleted_count} old backups"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule/start")
async def start_scheduled_backups(request: ScheduleBackupsRequest):
    """
    Start scheduled automatic backups
    
    Begins automatic backups at the specified interval. Backups
    run in the background and include automatic cleanup.
    """
    mgr = await get_backup_manager()
    
    try:
        await mgr.start_scheduled_backups(
            databases=request.databases,
            models=request.models,
            interval_hours=request.interval_hours
        )
        
        return {
            "success": True,
            "message": f"Scheduled backups started (every {request.interval_hours}h)",
            "databases": request.databases,
            "models": request.models
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule/stop")
async def stop_scheduled_backups():
    """
    Stop scheduled automatic backups
    
    Stops the automatic backup scheduler. Any backup currently
    in progress will complete.
    """
    mgr = await get_backup_manager()
    
    try:
        await mgr.stop_scheduled_backups()
        
        return {
            "success": True,
            "message": "Scheduled backups stopped"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint for backups

@router.get("/health")
async def backup_health():
    """
    Check backup system health
    
    Returns status of the backup system including available backups,
    disk space, and scheduler status.
    """
    mgr = await get_backup_manager()
    
    try:
        backups = await mgr.list_backups()
        
        # Check disk space
        backup_dir = Path(mgr.backup_dir)
        if backup_dir.exists():
            import shutil
            disk_usage = shutil.disk_usage(backup_dir)
            disk_free_gb = disk_usage.free / 1024 / 1024 / 1024
        else:
            disk_free_gb = 0
        
        return {
            "status": "healthy",
            "backup_count": len(backups),
            "backup_dir": str(mgr.backup_dir),
            "disk_free_gb": round(disk_free_gb, 2),
            "scheduler_running": mgr.is_running,
            "retention_days": mgr.retention_days,
            "max_backups": mgr.max_backups
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
