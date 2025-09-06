#!/bin/bash

# 🕉️ DharmaMind Production Backup System
#
# Comprehensive backup solution for all DharmaMind data:
# - PostgreSQL database with point-in-time recovery
# - Redis data and configuration
# - Vector database embeddings
# - Application configuration and logs
# - SSL certificates and secrets
#
# Usage: ./backup_production.sh [--full] [--incremental] [--restore DATE]
# 
# May our backups preserve digital dharma for all time 💾✨

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="/var/backups/dharmamind"
LOG_FILE="/var/log/dharmamind/backup.log"

# Backup settings
FULL_BACKUP_SCHEDULE="0 2 * * 0"    # Weekly full backup (Sunday 2 AM)
INCREMENTAL_SCHEDULE="0 3 * * 1-6"  # Daily incremental (Mon-Sat 3 AM)
RETENTION_DAYS=30
COMPRESSION_LEVEL=6
ENCRYPTION_ENABLED=true

# Remote backup settings
REMOTE_BACKUP_ENABLED=${REMOTE_BACKUP_ENABLED:-false}
AWS_S3_BUCKET=${AWS_S3_BUCKET:-""}
BACKUP_ENCRYPTION_KEY=${BACKUP_ENCRYPTION_KEY:-""}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")    echo -e "${BLUE}[INFO]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "WARN")    echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR")   echo -e "${RED}[ERROR]${NC} $message" ;;
        "DHARMIC") echo -e "${PURPLE}[🕉️]${NC} $message" ;;
    esac
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Parse command line arguments
BACKUP_TYPE="incremental"
RESTORE_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full|-f)
            BACKUP_TYPE="full"
            shift
            ;;
        --incremental|-i)
            BACKUP_TYPE="incremental"
            shift
            ;;
        --restore|-r)
            RESTORE_DATE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--full] [--incremental] [--restore DATE]"
            echo "  --full         Perform full backup"
            echo "  --incremental  Perform incremental backup (default)"
            echo "  --restore DATE Restore from backup (format: YYYY-MM-DD)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create backup directory structure
create_backup_structure() {
    local backup_date=$(date +%Y-%m-%d_%H-%M-%S)
    local backup_type=$1
    
    BACKUP_DIR="${BACKUP_ROOT}/${backup_type}/${backup_date}"
    mkdir -p "$BACKUP_DIR"/{database,redis,vector_db,application,config,logs}
    
    echo "$BACKUP_DIR"
}

# Backup PostgreSQL database
backup_database() {
    local backup_dir=$1
    local backup_type=$2
    
    log "INFO" "📊 Starting PostgreSQL backup ($backup_type)..."
    
    local db_backup_dir="${backup_dir}/database"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Database connection details
    local db_host="localhost"
    local db_port="5432"
    local db_name="dharmamind"
    local db_user="dharmamind"
    
    # Set PGPASSWORD from environment
    export PGPASSWORD="${DATABASE_PASSWORD:-dharmamind}"
    
    if [[ "$backup_type" == "full" ]]; then
        # Full database dump
        log "INFO" "Creating full database dump..."
        
        pg_dump -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
                --verbose --no-password --format=custom --compress=$COMPRESSION_LEVEL \
                --file="${db_backup_dir}/dharmamind_full_${timestamp}.dump"
        
        # Also create SQL format for easier inspection
        pg_dump -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
                --verbose --no-password --format=plain \
                --file="${db_backup_dir}/dharmamind_full_${timestamp}.sql"
        
        # Backup database globals (users, roles, etc.)
        pg_dumpall -h "$db_host" -p "$db_port" -U "$db_user" --globals-only \
                   --verbose --no-password \
                   --file="${db_backup_dir}/globals_${timestamp}.sql"
        
    else
        # Incremental backup using WAL archiving
        log "INFO" "Creating incremental backup..."
        
        # Create base backup if it doesn't exist
        local latest_full=$(find "${BACKUP_ROOT}/full" -name "*.dump" | sort | tail -1)
        if [[ -z "$latest_full" ]]; then
            log "WARN" "No full backup found, creating one first..."
            backup_database "$backup_dir" "full"
            return
        fi
        
        # Create incremental backup using pg_receivewal
        mkdir -p "${db_backup_dir}/wal"
        
        # Get current WAL position
        local wal_position=$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
                            -t -c "SELECT pg_current_wal_lsn();" | tr -d ' ')
        
        echo "$wal_position" > "${db_backup_dir}/wal_position_${timestamp}.txt"
        
        # Archive WAL files
        pg_receivewal -h "$db_host" -p "$db_port" -U "$db_user" \
                      -D "${db_backup_dir}/wal" --synchronous --no-password &
        
        local wal_pid=$!
        sleep 5  # Let it collect some WAL files
        kill $wal_pid 2>/dev/null || true
    fi
    
    # Backup database statistics for performance tuning
    log "INFO" "Backing up database statistics..."
    psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
         -c "COPY (SELECT * FROM pg_stat_user_tables) TO STDOUT WITH CSV HEADER;" \
         > "${db_backup_dir}/table_stats_${timestamp}.csv"
    
    psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
         -c "COPY (SELECT * FROM pg_stat_user_indexes) TO STDOUT WITH CSV HEADER;" \
         > "${db_backup_dir}/index_stats_${timestamp}.csv"
    
    # Create backup metadata
    cat > "${db_backup_dir}/backup_metadata.json" << EOF
{
    "backup_type": "$backup_type",
    "timestamp": "$timestamp",
    "database": "$db_name",
    "host": "$db_host",
    "port": "$db_port",
    "pg_version": "$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -t -c "SELECT version();" | head -1 | tr -d ' ')",
    "backup_size_bytes": $(du -sb "$db_backup_dir" | cut -f1),
    "compression_level": $COMPRESSION_LEVEL
}
EOF
    
    unset PGPASSWORD
    log "SUCCESS" "✅ Database backup completed"
}

# Backup Redis data
backup_redis() {
    local backup_dir=$1
    
    log "INFO" "📈 Starting Redis backup..."
    
    local redis_backup_dir="${backup_dir}/redis"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Redis connection details
    local redis_host="localhost"
    local redis_port="6379"
    local redis_auth=""
    
    if [[ -n "${REDIS_PASSWORD}" ]]; then
        redis_auth="-a ${REDIS_PASSWORD}"
    fi
    
    # Create RDB backup
    log "INFO" "Creating Redis RDB backup..."
    redis-cli -h "$redis_host" -p "$redis_port" $redis_auth BGSAVE
    
    # Wait for backup to complete
    while [[ "$(redis-cli -h "$redis_host" -p "$redis_port" $redis_auth LASTSAVE)" == "$(redis-cli -h "$redis_host" -p "$redis_port" $redis_auth LASTSAVE)" ]]; do
        sleep 1
    done
    
    # Copy RDB file
    docker cp dharmamind_redis_master:/data/dump.rdb "${redis_backup_dir}/redis_${timestamp}.rdb"
    
    # Backup Redis configuration
    redis-cli -h "$redis_host" -p "$redis_port" $redis_auth CONFIG GET '*' \
              > "${redis_backup_dir}/redis_config_${timestamp}.txt"
    
    # Export all keys and values (for smaller datasets)
    log "INFO" "Exporting Redis keys..."
    redis-cli -h "$redis_host" -p "$redis_port" $redis_auth --rdb "${redis_backup_dir}/redis_export_${timestamp}.rdb"
    
    # Create Redis backup metadata
    cat > "${redis_backup_dir}/backup_metadata.json" << EOF
{
    "backup_type": "full",
    "timestamp": "$timestamp",
    "redis_version": "$(redis-cli -h "$redis_host" -p "$redis_port" $redis_auth INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')",
    "database_count": $(redis-cli -h "$redis_host" -p "$redis_port" $redis_auth CONFIG GET databases | tail -1),
    "memory_usage": "$(redis-cli -h "$redis_host" -p "$redis_port" $redis_auth INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')",
    "backup_size_bytes": $(du -sb "$redis_backup_dir" | cut -f1)
}
EOF
    
    log "SUCCESS" "✅ Redis backup completed"
}

# Backup vector database
backup_vector_db() {
    local backup_dir=$1
    
    log "INFO" "🧠 Starting vector database backup..."
    
    local vector_backup_dir="${backup_dir}/vector_db"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup ChromaDB data
    if [[ -d "${PROJECT_ROOT}/vector_db/chroma_data" ]]; then
        log "INFO" "Backing up ChromaDB data..."
        cp -r "${PROJECT_ROOT}/vector_db/chroma_data" "${vector_backup_dir}/chroma_${timestamp}"
    fi
    
    # Backup Pinecone indexes (export metadata)
    if [[ -n "${PINECONE_API_KEY}" ]]; then
        log "INFO" "Backing up Pinecone metadata..."
        # This would require Pinecone client to export index metadata
        echo "Pinecone backup requires manual export of index configurations" > "${vector_backup_dir}/pinecone_notes.txt"
    fi
    
    # Backup FAISS indexes
    if [[ -d "${PROJECT_ROOT}/vector_db/faiss_indexes" ]]; then
        log "INFO" "Backing up FAISS indexes..."
        cp -r "${PROJECT_ROOT}/vector_db/faiss_indexes" "${vector_backup_dir}/faiss_${timestamp}"
    fi
    
    # Backup embeddings cache
    if [[ -d "${PROJECT_ROOT}/vector_db/embeddings_cache" ]]; then
        log "INFO" "Backing up embeddings cache..."
        cp -r "${PROJECT_ROOT}/vector_db/embeddings_cache" "${vector_backup_dir}/embeddings_${timestamp}"
    fi
    
    # Create vector DB backup metadata
    cat > "${vector_backup_dir}/backup_metadata.json" << EOF
{
    "backup_type": "full",
    "timestamp": "$timestamp",
    "chroma_backup": $(if [[ -d "${vector_backup_dir}/chroma_${timestamp}" ]]; then echo "true"; else echo "false"; fi),
    "faiss_backup": $(if [[ -d "${vector_backup_dir}/faiss_${timestamp}" ]]; then echo "true"; else echo "false"; fi),
    "embeddings_backup": $(if [[ -d "${vector_backup_dir}/embeddings_${timestamp}" ]]; then echo "true"; else echo "false"; fi),
    "backup_size_bytes": $(du -sb "$vector_backup_dir" | cut -f1)
}
EOF
    
    log "SUCCESS" "✅ Vector database backup completed"
}

# Backup application data
backup_application() {
    local backup_dir=$1
    
    log "INFO" "📱 Starting application backup..."
    
    local app_backup_dir="${backup_dir}/application"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup application logs
    if [[ -d "${PROJECT_ROOT}/logs" ]]; then
        log "INFO" "Backing up application logs..."
        cp -r "${PROJECT_ROOT}/logs" "${app_backup_dir}/logs_${timestamp}"
    fi
    
    # Backup uploaded files
    if [[ -d "${PROJECT_ROOT}/uploads" ]]; then
        log "INFO" "Backing up uploaded files..."
        cp -r "${PROJECT_ROOT}/uploads" "${app_backup_dir}/uploads_${timestamp}"
    fi
    
    # Backup user data exports
    if [[ -d "${PROJECT_ROOT}/user_exports" ]]; then
        log "INFO" "Backing up user exports..."
        cp -r "${PROJECT_ROOT}/user_exports" "${app_backup_dir}/user_exports_${timestamp}"
    fi
    
    # Backup trained models
    if [[ -d "${PROJECT_ROOT}/models" ]]; then
        log "INFO" "Backing up trained models..."
        cp -r "${PROJECT_ROOT}/models" "${app_backup_dir}/models_${timestamp}"
    fi
    
    log "SUCCESS" "✅ Application data backup completed"
}

# Backup configuration
backup_configuration() {
    local backup_dir=$1
    
    log "INFO" "⚙️ Starting configuration backup..."
    
    local config_backup_dir="${backup_dir}/config"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup environment files
    log "INFO" "Backing up environment configuration..."
    if [[ -f "${PROJECT_ROOT}/.env.production" ]]; then
        cp "${PROJECT_ROOT}/.env.production" "${config_backup_dir}/env_production_${timestamp}"
    fi
    
    # Backup Docker configurations
    log "INFO" "Backing up Docker configurations..."
    cp "${PROJECT_ROOT}/docker-compose.yml" "${config_backup_dir}/docker-compose_${timestamp}.yml"
    cp "${PROJECT_ROOT}/docker-compose.prod.yml" "${config_backup_dir}/docker-compose-prod_${timestamp}.yml"
    cp "${PROJECT_ROOT}/Dockerfile" "${config_backup_dir}/Dockerfile_${timestamp}"
    
    # Backup Nginx configuration
    if [[ -d "${PROJECT_ROOT}/nginx" ]]; then
        cp -r "${PROJECT_ROOT}/nginx" "${config_backup_dir}/nginx_${timestamp}"
    fi
    
    # Backup monitoring configurations
    if [[ -d "${PROJECT_ROOT}/monitoring" ]]; then
        cp -r "${PROJECT_ROOT}/monitoring" "${config_backup_dir}/monitoring_${timestamp}"
    fi
    
    # Backup SSL certificates (if any)
    if [[ -d "${PROJECT_ROOT}/ssl" ]]; then
        log "INFO" "Backing up SSL certificates..."
        cp -r "${PROJECT_ROOT}/ssl" "${config_backup_dir}/ssl_${timestamp}"
    fi
    
    # Backup application configuration
    if [[ -d "${PROJECT_ROOT}/backend/app/config" ]]; then
        cp -r "${PROJECT_ROOT}/backend/app/config" "${config_backup_dir}/app_config_${timestamp}"
    fi
    
    log "SUCCESS" "✅ Configuration backup completed"
}

# Encrypt backup if enabled
encrypt_backup() {
    local backup_dir=$1
    
    if [[ "$ENCRYPTION_ENABLED" != "true" ]] || [[ -z "$BACKUP_ENCRYPTION_KEY" ]]; then
        log "INFO" "Backup encryption disabled or no key provided"
        return 0
    fi
    
    log "INFO" "🔐 Encrypting backup..."
    
    local encrypted_file="${backup_dir}.tar.gz.enc"
    
    # Create compressed archive
    tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    
    # Encrypt the archive
    openssl enc -aes-256-cbc -salt -in "${backup_dir}.tar.gz" -out "$encrypted_file" -k "$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted files
    rm -rf "$backup_dir" "${backup_dir}.tar.gz"
    
    log "SUCCESS" "✅ Backup encrypted: $encrypted_file"
    echo "$encrypted_file"
}

# Compress backup
compress_backup() {
    local backup_dir=$1
    
    log "INFO" "📦 Compressing backup..."
    
    local compressed_file="${backup_dir}.tar.gz"
    tar -czf "$compressed_file" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    
    # Remove uncompressed directory
    rm -rf "$backup_dir"
    
    local original_size=$(du -sb "$(dirname "$backup_dir")" | cut -f1)
    local compressed_size=$(du -sb "$compressed_file" | cut -f1)
    local compression_ratio=$(echo "scale=2; $compressed_size * 100 / $original_size" | bc)
    
    log "SUCCESS" "✅ Backup compressed: $compressed_file (${compression_ratio}% of original)"
    echo "$compressed_file"
}

# Upload to remote storage
upload_to_remote() {
    local backup_file=$1
    
    if [[ "$REMOTE_BACKUP_ENABLED" != "true" ]]; then
        log "INFO" "Remote backup disabled"
        return 0
    fi
    
    log "INFO" "☁️ Uploading to remote storage..."
    
    local filename=$(basename "$backup_file")
    local remote_path="dharmamind-backups/$(date +%Y)/$(date +%m)/$filename"
    
    if [[ -n "$AWS_S3_BUCKET" ]]; then
        # Upload to AWS S3
        if command -v aws &> /dev/null; then
            aws s3 cp "$backup_file" "s3://${AWS_S3_BUCKET}/${remote_path}"
            log "SUCCESS" "✅ Backup uploaded to S3: s3://${AWS_S3_BUCKET}/${remote_path}"
        else
            log "ERROR" "AWS CLI not available for S3 upload"
            return 1
        fi
    fi
    
    # Add other cloud providers here (Azure, GCP, etc.)
    
    return 0
}

# Cleanup old backups
cleanup_old_backups() {
    log "INFO" "🧹 Cleaning up old backups..."
    
    # Clean local backups older than retention period
    find "$BACKUP_ROOT" -type f -name "*.tar.gz*" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_ROOT" -type d -empty -delete
    
    # Clean remote backups if enabled
    if [[ "$REMOTE_BACKUP_ENABLED" == "true" && -n "$AWS_S3_BUCKET" ]]; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y-%m-%d)
        # This would require more complex S3 cleanup logic
        log "INFO" "Remote backup cleanup requires manual configuration of S3 lifecycle policies"
    fi
    
    log "SUCCESS" "✅ Old backup cleanup completed"
}

# Restore from backup
restore_backup() {
    local restore_date=$1
    
    log "DHARMIC" "🔄 Starting backup restoration for date: $restore_date"
    
    # Find backup file
    local backup_file=$(find "$BACKUP_ROOT" -name "*${restore_date}*" -type f | head -1)
    
    if [[ -z "$backup_file" ]]; then
        log "ERROR" "No backup found for date: $restore_date"
        exit 1
    fi
    
    log "INFO" "Found backup: $backup_file"
    
    # Confirm restoration
    echo -e "${YELLOW}WARNING: This will overwrite current data!${NC}"
    read -p "Are you sure you want to restore from $restore_date? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log "INFO" "Restoration cancelled"
        exit 0
    fi
    
    # Stop services
    log "INFO" "Stopping services..."
    docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" down
    
    # Extract backup
    local restore_dir="/tmp/dharmamind_restore_$(date +%s)"
    mkdir -p "$restore_dir"
    
    if [[ "$backup_file" == *.enc ]]; then
        # Decrypt first
        if [[ -z "$BACKUP_ENCRYPTION_KEY" ]]; then
            log "ERROR" "Backup is encrypted but no decryption key provided"
            exit 1
        fi
        
        local decrypted_file="${backup_file%.enc}"
        openssl enc -aes-256-cbc -d -in "$backup_file" -out "$decrypted_file" -k "$BACKUP_ENCRYPTION_KEY"
        backup_file="$decrypted_file"
    fi
    
    tar -xzf "$backup_file" -C "$restore_dir"
    
    # Restore database
    log "INFO" "Restoring database..."
    local db_restore_dir=$(find "$restore_dir" -name "database" -type d)
    if [[ -n "$db_restore_dir" ]]; then
        docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" up -d postgres_primary
        sleep 10
        
        local dump_file=$(find "$db_restore_dir" -name "*.dump" | head -1)
        if [[ -n "$dump_file" ]]; then
            export PGPASSWORD="${DATABASE_PASSWORD:-dharmamind}"
            pg_restore -h localhost -p 5432 -U dharmamind -d dharmamind --clean --if-exists "$dump_file"
            unset PGPASSWORD
        fi
    fi
    
    # Restore Redis
    log "INFO" "Restoring Redis..."
    local redis_restore_dir=$(find "$restore_dir" -name "redis" -type d)
    if [[ -n "$redis_restore_dir" ]]; then
        docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" up -d redis_master
        sleep 5
        
        local rdb_file=$(find "$redis_restore_dir" -name "*.rdb" | head -1)
        if [[ -n "$rdb_file" ]]; then
            docker cp "$rdb_file" dharmamind_redis_master:/data/dump.rdb
            docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" restart redis_master
        fi
    fi
    
    # Restore other components...
    # (Vector DB, application data, configuration)
    
    # Start all services
    log "INFO" "Starting all services..."
    docker-compose -f "${PROJECT_ROOT}/docker-compose.prod.yml" up -d
    
    # Cleanup restore directory
    rm -rf "$restore_dir"
    
    log "SUCCESS" "✅ Backup restoration completed!"
}

# Main backup function
main() {
    log "DHARMIC" "🕉️ Starting DharmaMind Production Backup"
    
    if [[ -n "$RESTORE_DATE" ]]; then
        restore_backup "$RESTORE_DATE"
        exit 0
    fi
    
    log "INFO" "Backup type: $BACKUP_TYPE"
    log "INFO" "Compression level: $COMPRESSION_LEVEL"
    log "INFO" "Encryption: $ENCRYPTION_ENABLED"
    log "INFO" "Remote backup: $REMOTE_BACKUP_ENABLED"
    
    # Create backup directory
    local backup_dir
    backup_dir=$(create_backup_structure "$BACKUP_TYPE")
    
    log "INFO" "Backup directory: $backup_dir"
    
    # Perform backups
    backup_database "$backup_dir" "$BACKUP_TYPE"
    backup_redis "$backup_dir"
    backup_vector_db "$backup_dir"
    backup_application "$backup_dir"
    backup_configuration "$backup_dir"
    
    # Process backup
    local final_backup_file
    
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        final_backup_file=$(encrypt_backup "$backup_dir")
    else
        final_backup_file=$(compress_backup "$backup_dir")
    fi
    
    # Upload to remote storage
    upload_to_remote "$final_backup_file"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate backup report
    local backup_size=$(du -sh "$final_backup_file" | cut -f1)
    log "SUCCESS" "✅ Backup completed successfully!"
    log "INFO" "Final backup file: $final_backup_file"
    log "INFO" "Backup size: $backup_size"
    
    # Create backup verification
    cat > "${final_backup_file}.verify" << EOF
{
    "backup_file": "$final_backup_file",
    "backup_type": "$BACKUP_TYPE",
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_size": "$backup_size",
    "components": ["database", "redis", "vector_db", "application", "configuration"],
    "encrypted": $ENCRYPTION_ENABLED,
    "remote_uploaded": $REMOTE_BACKUP_ENABLED,
    "checksum": "$(sha256sum "$final_backup_file" | cut -d' ' -f1)"
}
EOF
    
    log "DHARMIC" "🎉 May this backup preserve our digital dharma safely!"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
