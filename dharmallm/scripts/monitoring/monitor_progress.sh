#!/bin/bash
# Real-time monitor for training and downloads

clear
echo "═══════════════════════════════════════════════════════════════════════"
echo "🕉️  DharmaLLM Training & Download Monitor"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

while true; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Training status
    echo "🎓 TRAINING STATUS:"
    TRAINING_PID=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $2}')
    if [ -n "$TRAINING_PID" ]; then
        TRAINING_CPU=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $3}')
        TRAINING_MEM=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $4}')
        TRAINING_TIME=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $10}')
        echo "   ✅ RUNNING (PID: $TRAINING_PID)"
        echo "   CPU: ${TRAINING_CPU}% | Memory: ${TRAINING_MEM}% | Runtime: ${TRAINING_TIME}"
        
        # Show latest training log
        if [ -f training_log.txt ]; then
            LOG_SIZE=$(wc -c < training_log.txt)
            if [ "$LOG_SIZE" -gt 100 ]; then
                echo ""
                echo "   Latest progress:"
                tail -5 training_log.txt | sed 's/^/   /'
            else
                echo "   (Initializing... log buffered)"
            fi
        fi
    else
        echo "   ⏸️  Not running or completed"
        if [ -f model/checkpoints/final_model.pt ]; then
            echo "   ✅ Training completed! Model saved."
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Download status
    echo "📥 WISDOM LIBRARY DOWNLOAD STATUS:"
    DOWNLOAD_PID=$(ps aux | grep download_wisdom_chapters | grep -v grep | awk '{print $2}')
    if [ -n "$DOWNLOAD_PID" ]; then
        DOWNLOAD_CPU=$(ps aux | grep download_wisdom_chapters | grep -v grep | awk '{print $3}')
        DOWNLOAD_MEM=$(ps aux | grep download_wisdom_chapters | grep -v grep | awk '{print $4}')
        DOWNLOAD_TIME=$(ps aux | grep download_wisdom_chapters | grep -v grep | awk '{print $10}')
        echo "   ✅ RUNNING (PID: $DOWNLOAD_PID)"
        echo "   CPU: ${DOWNLOAD_CPU}% | Memory: ${DOWNLOAD_MEM}% | Runtime: ${DOWNLOAD_TIME}"
        
        # Show latest download log
        if [ -f wisdom_chapters_log.txt ]; then
            LOG_SIZE=$(wc -c < wisdom_chapters_log.txt)
            if [ "$LOG_SIZE" -gt 100 ]; then
                echo ""
                echo "   Latest progress:"
                tail -5 wisdom_chapters_log.txt | sed 's/^/   /'
            else
                echo "   (Initializing...)"
            fi
        fi
    else
        echo "   ⏸️  Not running or completed"
        if [ -d data/authentic_sources/wisdom_library_chapters ]; then
            CHAPTER_FILES=$(ls -1 data/authentic_sources/wisdom_library_chapters/*.json 2>/dev/null | wc -l)
            if [ "$CHAPTER_FILES" -gt 0 ]; then
                echo "   ✅ Download completed! $CHAPTER_FILES files saved."
            fi
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # File status
    echo "📊 OUTPUT FILES:"
    
    if [ -f training_log.txt ]; then
        TRAIN_SIZE=$(du -h training_log.txt | awk '{print $1}')
        echo "   training_log.txt: $TRAIN_SIZE"
    fi
    
    if [ -f wisdom_chapters_log.txt ]; then
        WISDOM_SIZE=$(du -h wisdom_chapters_log.txt | awk '{print $1}')
        echo "   wisdom_chapters_log.txt: $WISDOM_SIZE"
    fi
    
    if [ -d model/checkpoints ]; then
        CHECKPOINT_COUNT=$(ls -1 model/checkpoints/*.pt 2>/dev/null | wc -l)
        if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
            echo "   Model checkpoints: $CHECKPOINT_COUNT saved"
            CHECKPOINT_SIZE=$(du -sh model/checkpoints 2>/dev/null | awk '{print $1}')
            echo "   Checkpoint size: $CHECKPOINT_SIZE"
        fi
    fi
    
    if [ -d data/authentic_sources/wisdom_library_chapters ]; then
        CHAPTER_COUNT=$(ls -1 data/authentic_sources/wisdom_library_chapters/*.json 2>/dev/null | wc -l)
        if [ "$CHAPTER_COUNT" -gt 0 ]; then
            echo "   Chapter files: $CHAPTER_COUNT saved"
            CHAPTERS_SIZE=$(du -sh data/authentic_sources/wisdom_library_chapters 2>/dev/null | awk '{print $1}')
            echo "   Chapters size: $CHAPTERS_SIZE"
        fi
    fi
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "Press Ctrl+C to stop monitoring (processes will continue running)"
    echo "═══════════════════════════════════════════════════════════════════════"
    
    sleep 10
    clear
done
