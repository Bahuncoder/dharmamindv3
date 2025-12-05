#!/bin/bash
# DharmaLLM Master Control Script

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

clear
echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}🕉️  DharmaLLM Master Control Panel${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}\n"

# Function to check if a process is running
check_process() {
    ps aux | grep "$1" | grep -v grep > /dev/null
    return $?
}

# Display current status
echo -e "${CYAN}📊 CURRENT STATUS:${NC}\n"

# Training status
if check_process "train_master_corpus"; then
    echo -e "${GREEN}✅ Training: RUNNING${NC}"
    TRAIN_PID=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $2}')
    echo -e "   PID: $TRAIN_PID"
else
    echo -e "${YELLOW}⏸️  Training: Not running${NC}"
fi

# API status
if check_process "uvicorn.*main:app"; then
    echo -e "${GREEN}✅ API Server: RUNNING${NC}"
    API_PID=$(ps aux | grep "uvicorn.*main:app" | grep -v grep | awk '{print $2}')
    echo -e "   PID: $API_PID"
else
    echo -e "${YELLOW}⏸️  API Server: Not running${NC}"
fi

# Download status
if check_process "download_gretil"; then
    echo -e "${GREEN}✅ GRETIL Download: RUNNING${NC}"
else
    echo -e "${YELLOW}⏸️  GRETIL Download: Not running${NC}"
fi

echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}📋 MENU:${NC}\n"

echo "  [1] Monitor Training (real-time dashboard)"
echo "  [2] Monitor LLM System (API health dashboard)"
echo "  [3] Start Training"
echo "  [4] Start API Server"
echo "  [5] Start GRETIL Download"
echo "  [6] View Training Log"
echo "  [7] View Checkpoints"
echo "  [8] View Corpus Stats"
echo "  [9] View System Resources"
echo "  [0] View All Status (detailed)"
echo ""
echo "  [s] Stop Training"
echo "  [k] Stop API Server"
echo "  [q] Quit"
echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -ne "${CYAN}Select option: ${NC}"
read option

case $option in
    1)
        echo -e "\n${GREEN}Starting Training Monitor...${NC}\n"
        python3 scripts/monitoring/monitor_training.py
        ;;
    2)
        echo -e "\n${GREEN}Starting LLM System Monitor...${NC}\n"
        python3 scripts/monitoring/monitor_llm_system.py
        ;;
    3)
        if check_process "train_master_corpus"; then
            echo -e "\n${YELLOW}Training is already running!${NC}"
        else
            echo -e "\n${GREEN}Starting Training...${NC}"
            nohup ./venv/bin/python scripts/training/train_master_corpus.py > training_log.txt 2>&1 &
            echo -e "${GREEN}Training started in background!${NC}"
            echo -e "Monitor with: python3 scripts/monitoring/monitor_training.py"
        fi
        ;;
    4)
        if check_process "uvicorn.*main:app"; then
            echo -e "\n${YELLOW}API Server is already running!${NC}"
        else
            echo -e "\n${GREEN}Starting API Server...${NC}"
            cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000
        fi
        ;;
    5)
        if check_process "download_gretil"; then
            echo -e "\n${YELLOW}GRETIL download is already running!${NC}"
        else
            echo -e "\n${GREEN}Starting GRETIL Download...${NC}"
            nohup python3 scripts/data_collection/download_gretil.py > gretil_log.txt 2>&1 &
            echo -e "${GREEN}Download started in background!${NC}"
            echo -e "Check progress: tail -f logs/downloads/gretil_log.txt"
        fi
        ;;
    6)
        echo -e "\n${CYAN}Training Log (last 50 lines):${NC}\n"
        tail -50 logs/training/training_log.txt
        ;;
    7)
        echo -e "\n${CYAN}Model Checkpoints:${NC}\n"
        ls -lh model/checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
        ;;
    8)
        echo -e "\n${CYAN}Corpus Statistics:${NC}\n"
        cat data/master_corpus/CORPUS_REPORT.txt
        ;;
    9)
        echo -e "\n${CYAN}System Resources:${NC}\n"
        echo "CPU:"
        top -bn1 | grep "Cpu(s)" | awk '{print "  Usage: " $2 + $4 "%"}'
        echo ""
        echo "Memory:"
        free -h | grep Mem | awk '{print "  Total: " $2 "\n  Used: " $3 "\n  Free: " $4}'
        echo ""
        echo "Disk:"
        df -h / | tail -1 | awk '{print "  Total: " $2 "\n  Used: " $3 "\n  Free: " $4 "\n  Usage: " $5}'
        ;;
    0)
        echo -e "\n${CYAN}Detailed Status:${NC}\n"
        
        echo -e "${YELLOW}Training Processes:${NC}"
        ps aux | grep -E "train_master_corpus" | grep -v grep || echo "  None"
        
        echo -e "\n${YELLOW}API Processes:${NC}"
        ps aux | grep -E "uvicorn|fastapi" | grep -v grep || echo "  None"
        
        echo -e "\n${YELLOW}Download Processes:${NC}"
        ps aux | grep -E "download_" | grep -v grep || echo "  None"
        
        echo -e "\n${YELLOW}Checkpoints:${NC}"
        ls -lh model/checkpoints/*.pt 2>/dev/null | wc -l | awk '{print "  Count: " $1}'
        
        echo -e "\n${YELLOW}Log Files:${NC}"
        ls -lh *log*.txt 2>/dev/null
        ;;
    s)
        if check_process "train_master_corpus"; then
            TRAIN_PID=$(ps aux | grep train_master_corpus | grep -v grep | awk '{print $2}')
            echo -e "\n${RED}Stopping training (PID: $TRAIN_PID)...${NC}"
            kill $TRAIN_PID
            echo -e "${GREEN}Training stopped${NC}"
        else
            echo -e "\n${YELLOW}Training is not running${NC}"
        fi
        ;;
    k)
        if check_process "uvicorn.*main:app"; then
            API_PID=$(ps aux | grep "uvicorn.*main:app" | grep -v grep | awk '{print $2}')
            echo -e "\n${RED}Stopping API server (PID: $API_PID)...${NC}"
            kill $API_PID
            echo -e "${GREEN}API server stopped${NC}"
        else
            echo -e "\n${YELLOW}API server is not running${NC}"
        fi
        ;;
    q)
        echo -e "\n${GREEN}Goodbye! 🕉️${NC}\n"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid option${NC}"
        ;;
esac

echo ""
echo -e "${CYAN}Press Enter to continue...${NC}"
read
exec "$0"
