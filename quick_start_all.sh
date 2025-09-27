#!/bin/bash
# Start all services in parallel
gnome-terminal --tab --title="Backend" -- bash -c "./quick_start_backend.sh; exec bash"
gnome-terminal --tab --title="Frontend" -- bash -c "./quick_start_frontend.sh; exec bash"
echo "🚀 Starting all services..."
echo "🐍 Backend: http://localhost:8000"
echo "📱 Frontend: http://localhost:3000"
