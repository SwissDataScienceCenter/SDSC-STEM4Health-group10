#!/bin/bash
# Stop all bioreactor services
echo "Stopping Bioreactor Application..."
pkill -f "bioreactor-app" 2>/dev/null && echo "  Java API stopped." || echo "  API not running."
pkill -f streamlit 2>/dev/null && echo "  Streamlit stopped." || echo "  Streamlit not running."
echo "  PostgreSQL keeps running (managed separately)."
echo "Done."
