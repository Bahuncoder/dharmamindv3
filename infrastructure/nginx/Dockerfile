# Nginx Dockerfile for DharmaMind Production
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create log directories
RUN mkdir -p /var/log/nginx

# Create SSL directory for future certificates
RUN mkdir -p /etc/nginx/ssl

# Copy custom error pages (optional)
# COPY error_pages/ /usr/share/nginx/html/errors/

# Set proper permissions
RUN chown -R nginx:nginx /var/log/nginx /etc/nginx/ssl

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
