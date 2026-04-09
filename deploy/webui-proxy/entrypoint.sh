#!/bin/sh
# Resolve SSL certs — use mkcert certs from shared volume, fallback to self-signed
CERT_DIR=/etc/nginx/ssl
SSL_DIR=/ssl

mkdir -p "$CERT_DIR"

# Try HOST_IP-named cert first, then generic cert.pem
if [ -f "$SSL_DIR/${HOST_IP}.pem" ]; then
    cp "$SSL_DIR/${HOST_IP}.pem" "$CERT_DIR/cert.pem"
    cp "$SSL_DIR/${HOST_IP}-key.pem" "$CERT_DIR/key.pem"
elif [ -f "$SSL_DIR/cert.pem" ]; then
    cp "$SSL_DIR/cert.pem" "$CERT_DIR/cert.pem"
    cp "$SSL_DIR/key.pem" "$CERT_DIR/key.pem"
else
    echo "No mkcert certs found, generating self-signed..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
        -subj "/CN=devai-webui" 2>/dev/null
fi

exec nginx -g "daemon off;"
