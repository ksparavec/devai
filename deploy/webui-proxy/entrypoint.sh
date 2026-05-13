#!/bin/sh
# Resolve SSL certs — use mkcert certs from shared volume, fallback to self-signed
CERT_DIR=/etc/nginx/ssl
SSL_DIR=/ssl

mkdir -p "$CERT_DIR"

# Defense-in-depth: HOST_IP is fed in unquoted into a filesystem path,
# so reject any value that isn't a plausible IP or hostname before use.
# Allowed: letters, digits, dot, hyphen, underscore. Anything else
# (path traversal, command sub, glob) is replaced with "localhost" so
# we deterministically fall through to the generic cert.pem branch.
case "$HOST_IP" in
    *[!A-Za-z0-9._-]*|"")
        echo "warning: HOST_IP=${HOST_IP:-<empty>} contains invalid chars; using localhost"
        HOST_IP="localhost"
        ;;
esac

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
