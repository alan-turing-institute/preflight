# Generate the portal's self-signed pair once. Args: certificate, key, subject name, subjectAltName.
set -eu
umask 077
if [ -s "$1" ] && [ -s "$2" ]; then exit 0; fi
install -d -m 0755 "$(dirname "$1")" "$(dirname "$2")"
openssl req -x509 -newkey rsa:4096 -nodes -days 365 -keyout "$2" -out "$1" -subj "/CN=$3" -addext "subjectAltName=$4" \
    -addext "basicConstraints=critical,CA:FALSE" -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth"
chmod 0644 "$1"
chmod 0600 "$2"
echo "WARNING: the portal is using a SELF-SIGNED certificate ($1). Browsers will warn; set portal.tls_cert and portal.tls_key in the site file for a real one." >&2
