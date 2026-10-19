# Generate the portal's self-signed pair once per server name (a renamed portal gets a new pair).
# Args: certificate, key, subject name, subjectAltName.
set -eu
umask 077
# -checkip, not both: given a host name it prints "does match" for any certificate (openssl 3.0.13, measured).
case "$4" in IP:*) check=-checkip ;; *) check=-checkhost ;; esac
if [ -s "$1" ] && [ -s "$2" ] && openssl x509 -in "$1" -noout "$check" "$3" | grep -q ' does match'; then
    exit 0
fi
install -d -m 0755 "$(dirname "$1")" "$(dirname "$2")"
openssl req -x509 -newkey rsa:4096 -nodes -days 365 -keyout "$2" -out "$1" -subj "/CN=$3" -addext "subjectAltName=$4" \
    -addext "basicConstraints=critical,CA:FALSE" -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth"
chmod 0644 "$1"
chmod 0600 "$2"
echo "WARNING: the portal is using a SELF-SIGNED certificate ($1). Browsers will warn; set portal.tls_cert and portal.tls_key in the site file for a real one." >&2
