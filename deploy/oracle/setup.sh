#!/usr/bin/env bash
# Bootstrap the Production RAG stack on a fresh Oracle Always Free ARM VM
# (VM.Standard.A1.Flex, Ubuntu 22.04/24.04). Idempotent — safe to re-run.
#
#   1. SSH to the VM, clone/copy this repo, cd into it
#   2. GROQ_API_KEY=gsk_... bash deploy/oracle/setup.sh
#
# Full runbook (OCI console steps, networking, updates): docs/DEPLOY-ORACLE.md
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
REPO_ROOT="$(pwd)"

echo "==> Production RAG — Oracle VM bootstrap (repo: ${REPO_ROOT})"

# ---------------------------------------------------------------- docker ----
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker Engine + Compose plugin"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
else
  echo "==> Docker already installed: $(docker --version)"
fi

# Compose !override tags in docker-compose.public.yml need v2.24.4+.
COMPOSE_VERSION="$(sudo docker compose version --short 2>/dev/null || echo 0)"
echo "==> Docker Compose version: ${COMPOSE_VERSION}"

# -------------------------------------------------------------- firewall ----
# Oracle's Ubuntu images ship iptables REJECT rules baked into the instance
# (separate from the OCI Security List!). Opening the Security List alone is
# NOT enough — the classic OCI gotcha. Allow 80/443 here and persist.
echo "==> Opening ports 80/443 in the instance firewall (iptables)"
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 5 -p tcp --dport 443 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 3000 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 5 -p tcp --dport 3000 -j ACCEPT
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

# ------------------------------------------------------------------ .env ----
# Generate strong secrets on first run; never ship defaults on a public VM.
if [ ! -f .env ]; then
  echo "==> Generating .env with fresh secrets"
  : "${GROQ_API_KEY:?Set GROQ_API_KEY when invoking: GROQ_API_KEY=gsk_... bash deploy/oracle/setup.sh}"
  rand() { tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24; }
  cat > .env <<EOF
GROQ_API_KEY=${GROQ_API_KEY}
NEO4J_PASSWORD=$(rand)
MINIO_ROOT_USER=rag-admin
MINIO_ROOT_PASSWORD=$(rand)
GRAFANA_ADMIN_PASSWORD=$(rand)
ENVIRONMENT=production
EOF
  chmod 600 .env
  echo "    (secrets written to .env — Grafana login: admin / see GRAFANA_ADMIN_PASSWORD)"
else
  echo "==> .env exists — keeping it"
  grep -q GRAFANA_ADMIN_PASSWORD .env || echo "GRAFANA_ADMIN_PASSWORD=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24)" >> .env
fi

# ----------------------------------------------------------------- build ----
# NOTE (ARM): the base images used here are all multi-arch. If the ingestion/
# retrieval build fails resolving torch from download.pytorch.org/whl/cpu on
# aarch64, change that Dockerfile line's --index-url to --extra-index-url —
# PyPI's own aarch64 torch wheels are CPU-only anyway. See DEPLOY-ORACLE.md.
echo "==> Building and starting the stack (public port lockdown applied)"
sudo docker compose \
  -f docker-compose.yml \
  -f deploy/oracle/docker-compose.public.yml \
  up -d --build

echo "==> Waiting for the UI to answer on :80"
for _ in $(seq 1 60); do
  curl -sf -o /dev/null http://localhost:80 && break
  sleep 10
done

PUBLIC_IP="$(curl -sf -m 5 https://api.ipify.org || hostname -I | awk '{print $1}')"
echo ""
echo "=================================================================="
echo "  UI:      http://${PUBLIC_IP}/"
echo "  Grafana: http://${PUBLIC_IP}:3000/  (admin / GRAFANA_ADMIN_PASSWORD in .env)"
echo ""
echo "  If unreachable from outside: add an OCI Security List ingress"
echo "  rule for TCP 80 (and 3000) — instance firewall is already open."
echo "=================================================================="
