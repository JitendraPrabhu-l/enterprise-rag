# Static UI + reverse proxy (ADR-021). Compose mounts these files directly
# into a stock nginx for the fast dev loop; this image exists for registry-
# based deploys (Helm/CI), where mounts aren't an option. Build context: ui/.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html app.js styles.css /usr/share/nginx/html/
