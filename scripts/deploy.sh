#!/bin/bash
# Деплоит агента на Hetzner сервер
# Запускать локально из корня проекта: bash scripts/deploy.sh

set -e

# Загружаем переменные из .env
export $(grep -v '^#' .env | xargs)

SERVER="root@135.181.40.162"
REMOTE_DIR="/opt/agentx"
PORT=9019
CONTAINER="agentx-research"
IMAGE="${DOCKER_IMAGE:-alinabelko/agentx-research:latest}"

echo "==> Копируем код на сервер..."
ssh "$SERVER" "mkdir -p $REMOTE_DIR"
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    -czf /tmp/agentx_deploy.tar.gz .
scp /tmp/agentx_deploy.tar.gz "$SERVER:/tmp/"
ssh "$SERVER" "tar -xzf /tmp/agentx_deploy.tar.gz -C $REMOTE_DIR"

echo "==> Собираем и пушим образ..."
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

echo "==> Перезапускаем контейнер на сервере..."
ssh "$SERVER" "
  docker pull $IMAGE
  docker stop $CONTAINER 2>/dev/null || true
  docker rm   $CONTAINER 2>/dev/null || true
  docker run -d \
    --name $CONTAINER \
    --restart unless-stopped \
    -p $PORT:$PORT \
    --env-file $REMOTE_DIR/.env \
    $IMAGE \
    --host 0.0.0.0 \
    --port $PORT \
    --card-url http://135.181.40.162:$PORT/
"

echo ""
echo "==> Проверяем..."
sleep 3
curl -s "http://135.181.40.162:$PORT/.well-known/agent.json" | python3 -m json.tool

echo ""
echo "ГОТОВО: http://135.181.40.162:$PORT/"
