# Docker Commands

Common Docker commands for the IndexTTS-2 TTS service. See [README.md](README.md) for setup and usage.

## Docker Common Commands

### Build
```bash
# Build image (use cache)
# Use --platform linux/amd64 to target AMD64 (required; ARM64 fails due to pynini)
docker build --platform linux/amd64 -t indextts-service .

# Build with docker-compose
docker-compose -f ../docker-compose.yml build

# Rebuild from scratch (no cache)
docker build --platform linux/amd64 --no-cache -t indextts-service .
```

### Run
```bash
# Run with docker run
docker run -d -p 8001:8001 \
  -v ~/tts-weights:/app/index-tts/checkpoints \
  -v $(pwd)/voices:/app/voices \
  --name tts-service indextts-service

# Run with docker-compose
docker-compose -f ../docker-compose.yml up -d
```

### Container Management
```bash
# View running containers
docker ps

# Stop container
docker stop tts-service

# Start container
docker start tts-service

# Restart container
docker restart tts-service

# View logs (follow)
docker logs -f tts-service

# Remove container (must stop first)
docker stop tts-service && docker rm tts-service
```

### Inspect
```bash
# Check weights mount
docker exec tts-service ls -lh /app/index-tts/checkpoints

# Check voices mount
docker exec tts-service ls -lh /app/voices

# Enter container shell
docker exec -it tts-service /bin/bash
```

---

## Clean All Docker Cache

Use these commands to free disk space and reset build state.

### Quick cleanup (safe)
Remove stopped containers, unused networks, dangling images:
```bash
docker system prune -f
```

### Full cleanup (removes all unused data)
```bash
# Remove all unused containers, networks, images, and optionally volumes
docker system prune -a --volumes -f
```
**Warning:** `--volumes` removes unused volumes. Use with caution if you have important data in Docker volumes.

### Build cache only
Clear Docker BuildKit/builder cache (speeds up `--no-cache` builds):
```bash
docker builder prune -a -f
```

### Per-resource cleanup
```bash
# Remove all stopped containers
docker container prune -f

# Remove all dangling images
docker image prune -f

# Remove ALL unused images (not just dangling)
docker image prune -a -f

# Remove unused networks
docker network prune -f

# Remove unused volumes
docker volume prune -f
```

### Nuclear option (complete reset)
```bash
# Stop all containers, remove everything (images, volumes, networks, build cache)
docker stop $(docker ps -aq) 2>/dev/null
docker system prune -a --volumes -f
docker builder prune -a -f
```

### Rebuild after cache clear
```bash
docker build --platform linux/amd64 --no-cache -t indextts-service .
```
