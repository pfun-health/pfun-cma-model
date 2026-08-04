#!/usr/bin/env bash

set -e


# Check if containers are running
docker ps --filter "name=ollama"
docker ps --filter "name=pfun-cma-model"

# See ollama logs
docker logs --tail=100 ollama

# Test connectivity directly to ollama (will show exact error)
curl -v http://localhost:11435/api/generate -X POST \
     -H "Content-Type: application/json" \
     -d '{"model": "test", "stream": false}'

# Test WITH the API key header
curl -v http://localhost:11435/api/generate -X POST \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer 28cc8223548548469d8635800358ba37.ejkoXbPk7IEzdk4N4Co_XZhs" \
     -d '{"model": "test", "stream": false}'

# Check pfun-cma-model logs
docker logs --tail=50 pfun-cma-model
