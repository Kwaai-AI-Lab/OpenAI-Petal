#!/bin/bash

set -e  # Exit on error

VERSION="v1.0.0"
REPO="kwaailab"

# Function to check if Docker is running
docker_check() {
    if ! docker info &>/dev/null; then
        echo "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if an image has changed
docker_image_changed() {
    local image=$1
    local latest_digest=$(docker images --no-trunc --quiet $REPO/$image:latest)
    local version_digest=$(docker images --no-trunc --quiet $REPO/$image:$VERSION)
    
    if [ "$latest_digest" != "$version_digest" ]; then
        return 0  # Image has changed
    else
        return 1  # No change
    fi
}

# Build images
docker_check
echo "Building production images..."
make build-production-server
make build-production-api
make build-production-bootstrap

# Tag images
echo "Tagging images..."
docker tag kwaainet-api $REPO/kwaainet-api:$VERSION
docker tag kwaainet-api $REPO/kwaainet-api:latest

docker tag kwaainet-node $REPO/kwaainet-node:$VERSION
docker tag kwaainet-node $REPO/kwaainet-node:latest

docker tag kwaainet-bootstrap $REPO/kwaainet-bootstrap:$VERSION
docker tag kwaainet-bootstrap $REPO/kwaainet-bootstrap:latest

echo "Images tagged successfully."

# Ask user for confirmation before pushing
echo "Checking for changes in images..."
images_to_push=()
for image in kwaainet-api kwaainet-node kwaainet-bootstrap; do
    if docker_image_changed "$image"; then
        images_to_push+=("$image")
    fi
done

if [ ${#images_to_push[@]} -eq 0 ]; then
    echo "No images have changed. Skipping push."
    exit 0
fi

echo "The following images have changed and will be pushed: ${images_to_push[*]}"
echo "Do you want to proceed? (y/N)"
read -r CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    for image in "${images_to_push[@]}"; do
        docker push $REPO/$image:$VERSION
        docker push $REPO/$image:latest
    done
    echo "Images pushed successfully."
else
    echo "Skipping push."
fi
