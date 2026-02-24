#!/bin/bash

set -e  # Exit on error

VERSION="v1.0.0"
REPO="ghcr.io/kwaai-ai-lab"

# Function to check if Docker is running
docker_check() {
    if ! docker info &>/dev/null; then
        echo "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

docker_image_changed() {
    local image=$1
    local remote_digest=$(docker manifest inspect $REPO/$image:$VERSION 2>/dev/null | jq -r '.manifests[0].digest' 2>/dev/null)
    local local_digest=$(docker inspect --format='{{index .RepoDigests 0}}' $REPO/$image:latest 2>/dev/null | awk -F '@' '{print $2}' 2>/dev/null)

    if [ -z "$remote_digest" ] || [ -z "$local_digest" ] || [ "$remote_digest" != "$local_digest" ]; then
        return 0  # Image has changed or is new
    else
        return 1  # No change
    fi
}

# Build images
docker_check

# Authenticate with GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "You will need a GitHub Personal Access Token with 'write:packages' scope."
docker login ghcr.io

echo "Building production images..."
make build-production-server
make build-production-api
make build-production-bootstrap
make build-production-health

# Tag images
echo "Tagging images..."
docker tag kwaainet-api $REPO/kwaainet-api:$VERSION
docker tag kwaainet-api $REPO/kwaainet-api:latest

docker tag kwaainet-node $REPO/kwaainet-node:$VERSION
docker tag kwaainet-node $REPO/kwaainet-node:latest

docker tag kwaainet-bootstrap $REPO/kwaainet-bootstrap:$VERSION
docker tag kwaainet-bootstrap $REPO/kwaainet-bootstrap:latest

docker tag kwaainet-health $REPO/kwaainet-health:$VERSION
docker tag kwaainet-health $REPO/kwaainet-health:latest

echo "Images tagged successfully."

# Check for changed images
echo "Checking for changes in images..."
declare -A image_status
IMAGES=("kwaainet-api" "kwaainet-node" "kwaainet-bootstrap" "kwaainet-health")

for image in "${IMAGES[@]}"; do
    if docker_image_changed "$image"; then
        image_status[$image]="changed"
    else
        image_status[$image]="unchanged"
    fi
done

# Present options to the user
echo "Please select which images to push:"
echo "0) Push all images"
echo "1) Select individual images"
echo "2) Exit without pushing"
read -r CHOICE

case $CHOICE in
    0)
        # Push all images
        echo "You've chosen to push all images."
        for image in "${IMAGES[@]}"; do
            echo "Pushing $image (${image_status[$image]})..."
            docker push $REPO/$image:$VERSION
            docker push $REPO/$image:latest
        done
        echo "All images pushed successfully."
        ;;
    1)
        # Let user select individual images
        images_to_push=()
        for i in "${!IMAGES[@]}"; do
            image="${IMAGES[$i]}"
            status="${image_status[$image]}"
            echo "Push $image? (Status: $status) [y/N]"
            read -r PUSH
            if [[ "$PUSH" =~ ^[Yy]$ ]]; then
                images_to_push+=("$image")
            fi
        done
        
        if [ ${#images_to_push[@]} -eq 0 ]; then
            echo "No images selected for push."
            exit 0
        fi
        
        echo "Pushing selected images: ${images_to_push[*]}"
        for image in "${images_to_push[@]}"; do
            echo "Pushing $image..."
            docker push $REPO/$image:$VERSION
            docker push $REPO/$image:latest
        done
        echo "Selected images pushed successfully."
        ;;
    2|*)
        echo "Exiting without pushing any images."
        exit 0
        ;;
esac