#!/bin/bash
# Quick commit and push script - bypasses all validations
# Usage: ./scripts/quick_commit.sh "commit message"

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if commit message is provided
if [ $# -eq 0 ]; then
    print_error "No commit message provided"
    echo "Usage: $0 \"commit message\""
    echo ""
    echo "Example:"
    echo "  $0 \"fix: resolve compilation errors\""
    echo "  $0 \"wip: work in progress\""
    exit 1
fi

COMMIT_MESSAGE="$1"

print_status "Starting quick commit and push (no validations)..."

# Remove any existing git lock files
if [ -f .git/index.lock ]; then
    print_warning "Removing existing git lock file"
    rm -f .git/index.lock
fi

# Check git status
print_status "Checking git status..."
if ! git status --porcelain | grep -q .; then
    print_warning "No changes to commit"
    exit 0
fi

# Stage all changes
print_status "Staging all changes..."
git add .

# Show what will be committed
print_status "Files to be committed:"
git diff --cached --name-status

# Commit with --no-verify to bypass pre-commit hooks
print_status "Creating commit: '$COMMIT_MESSAGE'"
if git commit --no-verify -m "$COMMIT_MESSAGE"; then
    print_success "Commit created successfully"

    # Get commit hash
    COMMIT_HASH=$(git rev-parse --short HEAD)
    print_success "Commit hash: $COMMIT_HASH"
else
    print_error "Failed to create commit"
    exit 1
fi

# Push with --no-verify to bypass pre-push hooks
print_status "Pushing to remote..."
if git push --no-verify; then
    print_success "Push completed successfully"

    # Show current branch and remote info
    CURRENT_BRANCH=$(git branch --show-current)
    REMOTE_URL=$(git remote get-url origin)
    print_success "Pushed to: $REMOTE_URL ($CURRENT_BRANCH)"
else
    print_error "Failed to push to remote"
    exit 1
fi

print_success "Quick commit and push completed!"
echo ""
echo "Summary:"
echo "  • Commit: $COMMIT_HASH"
echo "  • Message: $COMMIT_MESSAGE"
echo "  • Branch: $(git branch --show-current)"
echo "  • Remote: $(git remote get-url origin)"