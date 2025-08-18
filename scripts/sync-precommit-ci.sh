#!/bin/bash
# sync-precommit-ci.sh
# Script to synchronize pre-commit hooks with CI workflows

set -e

echo "🔄 Synchronizing pre-commit with GitHub Actions..."

# Update pre-commit hooks
echo "📦 Updating pre-commit hooks..."
pre-commit autoupdate

# Check if there are changes
if git diff --quiet .pre-commit-config.yaml; then
    echo "✅ Pre-commit hooks are up to date"
else
    echo "📝 Pre-commit hooks updated:"
    git diff .pre-commit-config.yaml

    echo ""
    echo "🔍 Checking for version mismatches with CI..."

    # Extract ruff version from pre-commit
    PRECOMMIT_RUFF=$(grep -A1 "astral-sh/ruff-pre-commit" .pre-commit-config.yaml | grep "rev:" | sed 's/.*rev: v*//' | tr -d ' ')
    echo "Pre-commit ruff version: v$PRECOMMIT_RUFF"

    # Check GitHub Actions for ruff version
    if grep -r "astral-sh/ruff-action" .github/workflows/; then
        echo "Found ruff in GitHub Actions workflows"
    fi

    echo ""
    echo "💡 Remember to:"
    echo "  1. Review the changes in .pre-commit-config.yaml"
    echo "  2. Test the updated hooks: pre-commit run --all-files"
    echo "  3. Update any corresponding GitHub Actions versions if needed"
    echo "  4. Commit the changes"
fi

echo ""
echo "🎯 Sync complete!"
