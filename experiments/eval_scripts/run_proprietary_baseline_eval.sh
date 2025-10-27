#!/bin/bash
#
# Run evaluation on proprietary models
#
# Usage:
#   bash run_proprietary_eval.sh <logfile> [models...]
#
# Examples:
#   # Evaluate GPT-3.5 and GPT-4
#   bash run_proprietary_eval.sh results/individual_vicuna7b.json gpt-3.5-turbo-0301 gpt-4-0314
#
#   # Evaluate all models
#   bash run_proprietary_eval.sh results/individual_vicuna7b.json all
#

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <logfile> [models...]"
    echo ""
    echo "Available models:"
    echo "  gpt-3.5-turbo-0301  - GPT-3.5"
    echo "  gpt-4-0314          - GPT-4"
    echo "  claude-instant-1    - Claude 1"
    echo "  claude-2            - Claude 2"
    echo "  text-bison-001      - PaLM-2"
    echo "  all                 - All models"
    exit 1
fi

LOGFILE=$1
shift

# Determine which models to evaluate
if [ $# -eq 0 ] || [ "$1" == "all" ]; then
    MODELS="gpt-3.5-turbo-0301 gpt-4-0314 claude-instant-1 claude-2 text-bison-001"
else
    MODELS="$@"
fi

# Create output filename
BASENAME=$(basename "$LOGFILE" .json)
OUTPUT="eval/proprietary_${BASENAME}.json"

echo "============================================================================"
echo "Proprietary Model Evaluation"
echo "============================================================================"
echo "Input logfile:  $LOGFILE"
echo "Output file:    $OUTPUT"
echo "Models:         $MODELS"
echo "============================================================================"
echo ""

# Check for API keys
missing_keys=""

if [[ $MODELS == *"gpt"* ]] && [ -z "$OPENAI_API_KEY" ]; then
    missing_keys="$missing_keys OPENAI_API_KEY"
fi

if [[ $MODELS == *"claude"* ]] && [ -z "$ANTHROPIC_API_KEY" ]; then
    missing_keys="$missing_keys ANTHROPIC_API_KEY"
fi

if [[ $MODELS == *"bison"* ]] && [ -z "$PALM_API_KEY" ]; then
    missing_keys="$missing_keys PALM_API_KEY"
fi

if [ -n "$missing_keys" ]; then
    echo "WARNING: Missing API keys: $missing_keys"
    echo "Set them as environment variables or pass via command line"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create eval directory
mkdir -p eval

# Run evaluation
python ../evaluate_proprietary_baseline.py \
    --logfile "$LOGFILE" \
    --output "$OUTPUT" \
    --models $MODELS \
    --batch-size 5 \
    --evaluate-every 10 \
    --delay 1.0

echo ""
echo "============================================================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT"
echo "============================================================================"

