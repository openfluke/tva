#!/bin/bash
# Batch 2 Experimental Benchmarks Runner
# Runs all 5 new benchmarks and collects results

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║   🧪 BATCH 2 EXPERIMENTAL BENCHMARKS                                                                         ║"
echo "║                                                                                                               ║"
echo "║   Running 5 benchmarks:                                                                                      ║"
echo "║   1. Multi-Agent Swarm (60s)                                                                                 ║"
echo "║   2. Catastrophic Forgetting (40s)                                                                           ║"
echo "║   3. Adversarial Robustness (60s)                                                                            ║"
echo "║   4. Transfer Learning (60s)                                                                                 ║"
echo "║   5. Sparse Data Learning (60s)                                                                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARKS=(
    "multi_agent"
    "forgetting"
    "adversarial"
    "transfer"
    "sparse_data"
)

TOTAL=${#BENCHMARKS[@]}
PASSED=0
FAILED=0

for i in "${!BENCHMARKS[@]}"; do
    bench="${BENCHMARKS[$i]}"
    num=$((i + 1))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════════════════"
    echo "[$num/$TOTAL] Running $bench..."
    echo "═══════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    cd "$SCRIPT_DIR/$bench"
    
    if go run .; then
        echo ""
        echo "✅ $bench completed successfully"
        PASSED=$((PASSED + 1))
    else
        echo ""
        echo "❌ $bench failed"
        FAILED=$((FAILED + 1))
    fi
    
    cd "$SCRIPT_DIR"
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║   📊 BATCH 2 RESULTS SUMMARY                                                                                 ║"
echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
echo "║   Passed: $PASSED/$TOTAL                                                                                        ║"
echo "║   Failed: $FAILED/$TOTAL                                                                                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
