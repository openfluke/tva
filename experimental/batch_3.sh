#!/bin/bash
# Batch 3: Real-Time Critical Benchmarks
# These scenarios are IMPOSSIBLE for NormalBP - blocking = failure

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║   🧪 BATCH 3: REAL-TIME CRITICAL BENCHMARKS                                                                  ║"
echo "║                                                                                                               ║"
echo "║   Running 5 benchmarks where BLOCKING = FAILURE:                                                             ║"
echo "║   1. Heartbeat Monitor (100Hz) - blocking = patient death                                                    ║"
echo "║   2. Reflex Game (50ms) - blocking = missed stimulus                                                         ║"
echo "║   3. Balance Beam (200Hz) - blocking = crash                                                                 ║"
echo "║   4. Packet Router (500pkt/s) - blocking = dropped packets                                                   ║"
echo "║   5. Auction Bidder (50tick/s) - blocking = missed profit                                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARKS=(
    "heartbeat"
    "reflex"
    "balance"
    "router"
    "auction"
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
echo "║   📊 BATCH 3 RESULTS SUMMARY                                                                                 ║"
echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
echo "║   Passed: $PASSED/$TOTAL                                                                                        ║"
echo "║   Failed: $FAILED/$TOTAL                                                                                        ║"
echo "║                                                                                                               ║"
echo "║   💡 Expect: Step-based modes should MASSIVELY outperform NormalBP                                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
