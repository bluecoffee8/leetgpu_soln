#!/bin/bash

# OpenMP Reduction Testing Script
# Demonstrates all compilation and usage methods

set -e

COMPILER="$(brew --prefix gcc)/bin/g++-15"
CFLAGS="-O3 -fopenmp"

echo "============================================"
echo "OpenMP Reduction - Comprehensive Test Suite"
echo "============================================"
echo ""

# Test 1: Standalone executable
echo "[1] Compiling standalone executable..."
$COMPILER $CFLAGS -o reduction_openmp reduction_openmp.cpp
echo "✓ Created reduction_openmp (executable)"
echo ""

# Test 2: Run executable with different parameters
echo "[2] Running standalone executable tests..."
echo "  - Default parameters (4 threads, 10M elements):"
timeout 10 ./reduction_openmp | head -15
echo ""
echo "  - Custom parameters (8 threads, 1M elements):"
timeout 10 ./reduction_openmp 8 1000000 | grep -E "^(Threads:|Array|Method|---)" | head -5
echo ""

# Test 3: Compile to object file
echo "[3] Compiling to object files..."
$COMPILER $CFLAGS -c -o reduction_openmp.o reduction_openmp.cpp
echo "✓ Created reduction_openmp.o (object file with main)"
$COMPILER $CFLAGS -c -o reduction_lib.o reduction_lib.cpp
echo "✓ Created reduction_lib.o (library object file)"
echo ""

# Test 4: Inspect object files
echo "[4] Inspecting object files..."
echo "  Object file info:"
file reduction_openmp.o
file reduction_lib.o
echo ""
echo "  Symbol counts:"
echo "    reduction_openmp.o: $(nm reduction_openmp.o 2>/dev/null | wc -l) symbols"
echo "    reduction_lib.o: $(nm reduction_lib.o 2>/dev/null | wc -l) symbols"
echo ""
echo "  Reduction functions in reduction_lib.o:"
nm reduction_lib.o 2>/dev/null | grep "reduction_" | grep " T " | sed 's/.*_Z/    /' | head -10
echo ""

# Test 5: Create and link example program
echo "[5] Compiling example program..."
$COMPILER $CFLAGS example_usage.cpp reduction_lib.o -o example_usage
echo "✓ Created example_usage (linked with reduction_lib.o)"
echo ""

# Test 6: Run example program
echo "[6] Running example program..."
./example_usage 4
echo ""

# Test 7: Show file sizes
echo "[7] File sizes:"
echo "  Executables:"
ls -lh reduction_openmp example_usage 2>/dev/null | awk '{print "    " $9 ": " $5}'
echo "  Object files:"
ls -lh *.o 2>/dev/null | grep reduction | awk '{print "    " $9 ": " $5}'
echo ""

echo "============================================"
echo "✓ All tests completed successfully!"
echo "============================================"
echo ""
echo "Quick reference:"
echo "  Run benchmarks:      ./reduction_openmp [threads] [array_size]"
echo "  Run example:         ./example_usage [threads]"
echo "  View full README:    cat README_REDUCTION.md"
echo ""
