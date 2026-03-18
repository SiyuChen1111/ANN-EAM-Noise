#!/bin/bash
# Unified Analysis Script for ConvLSTM RT Prediction Model
# Generates 4 core visualizations for each experiment
# 
# Usage:
#   ./run_unified_analysis.sh [experiment_name]
#   ./run_unified_analysis.sh all
#   ./run_unified_analysis.sh exp11_t40
#
# If no argument is provided, analyzes all experiments with results files.

set -e

PROJECT_ROOT="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie"
OUTPUT_BASE="$PROJECT_ROOT/outputs/experiments/mnist_convlstm"
SCRIPT="$PROJECT_ROOT/src/utils/unified_analysis.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_analysis() {
    local exp_name=$1
    local exp_dir="$OUTPUT_BASE/$exp_name"
    
    # Check if experiment directory exists
    if [ ! -d "$exp_dir" ]; then
        print_error "Experiment directory not found: $exp_dir"
        return 1
    fi
    
    # Find results file (prefer with_difficulty version)
    local results_file=""
    if [ -f "$exp_dir"/*_results_with_difficulty.csv ]; then
        results_file=$(ls "$exp_dir"/*_results_with_difficulty.csv | head -1)
    elif [ -f "$exp_dir"/*_results.csv ]; then
        results_file=$(ls "$exp_dir"/*_results.csv | head -1)
    else
        print_warning "No results file found in $exp_dir"
        return 1
    fi
    
    local output_dir="$exp_dir/analysis"
    
    print_header "Analyzing: $exp_name"
    echo "Results file: $results_file"
    echo "Output directory: $output_dir"
    echo ""
    
    # Run analysis
    python "$SCRIPT" "$results_file" "$output_dir"
    
    if [ $? -eq 0 ]; then
        print_success "Analysis complete for $exp_name"
        echo ""
        echo "Generated visualizations:"
        echo "  1. correct_error_rt_comparison.pdf"
        echo "  2. difficulty_analysis.pdf"
        echo "  3. rt_distribution_comparison.pdf"
        echo "  4. speed_accuracy_tradeoff.pdf"
    else
        print_error "Analysis failed for $exp_name"
        return 1
    fi
}

# Main script
print_header "ConvLSTM RT Prediction Model - Unified Analysis"

# Check if Python script exists
if [ ! -f "$SCRIPT" ]; then
    print_error "Analysis script not found: $SCRIPT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Define experiments to analyze
EXPERIMENTS=(
    "exp07_log_norm_full"
    "exp08_balanced"
    "exp10_t25_rt2"
    "exp11_t40"
    "exp12_t40_ep40"
    "learnable_noise_ep100"
)

# Handle command line argument
if [ -n "$1" ]; then
    if [ "$1" = "all" ]; then
        print_header "Running analysis for ALL experiments"
        SUCCESS_COUNT=0
        FAIL_COUNT=0
        
        for exp in "${EXPERIMENTS[@]}"; do
            if run_analysis "$exp"; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        done
        
        print_header "Summary"
        echo "Successful: $SUCCESS_COUNT"
        echo "Failed: $FAIL_COUNT"
        
    elif [ "$1" = "list" ]; then
        echo "Available experiments:"
        for exp in "${EXPERIMENTS[@]}"; do
            echo "  - $exp"
        done
        echo ""
        echo "Usage:"
        echo "  $0 all          # Analyze all experiments"
        echo "  $0 exp11_t40    # Analyze specific experiment"
        echo "  $0 list         # List available experiments"
        
    else
        # Analyze specific experiment
        run_analysis "$1"
    fi
else
    # No argument - analyze all experiments
    print_header "Running analysis for ALL experiments (default)"
    echo "Use '$0 list' to see available experiments"
    echo "Use '$0 <experiment_name>' to analyze specific experiment"
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for exp in "${EXPERIMENTS[@]}"; do
        if run_analysis "$exp"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
    done
    
    print_header "Summary"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAIL_COUNT"
fi

print_header "All Done!"
