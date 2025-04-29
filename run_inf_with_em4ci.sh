#!/bin/bash

# Base directories
MODEL_DIR="/Users/jiapengzhao/Desktop/EM4CI/models_xdsl"
LEARNED_MODEL_DIR="/Users/jiapengzhao/Desktop/EM4CI/learned_models"
OUTPUT_DIR="/Users/jiapengzhao/Desktop/EM4CI/output"
INF_EXEC="/Users/jiapengzhao/Desktop/EM4CI/inf.out"
EM4CI_WRAPPER="/Users/jiapengzhao/Desktop/EM4CI/em4ci_wrapper.sh"

# Experiments
EXPERIMENTS=("ex1_TD2_10" "ex2_TD2_10" "ex3_TD2_10" "ex4_TD2_10" "ex5_TD2_10" "ex6_TD2_10" "ex7_TD2_10" "ex8_TD2_10")

# Number of samples (adjust as needed)
NUM_SAMPLES=100

# Query variable
QUERY_VAR="Y"  # Replace with the actual query variable

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over experiments
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    echo "Processing experiment: $EXPERIMENT"

    # Dynamically set DO_VARS based on the experiment
    if [[ "$EXPERIMENT" == "ex2_TD2_10" || "$EXPERIMENT" == "ex4_TD2_10" || "$EXPERIMENT" == "ex6_TD2_10" || "$EXPERIMENT" == "ex7_TD2_10" ]]; then
        DO_VARS=("X1")
    else
        DO_VARS=("X")
    fi
    echo "DO_VARS for $EXPERIMENT: ${DO_VARS[@]}"

    # Paths
    MODEL_FILE="$MODEL_DIR/$EXPERIMENT.xdsl"
    LEARNED_MODEL_BIC_FILE="$LEARNED_MODEL_DIR/$EXPERIMENT/$NUM_SAMPLES/wrapper_learnedModelNum.csv"
    LEARNED_MODEL_DOMAIN_FILE="$LEARNED_MODEL_DIR/$EXPERIMENT/$NUM_SAMPLES/wrapper_domain.csv"

    # Check if the best learned model files exist
    if [[ ! -f "$LEARNED_MODEL_BIC_FILE" || ! -f "$LEARNED_MODEL_DOMAIN_FILE" ]]; then
        echo "Best learned model files not found for $EXPERIMENT. Running em4ci_wrapper.sh to generate them..."
        "$EM4CI_WRAPPER" "$EXPERIMENT" "$NUM_SAMPLES"
        if [[ $? -ne 0 ]]; then
            echo "Error: em4ci_wrapper.sh failed for $EXPERIMENT."
            continue
        fi
    fi

    # Get the best learned model index and domain size
    BEST_MODEL_INDEX=$(tail -n 1 "$LEARNED_MODEL_BIC_FILE")
    DOMAIN_SIZE=$(tail -n 1 "$LEARNED_MODEL_DOMAIN_FILE")

    # Construct the learned model file path
    LEARNED_MODEL_FILE="$LEARNED_MODEL_DIR/$EXPERIMENT/$NUM_SAMPLES/em_${EXPERIMENT}_ED${DOMAIN_SIZE}_${BEST_MODEL_INDEX}.xdsl"

    # Check if the learned model file exists
    if [[ ! -f "$LEARNED_MODEL_FILE" ]]; then
        echo "Error: Learned model file not found: $LEARNED_MODEL_FILE"
        continue
    fi

    # Run inf.out
    echo "Running inf.out for $EXPERIMENT"
    "$INF_EXEC" "$MODEL_FILE" "$LEARNED_MODEL_FILE" "$QUERY_VAR" "${DO_VARS[@]}" "$NUM_SAMPLES" > "$OUTPUT_DIR/${EXPERIMENT}_inf.log" 2>&1

    # Check if inf.out ran successfully
    if [[ $? -ne 0 ]]; then
        echo "Error: inf.out failed for $EXPERIMENT. Check log: $OUTPUT_DIR/${EXPERIMENT}_inf.log"
        continue
    fi

    echo "inf.out completed for $EXPERIMENT. Results saved to $OUTPUT_DIR/${EXPERIMENT}_inf.log"
done

echo "All experiments processed."
