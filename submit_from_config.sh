#!/bin/bash
set -e

CONFIG_FILE="config.yaml"
TEMPLATE_FILE="job_template.slurm"
JOB_FILE="job_generated.slurm"

if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Missing config or template"
  exit 1
fi

# Extract config values
read_config() {
  python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$1'])"
}

declare -A config_map=(
  ["JOB_NAME"]="job_name"
  ["OUTPUT_FILE"]="output_file"
  ["ERROR_FILE"]="error_file"
  ["TIME_LIMIT"]="time_limit"
  ["GPUS"]="gpus"
  ["PARTITION"]="partition"
)

# Read template and perform substitution
cp "$TEMPLATE_FILE" "$JOB_FILE"

for placeholder in "${!config_map[@]}"; do
  value=$(read_config "${config_map[$placeholder]}")
  sed -i '' "s|{{${placeholder}}}|${value}|g" "$JOB_FILE"
done

# Submit the job
chmod +x "$JOB_FILE"
sbatch "$JOB_FILE"
