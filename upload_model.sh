MODEL_PATH=$1
MODEL_NAME=$2

REPO="$MODEL_NAME"

huggingface-cli repo create $REPO
huggingface-cli upload $REPO $MODEL_PATH --repo-type model