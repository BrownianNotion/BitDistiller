MODEL_PATH=$1
MODEL_NAME=$2
OVERWRITE=$3

REPO="$MODEL_NAME"

if [[ $OVERWRITE != "overwrite" ]]; then
    if huggingface-cli repo create $REPO; then
        huggingface-cli upload $REPO $MODEL_PATH --repo-type model
    else
        echo "Did not upload model as repo $Repo already exists and overwrite set to false"
    fi
else
    huggingface-cli repo create $REPO
    if huggingface-cli upload $REPO $MODEL_PATH --repo-type model; then
        echo "\033[0;32mModel $MODEL_PATH successfully uploaded to $REPO\033[0m"
    fi
fi
