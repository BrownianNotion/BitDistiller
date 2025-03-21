# Run first to set up virtual environment and correct packages
# should take few min
sudo apt update
# handling large files
sudo apt install git-lfs 
# gpu monitoring
sudo apt install nvtop
# make python3.9 venv (3.9 needed for BitDistiller) and
# install required packages
sudo apt install -y python3.9 python3.9-venv python3.9-dev
python3.9 -m venv BitDistillerVenv
source BitDistillerVenv/bin/activate
pip install -r requirements.txt

# download TinyLlama locally, since repo requires this to run
python models/download_model.py

wandb login
