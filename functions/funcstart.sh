# install dependencies for local execution
python3 -m venv .venv
source .venv/bin/activate
sudo apt -y install python3-opencv
pip install --upgrade pip
pip install -r requirements.txt

# Kill Azure Function is already running
pkill func

# start Azure Function locally
func host start &
