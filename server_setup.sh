sudo apt update
#zsh
sudo apt install zsh -y
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

sudo apt install python3
sudo apt install python3-pip
sudo apt install git
sudo apt install neovim



#docker
sudo apt install apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin



#install anaconda
sudo apt update
sudo apt install -y wget
cd /tmp
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
source ~/anaconda3/bin/activate
conda init zsh
source ~/.zshrc


# ollama install
curl -fsSL https://ollama.com/install.sh | sh

#download model and start server
ollama serve


ollama run qwen3:30b

curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hi from Qwen3."}
    ]
  }'



cd ~ 
git clone https://github.com/THUDM/AgentBench.git
cd AgentBench

conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
docker pull mysql
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default  data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu   data/os_interaction/res/dockerfiles --tag local-os/ubuntu


export OPENAI_API_KEY=ollama       # arbitrary, Ollama ignores it locally
export OPENAI_BASE_URL=http://localhost:11434/v1
