rm -rf adapter-transformers-l
module load python/3.8.6-ff
python -m venv vnv/vnv-adp-l
echo "Install adapter latest"
source vnv/vnv-adp-l/bin/activate
wget -O adapters3.0.1.tar.gz https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters3.0.1.tar.gz
tar -xf adapters3.0.1.tar.gz
rm adapters3.0.1.tar.gz
mv adapter-transformers-adapters3.0.1 adapter-transformers-l
cd adapter-transformers-l
pip install .
cd ..
pip install --upgrade pip
pip install -r requirements.txt
deactivate