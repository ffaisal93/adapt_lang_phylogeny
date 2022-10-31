deactivate
rm -rf transformers-orig
rm -rf venv vnv/vnv-trns
module load python/3.8.6-ff
python -m venv vnv/vnv-trns
echo "install transformers orginal"
python -m venv vnv/vnv-trns
source vnv/vnv-trns/bin/activate
wget -O transformersv4.21.1.tar.gz "https://github.com/huggingface/transformers/archive/refs/tags/v4.21.1.tar.gz"
tar -xf transformersv4.21.1.tar.gz
rm transformersv4.21.1.tar.gz
mv transformers-4.21.1 transformers-orig
cd transformers-orig
pip install .
cd ..
pip install -r requirements.txt
deactivate
