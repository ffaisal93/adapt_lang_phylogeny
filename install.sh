deactivate
rm -rf adapter-transformers
rm -rf venv vnv/vnv-org
module load python/3.8.6-ff
python -m venv vnv/vnv-org
echo "install adapter-transformers orginal"
python -m venv vnv/vnv-org
source vnv/vnv-org/bin/activate
wget -O adapters2.3.0.tar.gz "https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters2.3.0.tar.gz"
tar -xf adapters2.3.0.tar.gz
rm adapters2.3.0.tar.gz
mv adapter-transformers-adapters2.3.0 adapter-transformers
cp new_scripts/adapter_training.py adapter-transformers/src/transformers/adapters/training.py
cp new_scripts/trainer_orig.py adapter-transformers/src/transformers/trainer.py
cp new_scripts/training_args.py adapter-transformers/src/transformers/training_args.py
cd adapter-transformers
pip install .
cd ..
pip install -r requirements.txt
deactivate
