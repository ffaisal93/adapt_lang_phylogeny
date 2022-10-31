deactivate
rm -rf adapter-transformers-joint
rm -rf venv vnv/vnv-joint
module load python/3.8.6-ff
python -m venv vnv/vnv-joint
echo "install adapter-transformers joint"
python -m venv vnv/vnv-joint
source vnv/vnv-joint/bin/activate
wget -O adapters2.3.0.tar.gz "https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters2.3.0.tar.gz"
tar -xf adapters2.3.0.tar.gz
rm adapters2.3.0.tar.gz
mv adapter-transformers-adapters2.3.0 adapter-transformers-joint
cp new_scripts/adapter_training.py adapter-transformers-joint/src/transformers/adapters/training.py
cp new_scripts/trainer_joint.py adapter-transformers-joint/src/transformers/trainer.py
cp new_scripts/trainer_adapter_joint.py adapter-transformers-joint/src/transformers/adapters/trainer.py
cp new_scripts/training_args.py adapter-transformers-joint/src/transformers/training_args.py
cd adapter-transformers-joint
pip install .
cd ..
pip install -r requirements.txt
deactivate
