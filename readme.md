# Overview
The goal of the competition is to create an energy prediction model of prosumers to reduce energy imbalance costs.

This competition aims to tackle the issue of energy imbalance, a situation where the energy expected to be used doesn't line up with the actual energy used or produced. Prosumers, who both consume and generate energy, contribute a large part of the energy imbalance. Despite being only a small part of all consumers, their unpredictable energy use causes logistical and financial problems for the energy companies.

https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers

# Set up data

```
kaggle competitions download -c predict-energy-behavior-of-prosumers -p data/original_data

```

Then unzip inside folder original_data

Create the required environment by executing following command:
```
//create venv
python -m venv .venv

//activate .venv
source .venv/Scripts/activate

//upgrade pip
python -m pip install --upgrade pip

//instal package in editable mode
python -m pip install -e .

//clean egg-info artifact
python setup.py clean
```

or simply execute install_all.bat
# How To

Run
- enefit/preprocess.py to create traning dataset
- enefit/train.py to train a lgbm with regression setup
