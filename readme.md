# Overview
The goal of the competition is to create an energy prediction model of prosumers to reduce energy imbalance costs.

This competition aims to tackle the issue of energy imbalance, a situation where the energy expected to be used doesn't line up with the actual energy used or produced. Prosumers, who both consume and generate energy, contribute a large part of the energy imbalance. Despite being only a small part of all consumers, their unpredictable energy use causes logistical and financial problems for the energy companies.

https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers

# Set up data

```
kaggle competitions download -c child-mind-institute-detect-sleep-states -p data/original_data

```

Then unzip

# How To

Run
- preprocess.py to create traning dataset
- train.py --train to train a lgbm with classification setup