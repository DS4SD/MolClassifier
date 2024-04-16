# MolClassifier

## Installation Instructions

- Ensure your distribution has the Python version 3.11:

  ```
  python3 --version
  pip3 -V
  ```

- Install poetry

  ```
  pip install poetry
  ```

- Install the dependencies

  ```
  poetry install
  ```

## Download the model 

```
wget https://huggingface.co/ds4sd/MolClassifier/resolve/main/models/molclassifier_model.chpt -P ./data/models/
```

## Running the model

- Place the images to annotate in `./data/dataset/`

- Run the model
  ```
  poetry run python ./mol_classifier/classifier.py infer --dir ./data/dataset/ --checkpoint ./data/models/molclassifier_model.chpt --output ./data/output.txt
  ```

- Read predictions in `./data/output.txt`