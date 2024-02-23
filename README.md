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

## Running the model

```
poetry run python ./mol_classifier/classifier.py infer --dir <your directory containing the images> --checkpoint <path to your checkpoint> --output <your output file name>
```
