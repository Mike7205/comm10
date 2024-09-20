name: Manual Run D1_LSTM.py

on:
  workflow_dispatch:  # Pozwala na ręczne uruchamianie workflow

jobs:
  run-lstm:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'  # Wybierz odpowiednią wersję Pythona

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt  # Zakładając, że masz plik requirements.txt

    - name: Run D1_LSTM.py
      run: python D1_LSTM.py

    - name: List files after saving
      run: ls -la

    - name: Check if files exist
      run: |
        if [ -f D1_EUR_a.pkl ]; then echo "D1_EUR_a.pkl exists"; else echo "D1_EUR_a.pkl does not exist"; fi
        if [ -f D1_USD_a.pkl ]; then echo "D1_USD_a.pkl exists"; else echo "D1_USD_a.pkl does not exist"; fi
