name: Manual Run D1_LSTM.py

on:
  workflow_dispatch:  # Pozwala na ręczne uruchamianie workflow

jobs:
  run-lstm:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'  # Wybierz odpowiednią wersję Pythona

    - name: Install NVIDIA CUDA keyring and dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y wget
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda

    - name: Install build tools
      run: |
        sudo apt-get update
        sudo apt-get install -y build-essential

    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install tensorflow tensorflow-gpu
        pip install -r requirements.txt  # Zakładając, że masz plik requirements.txt
        pip install git+https://github.com/statsmodels/statsmodels.git

    - name: Run D1_LSTM.py
      run: python D1_LSTM.py

    - name: Save results
      run: |
        mkdir -p artifacts
        cp D1_EUR_a.pkl artifacts/
        cp D1_USD_a.pkl artifacts/
        cp D5_eur_tabel.pkl artifacts/

    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: results
        path: artifacts/

    - name: Pull latest changes
      run: git pull origin master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Commit and push results
      run: |
        git config --global user.name 'github-actions'
        git config --global user.email 'github-actions@github.com'
        git add artifacts/D1_EUR_a.pkl artifacts/D1_USD_a.pkl artifacts/D5_eur_tabel.pkl
        git commit -m 'Add results from D1_LSTM.py'
        git push
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Move files
      run: |
        mv artifacts/D1_EUR_a.pkl .
        mv artifacts/D1_USD_a.pkl .
        mv artifacts/D5_eur_tabel.pkl .
        git config --global user.name 'github-actions[bot]'
        git config --global user.email 'github-actions[bot]@users.noreply.github.com'
        git add D1_EUR_a.pkl D1_USD_a.pkl D5_eur_tabel.pkl
        git commit -m 'Move D1_EUR_a.pkl and D1_USD_a.pkl and D5_eur_tabel.pkl from artifacts to root directory'
        git push origin master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
