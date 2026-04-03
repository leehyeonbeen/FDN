# Frequency-aware Decomposition Network (FDN)
Official implementation of `*Frequency-aware decomposition learning for sensorless wrench forecasting on a vibration-rich hydraulic manipulator*' (2026).

### **Code and data will be made available after publication.**

<!-- 
Requires ```python==3.10``` ```cuda>=11.8```
1. Clone this repository
2. Move to cloned directory and install requirements
```bash
cd PDF/ && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
3. [Download](https://drive.google.com/file/d/1EO3wGkW5VUGB3tu3URIasTTOYg63NlHy/view?usp=share_link) datasets to the cloned directory
4. Untar the datasets
```bash
tar -xvzf data_hydraulic_250810.tar.gz
```
> [!IMPORTANT]
> Your ```data``` directory structure should be:
> ```
> data
> ├── data_hydraulic
> ├── pretraining_data_daily_manipulation
> ├── pretraining_data_multishape_insertion
> └── *.py 
> ```
5. Run data preprocessing codes
```bash
python data/process_data_hydraulic_mp.py
python data/process_data_daily_manipulation.py
python data/process_data_multishape_insertion.py
``` -->
