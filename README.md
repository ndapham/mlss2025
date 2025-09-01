# mlss2025
Project for class Machine Learning in summer semester 2025. 

Check out our report [here]("https://github.com/ndapham/mlss2025/blob/main/assets/Machine_Learning_Project_Report.pdf").


## 1. Setup

Clone this project:
```
git clone git@github.com:ndapham/mlss2025.git
```

Download data and install necessary packages:
```
bash setup.sh
```
We have a `config.yml` file in the folder `src/configs/`, where you can set the appropriate paths to your data and some hyper parameter for training process.


## 2. Training
Access to the folder `src/tools/` by running this command
```
cd src/tools
```
Run training model by the command
```
python3 train.py --model_name unet --config_path /path/to/your/config.yml
```
Training checkpoints are saved in the folder specified in `config.yml`.
## 3. Inference and Submit

Run this command to create a folder of mask images based on the checkpoint you are using
```
python3 --checkpoint_path /path/to/your/checkpoint --config_path your/config.yml
```

The output images will be saved in a folder named after the checkpoint, located inside the predictions directory specified in your `config.yml`.
