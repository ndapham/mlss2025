FILEID="16zh-PJ5bDQkQskeh7FoVq6wEMD6YCLvp" 
FILENAME="./data/input/dataset.zip"
wget --no-check-certificate \
     "https://drive.usercontent.google.com/download?id=${FILEID}&confirm=t" \
     -O "${FILENAME}"
unzip $FILENAME -d ./data/input/
rm $FILENAME
python3 src/tools/test1_bitmap.py
pip install -r requirements.txt