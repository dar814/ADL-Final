# ADL-Final
## structure
### Multiple_Choice_github
- data:the 2 dataset for ARC and cycic
- plot.py:plot the training loss for 2 datasets
- predict.sh(predict.py):predict the multiple choice result based on finetuned model
- train.sh(train.py):finetune the bert-model with ARC or Cycic dataset
### Google Drive(https://drive.google.com/drive/folders/1IL1bLtiHDDZf2eZHe49f3fQ9QkZ0sFAb?usp=drive_link)
- Unity_project:The game project of Multiple Choice Go-Kart
- model:The finetuned model originally put in Multiple_Choice_github
- Game_windows:The game for windows platform
- Game_MAC:The game for MAC platform

## Run Training
### Cycic
- bash train.sh data/cycic_multiple_choice/train.csv data/cycic_multiple_choice/test.csv model/CYCIC cycic
### ARC
- bash train.sh data/ARC/train.csv data/ARC/test.csv model/ARC ARC

## Do Prediction
### Cycic
- bash predict.sh data\cycic_multiple_choice\test.csv cycic predict_cycic.csv
### ARC
- bash predict.sh data\ARC\test.csv ARC predict_ARC.csv
