#DSAI hw2

模型結構: lstm64->lstm64->lstm64_state->lstm64->fc
batchsize: 128
epoch: 50
training size: 80000
validation size: 30000

執行方法:python3 main.py
總共訓練四個目標(adder, subtractor, adder+subtractor, multiplier)，共用同一個model

