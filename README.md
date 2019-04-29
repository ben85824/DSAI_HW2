# DSAI hw2

模型結構: lstm64->lstm64->lstm64_state->lstm64->fc  
batchsize: 128  
epoch: 50  
training size: 80000  
validation size: 30000  

執行方法:python3 main.py  
總共訓練四個目標(adder, subtractor, adder+subtractor, multiplier)，共用同一個model  
每10個epoch 輸出一次測試結果  

# accuracy
在不同 digits 下, validation data 的準確率  
          |d = 3|d = 4|d = 5|  
adder     | 99% | 99% | 98% |  
subtractor| 99% | 99% | 98% |  
add+sub   | 99% | 98% | 97% |  

# training epoch  
在不同 epoch 下, validation data 的準確率
          |e=10 |e=20 |e=30 |e=40 |e=50 |
adder     | 94% | 97% | 99% | 99% | 99% |
subtractor| 94% | 97% | 99% | 99% | 99% |
add+sub   | 90% | 96% | 98% | 99% | 99% |

# multiplication
用相同的模型和訓練方式，並沒有辦法訓練出乘法器
將hidden size加大效果並不明顯
將layer加深效果也不是很好
可能要另外設計新的模型


