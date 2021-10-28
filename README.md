# 以Pytorch_DCGAN實現高階影像擴增

## 使用train.py進行模型訓練
輸入的隨機亂數會透過全卷積結構進行修復，過程中會由生成器及鑑別器相互對抗，最終輸出模型權重(.pth)。輸入dim不用太複雜，透過類神經網路模擬就可模擬資料分佈
          
## 使用generate.py進行模型推論
給定裝有權重的路徑、重複實驗的次數和虛擬影像輸出的路徑，就能生成虛擬影像。
    
## 新增功能FID.py(2021/10/28實現)
生成影像後必須要衡量影像的品質，然而除了目視檢查外，亦可以使用FID進行評估，但是僅做一次是不夠的，因此嘗試進行重複實驗，以降低實驗變異。最好組內變異越小越好(真實vs. 真實)，組外變異越大越好(真實vs. 虛擬)

## ref
https://github.com/Natsu6767/DCGAN-PyTorch
https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
