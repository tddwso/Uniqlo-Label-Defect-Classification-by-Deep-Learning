# label-identity
## 背景介紹:
此次為分辨產品標籤是否為OK或NG(分為走紗及油汙)品。
## 預計完成目標:
以卷積神經網絡(Convolutional Neural Network)學習分辨OK及NG品。
運用Transfer Learning(遷移式學習)，將他人訓練好的(pre-trained model)參數複製過來，當作我們模型參數，
使用的模型: VGG16，VGG 是英國牛津大學 Visual Geometry Group 的縮寫，主要貢獻是使用更多的隱藏層，大量的圖片訓練，提高準確率至90%。
## 資料集:
Train Data : 360
## 使用環境:
Python 3.8

TensorFlow 2.3.1 
## 訓練和測試結果
最佳模型訓練準確度100% 
























