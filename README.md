# Uniqlo Label Defect Classification by Deep Learning
## 背景介紹:
此次為分辨Uniqlo產品標籤屬於OK或NG(分為走紗及油汙)品。

![image](https://github.com/tddwso/Uniqlo-Label-Defect-Classification-by-Deep-Learning/blob/main/%E5%88%86%E9%A1%9E%E7%85%A7.PNG)

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

![image](https://github.com/tddwso/label-identity/blob/main/ACC.PNG)

ROC曲線 (Receiver operating characteristic curve) & AUC (Area Under Curve)

ROC曲線會以對角線為基準，曲線下的面積(AUC)來判別ROC曲線的鑑別力，AUC數值的範圍從0到1，數值愈大，代表模型的鑑別力越好。

![image](https://github.com/tddwso/label-identity/blob/main/ROC.PNG)

實際測試結果(產品分類編號['走紗': 0, '油汙': 1, 'OK': 2]

![image](https://github.com/tddwso/label-identity/blob/main/test1.PNG)

## 使用Streamlit App展示成果

![image](https://github.com/tddwso/Uniqlo-Label-Defect-Classification-by-Deep-Learning/blob/main/Stream%20Logo.png)

Streamlit 是一個開源Python函式庫，可以快速製作Data App。

![image](https://github.com/tddwso/Uniqlo-Label-Defect-Classification-by-Deep-Learning/blob/main/streamlit.png)


