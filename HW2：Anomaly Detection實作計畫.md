# 注意事項
* 實作時注意程式碼模組化並提升可重用性、可讀性



# 實作計畫

## Data Set：MVTec AD dataset
* Train set：3629 張正常圖片
* Test set： 1725 張圖片
* 輸出格式：  
	0: normal  
	1: anomaly  


## 主體架構
U-Net型Convolutional Autoencoder + CutPaste 偽異常驗證 + 多尺度滑窗重建誤差 + Ensemble


### U-Net型Convolutional Autoencoder
使用Convolutional Autoencoder架構進行異常偵測  
為了提升影像重建品質，採用U-Net型卷積自編碼器，在編碼與解碼階段建立跳接，使模型能同時保留局部紋理與全局語意特徵  



### CutPaste偽異常驗證
由於需要某種驗證指標來評估模型的表現，但此處的訓練集全是正常圖像，沒有ground truth  
所以用CutPaste自行製造「假異常」
* 作法
	從圖像上隨機剪一塊patch，再貼到別的位置
	這張圖就變成「看起來怪怪的」的偽異常樣本
	這樣就能建立一個「人工標註」的小驗證集：  
		正常圖片→label = 0
		CutPaste圖片→label = 1
	再用這些資料計算ROC-AUC，決定模型是否學到好的重建特徵

### 多尺度滑窗重建誤差
同一張圖中，不同大小的異常可能出現在不同尺度（例如小刮痕 vs 大面積污漬  
所以讓同一個模型在多個window size下重建圖片  
分別計算重建誤差，再把這些誤差整合起來，以此捕捉不同大小的異常  
小patch→對局部異常敏感  
大patch→對整體結構穩定  


### Ensemble（進階）
一模型用不同初始化、不同輸入尺寸訓練出多個版本的模型
每個模型給出一組anomaly score
用z-score標準化後平均，能顯著提升穩定性與分數
因為各模型觀察到的細節不同，融合後能互補誤判
