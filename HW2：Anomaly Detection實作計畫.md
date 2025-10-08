# 注意事項
* 實作時注意程式碼模組化並提升可重用性、可讀性
* 可重用的SlidingWindow
	把sliding window與合併error map的邏輯抽成通用函式，接受任意window size、stride、padding policy
* 數值穩定性
	標準化（z）時guarding against zero std（加 eps）  
	拼回canvas時注意邊界與padding排除
* 驗證設計
	CutPaste的參數（patch 大小、旋轉、scale）要與真實缺陷尺度匹配  
	驗證集必須完全獨立於訓練（不要用同一張圖的不同 patch 同時出現在 train 與 val）　
	建議把原始正常圖split成train/val例如90/10或85/15），只在valsubset上做CutPaste　　

* 評估：image-level AUROC

* 效能考慮
	多尺度推論很慢，先在單一尺度做profiling，再並行化batch推論
	

* 確保可重現
	固定seed、記錄env（PyTorch、CUDA、python版本），儲存config json

# 超參數建議
optimizer: Adam，lr = 1e-4  
weight_decay = 1e-5  
batch_size: 32（視GPU記憶體調整） 
epochs: 200（在訓練過程中每隔幾個epoch就用CutPaste驗證集算一次ROC-AUC。若這個AUC連續多次沒有提升，就停止訓練）
loss: L = MSE + λ*(1−SSIM)；λ = 0.84（可調；可嘗試 0.5 ~ 1.0）
SSIM window: 11（常用）
multi-scale: [128,256,512]；stride = size // 2
top-k: top 0.5% 或 top 1%（試驗看哪個在validation上表現較好）
Gaussian σ: 4（視圖像大小調整）
Ensemble（進階）: 3〜5 models（不同 seed / slight arch variations）


# 實作計畫

## Data Set：MVTec AD dataset
* Train set：3629 張正常圖片
* Test set： 1725 張圖片
* 輸出格式：  
	0: normal  
	1: anomaly  
訓練階段僅使用正常圖片進行重建學習，測試階段判斷重建誤差大小以進行分類	


## 主體架構
U-Net型Convolutional Autoencoder + CutPaste 偽異常驗證 + 多尺度滑窗重建誤差 + Ensemble


### U-Net型Convolutional Autoencoder
使用Convolutional Autoencoder架構進行異常偵測  
為了提升影像重建品質，採用U-Net型卷積自編碼器，在編碼與解碼階段建立跳接，使模型能同時保留局部紋理與全局語意特徵  
使用MSE + λ×(1−SSIM)作為重建損失，以兼顧像素誤差與結構相似度。

訓練細節：  
	為了確保模型能學習到各個尺度的特徵：
	1. 隨機縮放
		將影像縮放到不同大小後再裁剪（例如長邊在[256, 512]之間隨機取值），再resize回模型輸入大小 
		模型因此在訓練中看到不同「相對大小」的物體

	2. 隨機裁剪（Random Crop）  
		與多尺度resize結合，確保模型同時見過局部細節與全局上下文    
		
	3. 多尺度batch訓練（Multi-scale training） 
		在一個epoch內，動態改變輸入解析度，讓模型學到跨尺度的一致特徵  
	 

### CutPaste偽異常驗證
由於需要某種驗證指標來評估模型的表現，但此處的訓練集全是正常圖像，沒有ground truth  
所以用CutPaste自行製造「假異常」
* 作法
	從圖像上隨機剪一塊patch，再貼到別的位置
	這張圖就變成「看起來怪怪的」的偽異常樣本
	這樣就能建立一個「人工標註」的小驗證集：  
		正常圖片→label = 0
		CutPaste圖片→label = 1
	再用這些資料計算image-level AUROC，決定模型是否學到好的重建特徵

### 多尺度滑窗重建誤差（推論時使用）
同一張圖中，不同大小的異常可能出現在不同尺度（例如小刮痕 vs 大面積污漬  
所以讓同一個模型在多個window size下重建圖片  
分別計算重建誤差，再把這些誤差整合起來，以此捕捉不同大小的異常  
小patch→對局部異常敏感  
大patch→對整體結構穩定  

具體參數：
	三種尺度： 128×128（偵測細小刮痕）, 256×256（偵測中等缺陷）, 512×512（偵測大面積缺陷或整體不一致），須注意切割至模型可接受大小，原圖不匹配時進行padding
	
	stride = window_size // 2

合併多尺度的error maps：  
1. 先把patch-level error拼回同原圖大小的canvas（注意不要保留padding區域），為每一尺度產生error_map_scale

2. 不同尺度誤差分布通常不同，需標準化：z = (map - mean)/std

3. element-wise max： final_map = max(map_128, map_256, map_512)

4. 可用 Gaussian smoothing去雜訊

5. 計算anomaly score
	由於直接取整張error map的平均會導致異常的地方被稀釋，因此此處使用top-k平均（例如 top1%或top 0.5%）的方式　　
	把error map攤平成1D後，取其中誤差最大的前k%像素，計算這些top-k的平均值就是anomaly score
	top-k作為超參數，可選1%、0.5%、甚至 0.1%




### Ensemble（進階）
一模型用不同初始化、不同輸入尺寸訓練出多個版本的模型
每個模型給出一組anomaly score
用z-score標準化後平均，能顯著提升穩定性與分數
因為各模型觀察到的細節不同，融合後能互補誤判



### 回答問題
1. Explain your implementation which get the best performance in detail.
2. Explain the rationale for using auc score instead of F1 score for binary classification in this homework.
3. Discuss the difference between semi-supervised learning and unsupervised learning.