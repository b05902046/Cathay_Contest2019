# Cathay_Contest2019  
## Contributor: 廖昱誠,姜成翰,李佳忞,林昱賢  
### Last Updated: 2019/09/17  
Files:  

檔名 					| 說明
------------------	| ------------------
train.csv      		| 原始training data
test.csv       		| 原始testing data
train_ans.csv. 		| training data的答案
Cathay_DNN.py DNN   | DNN模型
Cathay_DNN.ipynb.   | DNN模型 for jupyter
newtrain.csv			| 去NA的train.csv
shaun					| 昱賢的資料夾
patrick				| 昱誠的資料夾
newtest.csv   | 去NA的test.csv
output        | 上傳的csv檔
getNewCSV.py  | 待補

個人進度：  

*  昱賢:  
  9/15：用隨機分類樹train沒Nan的資料，train出的準確率約97%，可能overfit
*  昱誠:  
  9/16: random forest 在balanced上約0.5  
        complementNB 在unbalanced約0.74  
  9/25: 使用imblearn做data resampling, 最後再用SVM classifier分類  
        用   SMOTE resample 後, 準確率0.7307411168  
        用 ADASYN resample 後, 準確率0.7343147208  
        ------------------------------------------  
        可以試試看ADASYN後的data拿給不同calssifier train, 可能有機會破紀錄
*  佳忞:  
*  成翰:  
  9/26: 用 deep variational information bottleneck 防止 overfit, 失敗  
  9/27: Use DNN as binary classifier, model architecture and params are as follow: 
  
  Layers  | Type
  ------------------	| ------------------
  1  |  Dense(121, 256)
  2  |  Dropout(p = 0.9)
  3  |  Dense(256, 192)
  4  |  Dropout(p = 0.7)
  5  |  Dense(192, 128)
  6  |  Dense(128, 64)
  7  |  Dense(64, 2)
  
  Params | Discription
  -- | --
  Batch size| 19563
  Learning rate | 0.005
  Optimizer | Adam
  Objective | Hinge loss
  Epoch | 200
  Step  | 9
  Input dims | 121
  
  
目前比賽準確率 ==> 0.7359390863

Issues:  


### Note
大家每次上傳的時候麻煩順便更新一下README，寫一下個人進度更新，還有檔案的說明~
如果不會markdown的話，可以看看這個[連結](https://guides.github.com/features/mastering-markdown/)

*大家比賽加油~*  
> ### We are slow walkers, but we never move backward.

<div style="text-align: right">By Shaun</div>



