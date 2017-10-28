MLDS HW1

1.	使用 Python 3.5.2 、 Keras 2.0.8 、 Tensorflow 1.3.0

2.	sh 檔中使用 'python' 來 call python 3

	若您的 python 3 是使用 'python3'  才能 call，則必須要修改

3.	Keras 使用 Tensorflow Backend

4.	train_mean_box, train_std_box 

	為記錄 training features 的平均值、標準差

	用於標準化用

5.	rnn: simpleRNN

	cnn: Conv1d + simpleRNN

	best: hw1_best_model_structure.png

6.	best 於 Azure NC6 VM 上， GPU: 1 x Tesla K80，運行時間約 8 min

7.	Kaggle Edit Distance Score (private/public)

	best: 6.99759/7.14689