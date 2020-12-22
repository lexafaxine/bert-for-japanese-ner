# BERT for Japanese named-entity-recognition

In Google's original BERT repository(https://github.com/google-research/bert), there is a template code for dealing with the binary classification(run_classifier.py). Named-entity-recognition, which is essentially a sequence labeling problem, can be modified from a binary classification problem. So we can try to use BERT for a Japanese named-entity-recognition.

In this repository, 3 different Japanese Bert model was tried to solve the named-entity-recognition:

1. BERT with SentencePiece を日本語 Wikipedia で学習してモデル https://yoheikikuta.github.io/bert-japanese/
2. Pretrained Japanese BERT models Tohoku University (tokenization:**`mecab-ipadic-char-4k`**): https://github.com/cl-tohoku/bert-japanese
3. Pretrained Japanese BERT models Tohoku University (tokenization:**`mecab-ipadic-bpe-32k`**): https://github.com/cl-tohoku/bert-japanese
4. Fully-Character based model modified from **`mecab-ipadic-char-4k`**

------

### Dataset

Hironsan.txt: https://github.com/Hironsan/IOB2Corpus

------

### File Structure:

```
bert-for-japanese-ner
|____ bert                  # git from https://github.com/google-research/bert
|____ checkpoint	    	# config file
|____ HironsanData		            # train data
|____ ja_checkpoint	        # download from https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O?usp=sharing
|____ ner_ber_multi			    # code of using Google's multilingual model
	|____ bert				# need git from https://github.com/cl-tohoku/bert-japanese
	|____ checkpoint		#download from https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
	|____ HironsanData		            # train data
	|____ output    		    # output of weight of the pre-trained model
	|____ bert-ner.py			# main code
|____ ner_mecab_4k		    # code of using 4k mecab tokenizer
	|____ bert				# need git from https://github.com/cl-tohoku/bert-japanese
	|____ checkpoint		#download from https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k.tar.xz
	|____ HironsanData		            # train data
	|____ output    		    # output of weight of the pre-trained model
	|____ bert-ner.py			# main code
|____ ner_mecab_32k		    # code of using 32k mecab tokenizer
	|____ bert				# need git from https://github.com/cl-tohoku/bert-japanese
	|____ checkpoint		#download from https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k.tar.xz
	|____ HironsanData		            # train data
	|____ output    		    # output of weight of the pre-trained model
	|____ bert-ner.py			# main code
|____ output    		    # output of weight of the pre-trained model
|____ bert-ner.py			# main code
```



### Usage:

Refer to the respectively usage.txt.

For the model with sentencepiece tokenization, we need to write the model_file like this:

```shell
python bert-ner.py --task_name="NER" --do_train=True --do_eval=True --do_predict=True --data_dir=HironsanData --model_file=ja-checkpoint/wiki-ja.model --vocab_file=ja-checkpoint/wiki-ja.vocab --bert_config_file=checkpoint/bert_config.json --init_checkpoint=ja-checkpoint/model.ckpt-1400000 --max_seq_length=128 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./output/result_dir/
```

### Environment:

python == 3.6.1

Tensorflow-gpu == 1.9.0

mecab-python-windows == 0.996.3

sentencepiece == 0.1.91

transformers == 3.5.1

### Result:

After 100 epochs pre-training  on Google Colab, the eval result is:

1. BERT with SentencePiece を日本語 Wikipedia で学習してモデル:

   ```
   eval_f = 0.7630389
   eval_precision = 0.89818686
   eval_recall = 0.7370544
   global_step = 4987
   loss = 160.51292
   ```

   

2. Pretrained Japanese BERT models Tohoku University (tokenization:**`mecab-ipadic-char-4k`**):

   ```
   eval_f = 0.70564973
   eval_precision = 0.8581138
   eval_recall = 0.6796762
   global_step = 4987
   loss = 214.99455
   ```

   

3. Pretrained Japanese BERT models Tohoku University (tokenization:**`mecab-ipadic-bpe-32k`**):

   ```
   eval_f = 0.74224067
   eval_precision = 0.8957223
   eval_recall = 0.7177472
   global_step = 4987
   loss = 158.71263
   ```

   

4. Fully-Character based model modified from **`mecab-ipadic-char-4k`**

   ```
   eval_f = 0.74987495
   eval_precision = 0.9256084
   eval_recall = 0.70091957
   global_step = 4987
   loss = 216.77464
   ```

Perhaps we can conclude that sentencepiece is a ideal tokenization for sequence labeling.

### Future:

1. The dataset is too small, which introduces an oscillation in the results. In the future, I will search for a bigger corpus and do this again.
2. I will deploy this model and a NER can be used directly through internet.