#ライブラリ・モジュールのインストール
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from datasets import load_dataset
from nlp import load_dataset
from sklearn.model_selection import train_test_split

import csv
import os
import glob

#CUDAを利用できるかどうかをしらべるためにpytorchをインポート
import torch
print(torch.cuda.is_available())

#classの定義
def tokenize(batch):
	return  tokenizer(batch["text"], padding = True , truncation=True)

#日本語モデルのimport
modelname = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(modelname)

text="花びらのように舞い散る。"
#tokenize
encoding1 = tokenizer.tokenize(text)
print(f'encoding1:{encoding1}')

#encode
encoding2 = tokenizer.encode(text)
print(f'encoding2:{encoding2}')

#convert_ids_to_tokens
encoding3 = tokenizer.convert_ids_to_tokens(encoding2)
print(f'encoding3:{encoding3}')

#encoding
#print(f'encoding_before:{encoding}')
#encoding = {k: v.cuda() for k, v in encoding.items()} #kのキーに対してv.cuda()の結果を追加する動作挙動
#print(f'encoding_after:{encoding}')

#ファインチューニング###################################################################################################
hogehoge = load_dataset("imdb")
print(f'hogehoge:{hogehoge}')
train_data, eval_data = load_dataset("imdb",split=["train","test[:20%]"]) #後述の補足:点線部分をsplit=[]で分けている。

##補足
#load_datasetに名前だけ指定

#・'train'
#　Dataset
#　　・'text',
#　　・'label'
#・num_rows
#----------------------------------
#・'test'
#　Dataset
#　　・'text',
#　　・'label'

#変数設定
executedir="./"
text_path=executedir + "text/" 
f_or_d_names=os.listdir(path=text_path)
text_label_data=[]

#classifiednameの取得方法
dirs = [d for d in f_or_d_names if os.path.isdir(os.path.join(text_path,d))]
for i in range(len(dirs)):
	dir = dirs[i]
	files = glob.glob(text_path + dir + "/*.txt")

	for file in files:
		if  os.path.basename(file) == "LICENSE.txt":
			continue
		with open(file,"r") as f:
			text = f.readlines()[3:]
			text = "".join(text)
			text = text.translate(str.maketrans({"\n":"","\r":"","\t":"","\u3000":""}))
			text_label_data.append([text,i])

train_data,test_data = train_test_split(text_label_data)
csv_path = executedir + "csv/"

if not os.path.exists(csv_path):
	os.makedirs(csv_path)

with open(csv_path+"train_data.csv","w") as f:
	writer = csv.writer(f)	
	writer.writerows(train_data)
with open(csv_path+"test_data.csv","w") as f:
	writer = csv.writer(f)	
	writer.writerows(test_data)



#csvfileの読込
print(train_data)
train_data = load_dataset("csv", data_files=csv_path+"train_data.csv", column_names=["text","label",], split="train") #会社だとうまく行かなかったが家パソコンだとうまくいく


#データの前処理
train_data=train_data.map(tokenize, batched=True, batch_size=len(train_data)) #batch定義されるオブジェクトそれぞれに対してtokenize関数rを実行するということ)
train_data.set_format("torch", columns=["input_ids","attention_mask","label"]) #datasetをpytorchやnumpy形式など指定して変換することができる
eval_data=eval_data.map(tokenize, batched=True, batch_size=len(eval_data))
train_data.set_format("torch", columns=["input_ids","attention_mask","label"])




