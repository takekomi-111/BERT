#ライブラリ・モジュールのインストール
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

#CUDAを利用できるかどうかをしらべるためにpytorchをインポート
import torch
print(torch.cuda.is_available())

#日本語モデルのimport
modelname = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(modelname)

#入力テキスト作成
text_list = [
"この映画は面白かった",
"この映画はがっかりした",
"この映画は感動した"
]
label_list=[1,0,1]

text="花びらのように舞い散る。"

#tokenize
encoding1 = tokenizer.tokenize(text)
print(f'encoding1:{encoding1}')

#encoding
encoding2 = tokenizer.encode(text)
print(f'encoding2:{encoding2}')

#convert_ids_to_tokens
encoding3 = tokenizer.convert_ids_to_tokens(encoding2)
print(f'encoding3:{encoding3}')

#main処理
encoding = tokenizer(
    text_list,
    padding='longest',
    return_tensors = 'pt'
)

#encoding
print(f'encoding_before:{encoding}')
encoding = {k: v.cuda() for k, v in encoding.items()} #kのキーに対してv.cuda()の結果を追加する動作挙動
print(f'encoding_after:{encoding}')


