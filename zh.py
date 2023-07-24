# import torch
# import librosa
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModel
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# processor = Wav2Vec2Processor.from_pretrained(
#     'qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1')
# model = Wav2Vec2ForCTC.from_pretrained('qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1').to(device)
# #model = torch.load('/home/lab-su.di/huggingface/model/wenet/20210618_u2pp_conformer_exp/final.pt')

# filepath = '/home/lab-su.di/diffspeech/Chinese.wav'
# audio, sr = librosa.load(filepath, sr=16000, mono=True)
# inputs = processor(audio, sample_rate=16000, return_tensors="pt").to(device)
# with torch.no_grad():
#     logits = model(inputs.input_values,
#                    attention_mask=inputs.attention_mask).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# pred_str = processor.decode(predicted_ids[0])
# print(pred_str)

# from lasr.process.asrprocess import ASRProcess

# train_config="/home/lab-su.di/huggingface/lighting-asr/hparams.yaml" 
# decode_config="/home/lab-su.di/huggingface/lighting-asr/decode.yaml"
# model_path="/home/lab-su.di/huggingface/lighting-asr/model.ckpt"
# asrpipeline = ASRProcess(
#     train_config=train_config, 
#     decode_config=decode_config, 
#     model_path=model_path
# )
# token, text = asrpipeline('/home/lab-su.di/diffspeech/Chinese.wav')
# print(token)
# print(text)


## Evaluation

#wer_metric = load_metric("wer")
#def compute_metrics(pred):
#    pred_logits = pred.predictions
#    pred_ids = np.argmax(pred_logits, axis=-1)
#    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
#    pred_str = processor.batch_decode(pred_ids, spaces_between_special_tokens=True)
#    label_str = processor.batch_decode(pred.label_ids, group_tokens=False, spaces_between_special_tokens=True)
#    wer = wer_metric.compute(predictions=pred_str, references=label_str)
#    return {"wer": wer}

# import nltk
# import jieba
# #nltk.download('cmudict')
# #from nltk.corpus import cmudict

# # 获取CMU音素字典
# #d = cmudict.dict()

# my_dict = {}

# # 打开文本文件，并逐行读取
# with open('lexicon.txt', 'r') as f:
#     for line in f:
#         # 去掉每行两端的空格和换行符
#         line = line.strip()
#         # 使用空格分割每行数据
#         data = line.split()
#         # 将第一列作为键，第二列作为值，添加到字典中
#         my_dict[data[0]] = data[1]

# # 打印字典
# #print(my_dict)

# # 定义音素拼接算法
# def get_phonemes(word):
#     if word.lower() in my_dict:
#         return my_dict[word.lower()][0]
#     else:
#         return ['<unk>']

# # 定义转换函数
# def text2phonemes(text):
#     words = nltk.word_tokenize(text)
#     phonemes = []
#     for word in words:
#         phonemes += get_phonemes(word)
#     return ['<BOS>'] + phonemes + ['<EOS>']

# 测试
#text = "The rainbow is a division of white light into many beautiful colors."
#pred_str = '我只很感及没有能够表达我的感受'
#text = jieba.cut(pred_str,cut_all=False)
# phonemes = text2phonemes(pred_str)
# print(phonemes)

# from snownlp import SnowNLP
# s = SnowNLP(pred_str)
# print(s.pinyin)

import whisper
import time
model = whisper.load_model("tiny").to('cpu')
s = time.time() 
result = model.transcribe("/home/lab-su.di/diffspeech/ref_audio/Chinese.wav")
t = time.time()
print(result["text"],t-s,"s")
from phkit.chinese.sequence import text2phoneme
phoneme = text2phoneme(result['text'])
p = time.time()
print(phoneme,p-t,"s")