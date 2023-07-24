# 导入必要的库
import nltk
#nltk.download('cmudict')
from nltk.corpus import cmudict

# 获取CMU音素字典
d = cmudict.dict()

# 定义音素拼接算法
def get_phonemes(word):
    if word.lower() in d:
        return d[word.lower()][0]
    else:
        return ['<unk>']

# 定义转换函数
def text2phonemes(text):
    words = nltk.word_tokenize(text)
    phonemes = []
    for word in words:
        phonemes += get_phonemes(word)
    return ['<BOS>'] + phonemes + ['<EOS>']

# 测试
#text = 'printing , in the only sense with which we are at present concerned , differs from most if not from all the arts and crafts represented in the exhibition","Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
#import time
#s = time.time()
#text = 'I say neither yea nor nay'
#text = "you are so nervous, didn't you"
#text = "I know you."
text = "The rainbow is a division of white light into many beautiful colors."
phonemes = text2phonemes(text)
#print(time.time()-s,'s')
print(phonemes)

