import torchtext.vocab as vocab
import numpy as np
import requests
import zipfile
import io
np.random.seed(42)

r = requests.get("http://web.stanford.edu/class/cs21si/resources/unit1_resources.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

VEC_SIZE = 300
glove = vocab.GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    return glove.vectors[glove.stoi[word]].numpy()
  
def read_train_examples():
    with open('unit1_resources/train.txt', 'r') as f:
        raw_text = f.read()
        lines = raw_text.split('\n')
        examples = [line.split() for line in lines]
        examples = [(line[0], int(line[1])) for line in examples]
    return examples

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def compute_logistic_regression(word, weights, bias):
    word_vector = get_word_vector(word)
    result = sigmoid(np.dot(weights, word_vector) + bias)
    return result
def fit_logistic_regression (O000O0OO0OOOOO000 ,OOO000O0O000OOO00 =1000 ,O00O0O0OOO00O00OO =0.001 ):
    np .random .seed (42 )
    O0OOO0OO0O0O0OO00 =np .random .randn (VEC_SIZE )
    O0OOO00O0OOOOO000 =0 
    for OO0O0000O000OOO00 in range (OOO000O0O000OOO00 ):
        O0OOOO0OO0OO00000 =0 
        for O00OOO0OO0000OO0O in O000O0OO0OOOOO000 :
            OO0O00O000OOO0O00 ,O00OO0O0O00OO0000 =O00OOO0OO0000OO0O 
            O0O0OO0000OOO0O0O =compute_logistic_regression (OO0O00O000OOO0O00 ,O0OOO0OO0O0O0OO00 ,O0OOO00O0OOOOO000 )
            O0OOOO0OO0OO00000 +=(1 -O00OO0O0O00OO0000 )*np .log (1 -O0O0OO0000OOO0O0O )+O00OO0O0O00OO0000 *np .log (O0O0OO0000OOO0O0O )
            O0OOO0O0OO00O0O0O =O0O0OO0000OOO0O0O -O00OO0O0O00OO0000 
            OOO00OO0O0OOOO000 =O0OOO0O0OO00O0O0O
            OOOOOO0OOOOO0OO0O =get_word_vector (OO0O00O000OOO0O00 )*O0OOO0O0OO00O0O0O
            O0OOO0OO0O0O0OO00 -=O00O0O0OOO00O00OO *OOOOOO0OOOOO0OO0O
            O0OOO00O0OOOOO000 -=O00O0O0OOO00O00OO *OOO00OO0O0OOOO000
        if OO0O0000O000OOO00 %100 ==0 :
            print ("Epoch %d, loss = %f"%(OO0O0000O000OOO00 ,O0OOOO0OO0OO00000 ))
    return O0OOO0OO0O0O0OO00 ,O0OOO00O0OOOOO000 
examples = read_train_examples()
weights, bias = fit_logistic_regression(examples)
def print_test_output(test_examples, weights, bias):
    for test_example in test_examples:
        pred = compute_logistic_regression(test_example, weights, bias)
        print("%s is %s" % (test_example, 'male' if pred < .5 else 'female'))
        
print_test_output(['nurse', 'homemaker', 'carpenter', 'surgeon', 'doctor', 'artist', 
                   'engineer', 'entrepreneur', 'genius', 'intellectual', 'chef', 'cook', 
                   'maid', 'teacher', 'boss', 'manager', 'founder', 'neurosurgeon', 'architect'], weights, bias)
