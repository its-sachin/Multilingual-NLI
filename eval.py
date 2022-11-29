import sys
from sklearn import metrics

def read_output(path):
    ans = []
    with open(path,'r') as file:
        for l in file.readlines():
            line = l.strip()
            ans.append(line)
    return ans

gold = read_output(sys.argv[1])
pred = read_output(sys.argv[2])
micro = metrics.f1_score(gold, pred, average='micro')
macro = metrics.f1_score(gold, pred, average='macro')

print(f'micro: {micro}\nmacro: {macro}')