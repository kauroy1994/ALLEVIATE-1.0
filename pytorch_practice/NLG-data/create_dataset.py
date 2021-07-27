import pickle
import pandas

#unpickle datafile
f = open('QADiscourse_train_set.pkl','rb')
df = pickle.load(f)
f.close()

#preprocess
#columns 0: context, 1: question, 2: answer
n = len(df)
data = {}

#add to data dictionary
for i in range(n):
    x = df['context'][i]
    if x not in data:
        data[x] = []
    if df['question'][i] != '_':
        y = df['question'][i]
        data[x].append(y)

'''
c = 0
for item in data:
    print (item)
    print (data[item])
    c += 1
    if c == 3:
        break
'''

#create pickle dump
f = open("output.pickle","wb")
pickle.dump(data, f)
