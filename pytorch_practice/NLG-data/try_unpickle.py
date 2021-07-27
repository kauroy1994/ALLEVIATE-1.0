import pickle

data = None
with open("dataset.pkl","rb") as f:
    data = pickle.load(f)

#example for 0th dataset creation, change accordingly
n = 0
dn = []
for patient in data:
    patient_dict = data[patient]
    for question in patient_dict:
        with open("data"+str(n)+".csv","a") as f:
            f.write(question.replace+'; ')
        '''
        x = question
        '''
        generated_questions = patient_dict[question]
        '''
        y = generated_questions[n]
        dn.append((x,y))
        #print (dn)
        #input()
        '''
        with open("data"+str(n)+".csv","a") as f:
            f.write(generated_questions[n]+"\n")
