from tqdm import tqdm
from parrot import Parrot
import torch
import pickle
import warnings
from random import choice
warnings.filterwarnings("ignore")
GAD_data = {'Feeling nervous, anxious, or on edge': ['Do you feel nervous anxious or on edge',
                                                 'How likely are you to feel this way',
                                                 'Any ideas on what may be causing this',
                                                 'Have you tried any remedies to feel less nervous',
                                                 'Are you also feeling any other symptoms such as jitters or dread'],
        'Not being able to stop or control worrying': ['Do you feel not able to stop or control worrying',
                                                       'How likely are you to feel this way',
                                                       'Any thoughts on what may be causing this',
                                                       'Have you tried any remedies to stop worrying',
                                                       'Are you also feeling any other symptoms'],
        'Worrying too much about different things': ['Do you feel like you worry about too many different things',
                                                     'How likely are you to feel this way',
                                                     'Why do you think this happens',
                                                     'Have you tried any techniques to try not to worry',
                                                     'Do you also feel any other symptoms'],
        'Trouble relaxing': ['Do you feel trouble relaxing',
                             'How likely are you to feel this way',
                             'Any ideas on what may be causing this',
                             'Have you tried any remedies to relax yourself',
                             'Are you also feeling any other symptoms'],
        'Being so restless that it is hard to sit still': ['Do you feel restless sometimes',
                                                           'How often do you feel this way',
                                                           'Any ideas on what may be causing you to feel restless',
                                                           'Have you tried techniques to try to be still',
                                                           'Do you feel any other symptoms when you are restless'],
        'Becoming easily annoyed or irritable': ['Do you feel easily annoyed or irritable',
                                                 'How likely are you to feel this way',
                                                 'Any ideas on what may be causing these feelings',
                                                 'Have you tried any remedies to feel less irritable',
                                                 'Are you also feeling any other symptoms'],
        'Feeling afraid, as if something awful might happen': ['Do you feel afraid or as if something awful might happen',
                                                               'How likely are you to feel this way',
                                                               'Any ideas on why you may be feeling afraid',
                                                               'Have you tried any remedies to calm yourself',
                                                              'Are you also feeling any other symptoms such as dread']}

'''
def random_state(seed):
    torch.manual_seed(seed)

random_state(1234)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5",use_gpu=False)
phrase = GAD_data['Feeling nervous, anxious, or on edge'][0]+'?'
para_phrase = parrot.augment(input_phrase=phrase)
print (para_phrase)
'''
def create_one_dataset():
    
    torch.manual_seed(1234)
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5",use_gpu=False)
    dataset = {}
    print ("iterating through dictionary items")
    for item in tqdm(GAD_data):
        dataset[item] = []
        print ("iterating through dictionary item list")
        for list_item in tqdm(GAD_data[item]):
            phrase = list_item+'?'
            para_phrases = parrot.augment(input_phrase=phrase)
            para_phrases_list = [x[0] for x in para_phrases]
            dataset[item].append(choice(para_phrases_list))

    return (dataset)

n = 100000
datapoints = {}
print ("creating dataset")
for i in tqdm(range(n)):
    x = create_one_dataset()
    datapoints[i] = x

with open('datapoints.pkl','wb') as f:
    pickle.dump(datapoints,f)
'''
dataset = {}
for i in tqdm(range(100000)):
    dataset[i] = GAD_data

with open('dataset.pkl','wb') as f:
    pickle.dump(dataset,f)
'''
