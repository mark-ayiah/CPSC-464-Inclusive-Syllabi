import json
import re
import requests
import csv
import pandas as pd
import time
from ast import literal_eval
import urllib
from urllib.parse import quote
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #to remove uninformative words
import pandas as pd
import numpy as np
import tensorflow_hub
from sentence_transformers import SentenceTransformer #"tensorflow>=1.7.0", tensorflow-hub


class SyllabiPipeline:
    def __init__(self):
        
        model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = tensorflow_hub.load(model_url)
        
        
        
        
        with open('../backend/library_of_congress/detailed-lcc.json') as detailed_file, open('../backend/library_of_congress/final_merged_lcc.json') as top_level_file:
            self.detailed_lcc = json.load(detailed_file)
            self.top_level_lcc = json.load(top_level_file)
            
            
            
    def _split_lcc(self, call_number):
        """
        Splits a Library of Congress Classification (LCC) call number into a tuple of the form (letter, number).
        """
        if call_number == None:
            return None
        #specifically for how OL formats their LCC call numbers
        out = call_number.replace('-', '') #remove hyphen
        out = re.sub(r'(?<=[A-Z])0+', '', out) #Remove leading zeros after the letters segment
        
        # Adjust regex match for letter and number segments, ensuring float conversion for fractional part
        match = re.match(r'([A-Z]+)(\d+(\.\d+)?)', out)
        
        if match:
            return (match.group(1), float(match.group(2)))
        else:
            return None

    def _lookup_meaning(self, code): #takes in tuple (call number)
        l = []

        try:
            d = self.detailed_lcc[code[0]]
            for i in d:
                if code[1] >= i['start'] and code[1] <= i['stop']:
                    l.append(i['subject'])
        except:
            pass

        return l
        #returns a list of definitions for the code
        
    def _searchby_lccn(self, lccn, fields = 'author_name,subject,lcc,title', limit = 5): 
        r = []
        response = requests.get(f'https://openlibrary.org/search.json?q=lcc:{lccn}&fields={fields}&limit={limit}').json()
        return response['docs']

    def _searchby_isbn(self, isbn, field = 'lcc', limit = 1):
        time.sleep(2) #being polite
        response = requests.get(f'https://openlibrary.org/search.json?q=isbn:{isbn}&fields={field}&limit={limit}').json()

        if bool(response['docs']): #falsy
            #print(response['docs'], isbn) #error checking

            if bool(response['docs'][0].get(f'{field}')): #if there is an lcc
                return response['docs'][0].get(f'{field}')[0] #string, first lcc returned
        else:
            return '' #nothing returned

    def _reformat_openlibrary_lccn(self, syllabus): #doesn't account for specific subclasses
        lccn_tup = []

        for isbn in syllabus['isbn']: #get the lccn
            val = self._split_lcc(self._searchby_isbn(isbn)) #after querying Open library, split them into tuples

            if val is not None:
                lccn_tup.append(val)
                
        return lccn_tup
    
    def _get_all_parents(self, lccn, lcc_data):
        init = lccn[0] + str(lccn[1])
        all_parents = {init} #let itself be a "parent" just in case!

        try:
            d = lcc_data[lccn[0]] #key is first element in list

            for i in d: #for each dictionary in the list
                if lccn[1] >= i['start'] and lccn[1] <= i['stop']: #check if a subset

                    #if this is the deepest node; if itself is the only parent, until now, then overwrite
                    if len(i['parents']) >= len(all_parents): 
                        all_parents = i['parents']
            return all_parents
        except:
            return None


    def _find_most_recent_common_parent(self, tupes, lcc_data):
        node_parent_sets = [self._get_all_parents(t, lcc_data) for t in tupes]
        
        prefixes = {}
        inter = {}
        #get all parents for each prefix
        for t in tupes:
            val = self._get_all_parents(t, lcc_data)
            if val != None:
                if t[0] not in prefixes.keys():
                    prefixes[t[0]] = [val] #make a list with all the floats
                else:
                    prefixes[t[0]].append(val) #make a list with all the floats

        for k,v in prefixes.items():
            inter[k] = list(set(v[0]).intersection(*map(set, v[1:])))[0] #make it a string

        return inter
    
    def _find_diversity_topics(self, div_dict):
        #vals = []
        topics = []
        for k,v in mrcp.items():
            try:
                entries = div_dict[k]
                #vals.append({k: [entry for entry in entries if v in entry['parents']]}) #gives a lot of topics underneath the parent node
                topics.append([entry['subject'] for entry in entries if v in entry['parents']])

            except:
                pass
        return topics

    def _get_prop_occurrences(self, topics_lst, kind = 'by phrase', top_n = 15): #splits by phrase/full subject
        
        stop_words = set(stopwords.words('english'))
        lcc_stop = open("lcc_stop_words.txt", "r").read().split("\n")
        
        all_tags = []

        if type(topics_lst[0]) == str:
            topics_lst = [topics_lst] #kinda nasty srry, but its how it functions

        if kind == 'by phrase':
            for i in topics_lst:
                tags = '. '.join(i).split('. ')
                tags = [x.lower().split('.')[0] for x in tags]
                tags = [x for x in tags if not any(sub in x for sub in lcc_stop)] #get rid of common but uninformative loc terms
                tags = [' '.join([word for word in x.split(' ') if word not in stop_words]) for x in tags] #get rid of stop words
                tags = [x.lstrip().rstrip() for x in tags] #remove leading and trailing ws

                all_tags += tags

        else: #if by words
            for i in topics_lst:
                tags = ' '.join(i).split() #split by words
                tags = [x.lower().split('.')[0].split(',')[0].split(')')[0] for x in tags] #pruning for commas, periods, and parenthesis
                tags = [x for x in tags if x not in lcc_stop and x not in stop_words]
                tags = [x.lstrip().rstrip() for x in tags] 

                all_tags += tags

        #make proportions
        prop = Counter(all_tags) 
        prop = dict(prop.most_common(top_n))
        total = sum(prop.values())
        prop = {k: v/total for k, v in prop.items()}
        return prop
    
    def _search_subjects(self, lcc, topics, field = 'subject', limit = 1, exact_string_matching = False):
        if type(topics) == str:
            time.sleep(2) #being polite
            response = requests.get(f'https://openlibrary.org/search.json?q=lcc:{lcc}&subject={topics}&fields={field}&limit={limit}').json()

            if bool(response['docs']): #falsy
                return response['docs']
            else:
                return '' #nothing returned

        elif type(topics) == list:    
            q = f'https://openlibrary.org/search.json?q=lcc:{lcc}&fields={field}&limit={limit}'

            if exact_string_matching: #for cases where a single word is used
                
                topics = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), topics)) #encode tags
                topics = list(map(lambda x: f"\"{x}\"", topics)) #exact string matching

                #topics = '+OR+'.join(topics) #comma (,) and pipe (|) are similar AND, not OR for some reason
                topics = ''.join(list(map(lambda x: f'&subject={x}', topics)))
                q += topics
            
            else:
                topics = ','.join(topics)
                q += topics
            
            print(q)

            time.sleep(2) #being polite
            response = requests.get(q).json()
            
            if bool(response['docs']): #falsy
                return response['docs']
            else:
                return '' #nothing returned

        else:
            return None
        
    def raos_entropy(self, all_cats):
        #i'm aware this is presently incorrect bc the probably of topics is not btwn 0 and 1, but This Is a Start!
        entropy = 0.0

        # Calculate pairwise cosine distances between topics
        tags = list(all_cats.keys())
        embeddings = self.model(tags) #needs to be embedded over one space
        distance_matrix = np.inner(embeddings, embeddings) #cosine sim

        # rao's entropy
        rqe = 0.0

        for i, cat_i in enumerate(tags):
            for j, cat_j in enumerate(tags):
                p_i = all_cats.get(cat_i, 0) # Probability for category i (fall through if 0)
                p_j = all_cats.get(cat_j, 0)
                #print(p_i, p_j)
                # cosine distance (1 - cosine similarity)
                distance = 1 - distance_matrix[i, j]
                #print(distance)
                
                rqe += p_i * p_j * distance

        return rqe




            
if __name__ == "__main__":

    sp = SyllabiPipeline()
    print(sp._split_lcc('DAW1008.00000000.B37 1987'))

