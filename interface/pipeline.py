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
import pandas as pd
import numpy as np
# import tensorflow_hub as hub
# from sentence_transformers import SentenceTransformer #"tensorflow>=1.7.0", tensorflow-hub


class SyllabiPipeline:
    """
    Class for the Syllabi Pipeline. Contains all the functions necessary for the measuring diversity and making recommendations.
    """
    def __init__(self):
        
        # model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        # self.model = hub.load(model_url)
        
    
        with open('../backend/library_of_congress/detailed-lcc.json') as detailed_file, open('../backend/library_of_congress/final_merged_lcc.json') as top_level_file:
            self.detailed_lcc = json.load(detailed_file)
            self.top_level_lcc = json.load(top_level_file)
        
        self.syllabus = pd.read_csv('../example_syllabi/TEST Syllabi/edstud_lgbt_med')
            
            
    def _split_lcc(self, call_number):
        """
        Splits a Library of Congress Classification (LCC) call number into a tuple of the form (letter, number).
        Args:
            call_number (str): The LCC call number to split.
        Returns:
            tuple: A tuple of the form (alphabetical category, number) representing the LCC call number.
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
        """
        Looks up the meaning of a LCCN in detailed_lcc.json. 
        Args:
            code (tuple): A tuple of the form (alphabetical category, number) representing the LCC call number.
        Returns: a list of definitions for the code.
        """

        l = []

        try:
            d = self.detailed_lcc[code[0]]
            for i in d:
                if code[1] >= i['start'] and code[1] <= i['stop']:
                    l.append(i['subject'])
        except:
            pass

        return l
        
    def _searchby_lccn(self, lccn, fields = 'author_name,subject,lcc,title', limit = 5): 
        """
        Queries the Open Library API using a formatted LCC Call number. 
        See documentation: https://openlibrary.org/dev/docs/api/search
        Args:
            lccn (str): The LCC call number to search for.
            fields (str): The field(s) to return in the response.
            limit (int): The maximum number of results to return
        Returns:
            a dictionary with the specified fields
        """


        r = []
        response = requests.get(f'https://openlibrary.org/search.json?q=lcc:{lccn}&fields={fields}&limit={limit}').json()
        return response['docs']

    def _searchby_isbn(self, isbn, field = 'lcc', limit = 1):
        """
        Queries the Open Library API using an ISBN. 
        See documentation: https://openlibrary.org/dev/docs/api/search
        Args:
            isbn (str): The ISBN to search for.
            field (str): The field(s) to return in the response.
            limit (int): The maximum number of results to return.
        Returns:
            a dictionary with the fields specified.
        """

        
        url = 'https://openlibrary.org/search.json'
        params = {
            'q': f'isbn:{isbn})', 
            'fields': f'{field}',
            'limit': f'{limit}'
            }
        response = requests.get(url, params=params, timeout=20).json()

        if bool(response['docs']): #falsy
            #print(response['docs'], isbn) #error checking

            if bool(response['docs'][0].get(f'{field}')): #if there is an lcc
                return response['docs'][0].get(f'{field}')[0] #string, first lcc returned
        else:
            return '' #nothing returned

    def _get_lccn_for_syllabus(self, syllabus): #doesn't account for specific subclasses
        """
        Gets the LCCN for each book in a syllabus. Ignores more specific subclass demarcations.
        Args:
            syllabus (dict): A dictionary containing a list of ISBNs.
        Returns:
            a list of tuples representing the LCCN for each book in the syllabus.
        """
        

        lccn_tup = []
        # print(syllabus)

        for isbn in syllabus['isbn']: #get the lccn
            val = self._split_lcc(self._searchby_isbn(isbn)) #after querying Open library, split them into tuples

            if val is not None:
                lccn_tup.append(val)
                
        return lccn_tup
    
    def _get_all_parents(self, lccn, lcc_data):
        """
        Gets all the parents for a given LCC code.
        Args:
            lccn (tuple): A tuple of the form (alphabetical category, number) representing the LCC call number.
            lcc_data (dict): A dictionary containing the LCC classification data.
        Returns:
            a list of parents for the given LCC code.
        """

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


    def _find_lcas(self, tupes, lcc_data):
        """
        Gets the least common ancestor for a list of LCC call numbers.
        Args:
            tupes (list): A list of tuples of the form (alphabetical category, number) representing the LCC call numbers.
            lcc_data (dict): A dictionary containing the LCC classification data.
        Returns:
            a dictionary with the key as the LCC class and the value as the LCA range.
        """

        node_parent_sets = [self._get_all_parents(t, lcc_data) for t in tupes]
        
        prefixes = {}
        inter = {}
        #get all parents for each prefix
        for t in tupes:
            val = self._get_all_parents(t, lcc_data)
            val = self._get_all_parents(t, lcc_data)
            if val != None:
                if t[0] not in prefixes.keys():
                    prefixes[t[0]] = [val] #make a list with all the floats
                else:
                    prefixes[t[0]].append(val) #add to a list with all the floats

        for k,v in prefixes.items():
            inter[k] = list(set(v[0]).intersection(*map(set, v[1:])))[-1] #make it a string, choose most specific one

        return inter
    
    def _find_diversity_topics(self, syll, diversity):
        """
        Gets topics in the area of desired diversity that also have the same subclass as the books in the syllabus.
        Args:
            syll (dict): A dictionary containing a list of ISBNs.
            diversity (dict): A dictionary containing the diversity topics.
        Returns:
            a list of lists. Each list has topics that are in the area of desired diversity.
        
        """

        topics = []
        
        lcas = self._find_lcas(self._get_lccn_for_syllabus(syll), self.detailed_lcc)
        div_dict = {k: v for k,v in diversity.items() if k in lcas.keys()}

        for k,v in lcas.items():
            try:
                entries = div_dict[k]
                #vals.append({k: [entry for entry in entries if v in entry['parents']]}) #gives a lot of topics underneath the parent node
                topics.append([entry['subject'] for entry in entries if v in entry['parents']])

            except:
                pass
        return topics

    def _get_prop_occurrences(self, topics_lst, kind = 'by phrase', top_n = 15): #splits by phrase/full subject
        """
        Takes in a list of topics from either find_diversity-topics or lookup_meaning.
        Args:
            topics_lst (list): A list of topics to analyze.
            kind (str): 'by phrase' or 'by words'. Default is 'by phrase'.
            top_n (int): The number of top topics to return.
        Returns:
            a dictionary with the key as the topic and the value as the proportion of occurrences in the top_n
        """
        
        # nltk.download('stopwords') #to remove uninformative words
        stop_words = set(stopwords.words('english'))
        lcc_stop = open("../backend/library_of_congress/lcc_stop_words.txt", "r").read().split("\n")
        
        all_tags = []

        if type(topics_lst[0]) == str:
            topics_lst = [topics_lst] #kinda nasty srry, but its how it functions

        if kind == 'by phrase':
            for i in topics_lst:
                tags = '. '.join(i).split('. ')
                tags = [x.lower().split('.')[0] for x in tags]
                tags = [x for x in tags if not any(sub in x for sub in lcc_stop)] #get rid of common but uninformative loc terms
                tags = [' '.join([word for word in x.split(' ') if word not in stop_words]) for x in tags] #keep the words not in stop words
                tags = [x.lstrip().rstrip() for x in tags] #remove leading and trailing ws

                all_tags += tags

        else: #if by words
            for i in topics_lst:
                tags = ' '.join(i).split() #split by words
                tags = [x.lower().split('.')[0].split(',')[0].split(')')[0] for x in tags] #pruning for commas, periods, and parenthesis
                tags = [x.lstrip().rstrip() for x in tags] 
                tags = [x for x in tags if x not in lcc_stop or x not in stop_words] #words that I don't want are appearing for some reason
                print(tags)

                all_tags += tags

        #make proportions
        prop = Counter(all_tags) 
        prop = dict(prop.most_common(top_n))
        total = sum(prop.values())
        prop = {k: v/total for k, v in prop.items()}
        return prop
    
    def _search_subjects(self, lcc, topics = [], discipline_tags = [], diversity_tags = [], field = 'subject', limit = 1, exact_string_matching = False):
        """
        Queries the Open Library API for books with subjects in a specific LCC code.
        Args:
            lcc (str): The LCC code to search for.
            topics (list): A list of topics to search for.
            discipline_tags (list): A list of discipline tags to search for.
            diversity_tags (list): A list of diversity tags to search for.
            field (str): The field(s) to return in the response.
            limit (int): The maximum number of results to return.
            exact_string_matching (bool): Whether to use exact string matching. Default is False.
        Returns:
            fields listed by user. Default is subject.
        """
        
        if type(topics) == str:
            # time.sleep(2) #being polite
            response = requests.get(f'https://openlibrary.org/search.json?q=lcc:{lcc}&subject={topics}&fields={field}&limit={limit}').json()

            if response['docs']: #falsy
                return response['docs']
            else:
                return '' #nothing returned

        elif topics:    
            q = f'https://openlibrary.org/search.json?q=lcc:{lcc}&fields={field}&limit={limit}&subject:'

            if exact_string_matching: #for cases where a single word is used
                topics = list(map(lambda x: f"\"{x}\"", topics)) #exact string matching

                
            topics = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), topics)) #encode tags

            topics = '+OR+:'.join(topics) #comma (,) and pipe (|) are similar AND, not OR for some reason
            #topics = ''.join(list(map(lambda x: f'&subject={x}', topics)))
            #topics = ','.join(topics)

            q += "(" + topics + ")"
            
            print(q)

            # time.sleep(2) #being polite
            response = requests.get(q).json()
            
            if response['docs']: #falsy
                return response['docs']
            else:
                return '' #nothing returned

        elif discipline_tags and diversity_tags:
            #encode URI
            discipline_tags, diversity_tags = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), discipline_tags)), list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), diversity_tags))
            #if this ever throws errors, maybe we need to specify unicode

            if exact_string_matching: #for cases where results by word is used
                discipline_tags, diversity_tags = list(map(lambda x: f"\"{x}\"", discipline_tags)), list(map(lambda x: f"\"{x}\"", diversity_tags))

            #match any of the tags
            str_disc, str_div = '+OR+subject:'.join(discipline_tags), '+OR+subject:'.join(diversity_tags)
            q = f'https://openlibrary.org/search.json?q=lcc:{lcc} AND ((subject:{str_disc}) AND (subject:{str_div}))&fields={field}&limit={limit}'
            
            print(q)
            response = requests.get(q).json()
            
            if bool(response['docs']): #falsy
                return response['docs']
            else:
                return '' #nothing returned
                
        else:
            return None
        
    def raos_entropy(self, all_cats):
        """
        Calculates Rao's Quadratic Entropy for a set of categories.
        Args:
            all_cats (dict): A dictionary with the key as the category and the value as the proportion of occurrences.
        Returns:
            a float representing the Rao's Quadratic Entropy between the discipline area and area of desired diversity.
        """

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

        return rqe/2

    def _get_suggestions(self, lca, syll_topics, diversity_topics):
        """
        (IN PROGRESS) Begin getting suggestions for syllabus using entropy measure of choice.
        """
        suggestions = []
        for k,v in lca.items(): 
            if '-' in v:
                lst = v.split('-')
                lccn_query = '[' + lst[0] + ' TO ' + k + lst[1] + ']'
            suggestions += self._search_subjects(lccn_query, discipline_tags = syll_topics, diversity_tags = diversity_topics, field = 'title,subject,isbn,author_name', limit = 50, exact_string_matching=True)
            suggestions += self._search_subjects(lccn_query, discipline_tags = syll_topics, diversity_tags = diversity_topics, field = 'title,subject,isbn,author_name', limit = 50, exact_string_matching=True)
            #list of author names, ISBNs, titles, subjects
        
        for book in suggestions:
            try:
                #print(next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None))
                book['isbn'] = next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None) #shorten the list of ISBNs, ISBN-13
            except:
                #print(book)
                book.update({'isbn': ''})
            book['subject'] = [x.lower() for x in book['subject']]
            #book['subject'] = [sub for sub in [x for x in book['subject']] if sub in syll_topics or sub in diversity_topics] #not working
            #book['subject'] = [x for x in book['subject'] if any(sub in x for sub in syll_topics) or any(sub in x for sub in diversity_topics)] #if it doesn't contain the tags for whatever reason
            book['subject'] = [sub for sub in syll_topics if any(sub in x for x in book['subject'])]
            book['subject'] += [sub for sub in diversity_topics if any(sub in x for x in book['subject'])]

        return suggestions

    




            
if __name__ == "__main__":
    print("Beginning Syllabi Pipeline")
    sp = SyllabiPipeline()
    print("Syllabi Pipeline Initialized")
    ex1 = 'HV-1568.00000000.B376 2016' #The Minority Body by Elizabeth Barnes
    ex2 = 'DAW1008.00000000.B37 1987' #A guide to Central Europe by Richard Bassett
    
    # print(sp.syllabus)
    # print(sp._split_lcc('DAW1008.00000000.B37 1987'))
    # print("SPlit LCC Test Passed")
    lccn_tup = sp._get_lccn_for_syllabus(sp.syllabus)
    # print(lccn_tup)
    lcas = sp._find_lcas(lccn_tup, sp.detailed_lcc)
    # print(lcas)
    with open('../backend/library_of_congress/lgbtq_lcc.json', 'r') as reader: #everything about lgbtq studies, specifically
        diversity = json.load(reader) #in practice, we call also ONLY load in the relevant subclasses
    # print("Diversity JSON Loaded")
    diversity_subset = {k: v for k,v in diversity.items() if k in lcas.keys()}
    topics = sp._find_diversity_topics(sp.syllabus, diversity_subset)
    # print(topics)
    
    
    prop_div = sp._get_prop_occurrences(topics)
    print(prop_div)
    topics_syll = []

    for i in lccn_tup:
        topics_syll += sp._lookup_meaning(i)

    # print(topics_syll)
    prop_syll = sp._get_prop_occurrences(topics_syll)
    
    lccn_query = f"[HQ1 TO HQ2044]"
    # result = sp._search_subjects(lccn_query, discipline_tags = list(prop_syll.keys()), diversity_tags = list(prop_div.keys()), field = 'title,subject,isbn,author_name', limit = 3, exact_string_matching=True)

    all_cats = {**prop_syll, **prop_div}
    # print(all_cats)
    # rqe = sp.raos_entropy(all_cats)

