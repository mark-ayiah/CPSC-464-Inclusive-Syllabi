import json
import re
import requests
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
from ast import literal_eval
import urllib
from urllib.parse import quote
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import os
# from sentence_transformers import SentenceTransformer #"tensorflow>=1.7.0", tensorflow-hub


class SyllabiPipeline:
    """
    Class for the Syllabi Pipeline. Contains all the functions necessary for the measuring diversity and making recommendations.
    """
    def __init__(self, syllabus_path = '../example_syllabi/TEST Syllabi/test2', diversity_measure = 'raos_entropy'):
    
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_path = os.path.join(self.base_dir, 'library_of_congress/detailed-lcc.json')
        final_merged_path = os.path.join(self.base_dir, 'library_of_congress/final_merged_lcc.json')
    
        with open(final_merged_path) as top_level_file: #open(detailed_path) as detailed_file, 
            self.detailed_lcc = json.load(top_level_file) #using final_merged path instead
            #self.top_level_lcc = json.load(top_level_file)
            
        self.syllabus = pd.read_csv(syllabus_path)   
        self.diversity_measure = diversity_measure     
        # syllabus_lccn = self._get_lccn_for_syllabus(self.syllabus)
        # print(syllabus_lccn)
        # self.lcas = self._find_lcas(syllabus_lccn, self.detailed_lcc)
        
        # print(syllabus_lccn)
        
        self.diversity_topics = ['gay', 'homosexuality', 'lgbt', 'bisexual', 'lesbian', 'transgender', 'queer', 'homophobia', 'same-sex']
        # self.diversity_topics = self._clean_topics(self.diversity_topics)

        self.prop_diversity = self._get_prop_occurrences(self.diversity_topics, 'by phrase')
        
        # self.syllabus_topics = []
        # for i in syllabus_lccn:
        #     self.syllabus_topics += self._lookup_meaning(i)
        # self.syllabus_topics = _clean_topics(self.syllabus_topics)
        
        self.syllabus_topics = self._get_tags_for_syllabus()
        # print(self.syllabus_topics)
        self.prop_discipline = self._get_prop_occurrences(self.syllabus_topics, 'by word')
        # print("PROP DISCIPLINE")
        # print(self.prop_discipline)
        
        #self.all_props = {**self.prop_discipline, **self.prop_diversity} #not good if any of prop_disc.keys() = prop_div.keys()
        
        if diversity_measure == 'raos_entropy':
    
            self.diversity_score = self.raos_entropy(self.prop_diversity, self.prop_discipline)

        elif diversity_measure == 'jaccard_distance':
            self.diversity_score = self.jaccard_distance(self._clean_topics(self.syllabus_topics, 'by words'), self._clean_topics(self.diversity_topics, 'by words'))


            
            
                        
            
    def _get_tags_for_syllabus(self):
        """
        Gets the tags for a syllabus.
        Returns:
            a list of tags for the syllabus.
        """
        topics = []
        for isbn in self.syllabus['isbn']: #get the lccn
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'isbn:{isbn})', 
                'fields': 'subject',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=20).json()
            topic = response['docs'][0]['subject'] if response['docs'] else None
            # topic = result[0]['subject'] if result else None
            if topic is not None:
                topics.append(topic)
        return topics
        
        
    
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
        match = re.match(r'([A-Z]+)(\d+(\.\d+)?)(?:\.(\w+))?', out)
        
        if match:
            subclass = match.group(1)
            number = float(match.group(2))
            specificity = match.group(4) if match.group(4) else None
            if specificity and number % 1 == 0: 
                return (subclass, number, ''.join((subclass, str(int(number)), '.' + specificity)))
            elif specificity: 
                return (subclass, number, ''.join((subclass, str(number),'.' + specificity)))
            else:
                return (subclass, number)
            # return (match.group(1), float(match.group(2)))
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
            d = detailed_lcc[code[0]]
            for i in d:
                if code[2] == i['id']:
                    l.append(i['subject']) # would be helpful to find an exact match first, but here we are.
                    break
                elif code[1] >= i['start'] and code[1] <= i['stop']:
                    l.append(i['subject'])
                else:
                    continue
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

    def _flatten_list(self, nested_list):
        """
        Flattens a list of lists.
        Args:
            l (list): A list of lists.
        Returns:
            a flattened list.
        """
        flat_list = []
        for item in nested_list:
            if type(item) == list:
                flat_list.extend(self._flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list
    
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
            a list of topics that are in the area of desired diversity.
        
        """

        topics = []
        
        lcas = self._find_lcas(self._get_lccn_for_syllabus(syll), self.detailed_lcc)
        div_dict = {k: v for k,v in diversity.items() if k in lcas.keys()}

        for k,v in lcas.items():
            try:
                entries = div_dict[k]
                # print("entries" + entries)
                #vals.append({k: [entry for entry in entries if v in entry['parents']]}) #gives a lot of topics underneath the parent node
                topics.extend([entry['subject'] for entry in entries if v in entry['parents']])

            except:
                pass
        return topics

    def _clean_topics(self, topics_lst, kind = 'by phrase'):
            
        # nltk.download('stopwords') #to remove uninformative words
        stop_words = set(stopwords.words('english'))
        stop_words_path = os.path.join(self.base_dir, 'library_of_congress/lcc_stop_words.txt') 

        lcc_stop = open(stop_words_path, "r").read().split("\n")
        
        all_tags = []

        if type(topics_lst[0]) == str:
            topics_lst = [topics_lst] #kinda nasty srry, but its how it functions

        if kind == 'by phrase':
            for i in topics_lst:
                tags = '. '.join(i).split('. ')
                tags = [x.lower().split('.')[0] for x in tags]
                tags = [x for x in tags if not any(sub in x for sub in lcc_stop)] #get rid of common but uninformative loc terms
                tags = [x for x in tags if not any(sub in x for sub in stop_words)] #keep the words not in stop words
                tags = [x.lstrip().rstrip() for x in tags] #remove leading and trailing ws

                all_tags += tags

        else: #if by words
            for i in topics_lst:
                tags = ' '.join(i).split() #split by words
                tags = [x.lower().split('.')[0].split(',')[0].split(')')[0].split('(')[0] for x in tags] #pruning for commas, periods, and parenthesis
                tags = [x.lstrip().rstrip() for x in tags] 
                tags = [x for x in tags if x not in lcc_stop or x not in stop_words] #words that I don't want are appearing for some reason
                print(tags)

                all_tags += tags

        return tags
    
    def _clean_tags(self, tag_list):
        """
        Cleans the tags by removing common stop words and Library of Congress stop words.
        Args:
            tags (list): A list of tags to clean.
        Returns:
            a list of cleaned tags.
        """
        stop_words = set(stopwords.words('english'))
        stop_words_path = os.path.join(self.base_dir, 'library_of_congress/lcc_stop_words.txt') 

        lcc_stop = open(stop_words_path, "r").read().split("\n")
        cleaned_tags = []
        for i in tag_list:
            # print(tag)
            tags = i.split()
            tags = [x.lower() for x in tags]
            tags = [re.sub(r'[^\w\s]', '', tag) for tag in tags]
            tags = [x for x in tags if x not in lcc_stop]
            tags = [x for x in tags if x not in stop_words]
    
        
            tags = ["lesbian" if "lesb" in tag else tag for tag in tags]
            tags = ["gay" if "gay" in tag else tag for tag in tags]
            
            cleaned_tags += tags
            
        cleaned_tags = [tag for tag in cleaned_tags if tag]
        return cleaned_tags

    def _get_prop_occurrences(self, topics_lst, kind, top_n = 40): #splits by phrase/full subject
        """
        Takes in a list of topics from either find_diversity-topics or lookup_meaning.
        Args:
            topics_lst (list): A list of topics to analyze.
            kind (str): 'by phrase' or 'by words'. Default is 'by phrase'.
            top_n (int): The number of top topics to return.
        Returns:
            a dictionary with the key as the topic and the value as the proportion of occurrences in the top_n
        """
        # print(f"original topics: {topics_lst}") 
        #make proportions
        all_tags = self._clean_tags(self._flatten_list(topics_lst))
        # print(f"cleaned tags tags: {all_tags}")
        prop = Counter(all_tags) 
        prop = dict(prop.most_common(top_n))
        total = sum(prop.values())
        prop = {k: v/total for k, v in prop.items()}
        return prop
    
    def _search_subjects(self, lcc, topics = None, discipline_tags = None, diversity_tags = None, field = 'subject', limit = 1, exact_string_matching = False):
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
            
            
            # print(q)
            
            
            # time.sleep(2) #being polite
            print("Querying Open Library " + q)
            response = requests.get(q, timeout=5).json()
            
            if response['docs']: #falsy
                return response['docs']
            else:
                return '' #nothing returned

        elif discipline_tags and diversity_tags: #can do you do and on nonetypes?
            #encode URI

            # discipline_tags, diversity_tags = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), discipline_tags)), list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), diversity_tags))
            #if this ever throws errors, maybe we need to specify unicode

            # if exact_string_matching: #for cases where results by word is used
            #     discipline_tags, diversity_tags = list(map(lambda x: f"\"{x}\"", discipline_tags)), list(map(lambda x: f"\"{x}\"", diversity_tags))

            # str_disc = ' OR '.join(discipline_tags[0:5]) 
            
            str_div =  ' OR '.join(diversity_tags)
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'lcc:L* AND subject:({str_div}) AND language:("eng")', 
                'fields': 'author_name,title,isbn,subject,lcc', 
                'limit': 20
                }
            # print(requests.get(url, params=params).url)
            response = requests.get(url, params=params, timeout=30).json()
            # data = response.json()
    
            # str_disc, str_div = '+OR+:'.join(discipline_tags), '+OR+:'.join(diversity_tags)
        
            # q = f'https://openlibrary.org/search.json?q=lcc:{lcc} AND ((subject:{str_disc}) AND (subject:{str_div}))&fields={field}&limit={limit}'
            
            # print(q)
            # response = requests.get(q).json()
            
            if response['docs']: #falsy
                return response['docs']
            else:
                return '' #nothing returned
                
        else:
            return None
        
    def raos_entropy(self, prop_diversity, prop_discipline):
        """
        Calculates Rao's Quadratic Entropy for a set of categories.
        Requires definition of prop_diversity and prop_discipline.
        (Dictionaries with the key as the category and the value as the proportion of occurrences.)
        Returns:
            a float representing the Rao's Quadratic Entropy between the discipline area and area of desired diversity.
        """
        
        model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(model_url)

        # Calculate pairwise cosine distances between topics
        tags = list(prop_diversity) + list(prop_discipline)
        embeddings = model(tags) #needs to be embedded over one space
        distance_matrix = np.inner(embeddings[:len(prop_diversity)], embeddings[len(prop_diversity):]) #cosine sim

        # rao's entropy
        rqe = 0.0

        for i, cat_i in enumerate(list(prop_diversity)):
            for j, cat_j in enumerate(list(prop_discipline)):
                p_i = prop_diversity.get(cat_i, 0) # Probability for category i (fall through if 0)
                p_j = prop_discipline.get(cat_j, 0)
                #print(p_i, p_j)
                # cosine distance (1 - cosine similarity)
                distance = distance_matrix[i, j]
                #print(distance)
                
                rqe += p_i * p_j * distance

        return rqe

    def jaccard_distance(self, disc_lst, div_lst):
        """
        Jaccard distance is 1 - Jaccard similarity, the proportion of words that are the same between two sets.
        Here, the sets are the key words in the syllabus (list) and the key words of the area of desired diversity (list).
        Lower distance equals more diversity.

        Returns:
            a float representing the Jaccard Distance between the discipline area and area of desired diversity.    
        """

        jd = len(set(disc_lst).intersection(set(div_lst)))/len(set(disc_lst).union(set(div_lst)))

        return 1 - jd

    def _get_suggestions(self, syll_topics, diversity_topics):
        """
        Get potential book suggestions.
        Args:
            lca (dict): A dictionary with the key as the LCC class and the value as the LCA range.
            syll_topics (list): A list of topics in the syllabus.
            diversity_topics (list): A list of topics in the area of desired diversity.
        Returns:
            a list of book suggestions.
        """
        
        suggestions = []
        
        suggestions += self._search_subjects('', discipline_tags = syll_topics, diversity_tags = diversity_topics, field = 'title,subject,lcc,isbn,author_name', limit = 20, exact_string_matching=True)
        # suggestions = []
        # for k,v in lca.items(): 
        #     if '-' in v:
        #         lst = v.split('-')
        #         lccn_query = '[' + lst[0] + ' TO ' + k + lst[1] + ']'
        #     suggestions += self._search_subjects(lccn_query, discipline_tags = syll_topics, diversity_tags = diversity_topics, field = 'title,subject,isbn,author_name', limit = 50, exact_string_matching=True)
        #     suggestions += self._search_subjects(lccn_query, discipline_tags = syll_topics, diversity_tags = diversity_topics, field = 'title,subject,isbn,author_name', limit = 50, exact_string_matching=True)
        #     #list of author names, ISBNs, titles, subjects
        # valid_suggestions = []
        for book in suggestions:
            try:
                #print(next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None))
                book['isbn'] = next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None) #shorten the list of ISBNs, ISBN-13
            except:
                #print(book)
                book.update({'isbn': ''})
            book['subject'] = [x.lower() for x in book['subject']]
            # #book['subject'] = [sub for sub in [x for x in book['subject']] if sub in syll_topics or sub in diversity_topics] #not working
            # book['subject'] = [x for x in book['subject'] if x in syll_topics or any(sub in x for sub in diversity_topics)] #if it doesn't contain the tags for whatever reason
            # book['subject'] = [sub for sub in self.syllabus_topics if any(sub in x for x in book['subject'] or any(x in sub for sub in diversity_topics))]
            # book['subject'] += [sub for sub in diversity_topics if any(sub in x for x in book['subject'])]
            book['author_name'] = ', '.join(book['author_name'])
            
            # isbn_dict = book['isbn']
            # lccn = self._get_lccn_for_syllabus(pd.DataFrame({'isbn': [isbn_dict]}))
            # if not lccn or lccn[0] == None:
            #     continue
            # lccn = book['lcc'][0]
            # book['topic'] = lookup_meaning(lccn) #instead of directly using subjects, check it against how we define topics? potentially unnec computation
            # if book['topic']:
            #     valid_suggestions.append(book)

        return suggestions
    
    def _prune_suggestions(self, suggestions, n = 5):
        """
        Prunes the list of suggestions to remove duplicates.
        Args:
            suggestions (list): A list of suggestions.
            n (int): The number of suggestions to return.
        Returns:
            a list of pruned suggestions.
        """
        original_diversity = self.diversity_score
        max_decrease = 0.0
        best_book = None
        
        pruned = []
      
        entropy_list = []

        for book in suggestions:
            # print(book)
            # new_all_topics = self.syllabus_topics + self._clean_topics(book['topic'])
            # print(book)
            # print("adding topics: ", book['subject'])
            new_all_topics = self.syllabus_topics + [book['subject']]
            # print(f"new all topics: {new_all_topics}")
            prop_new_syllabus = self._get_prop_occurrences(new_all_topics, 'by phrase')
    
            # print(f"prop new syllabus: {prop_new_syllabus}")
            # print(f"prop discipline: {self.prop_discipline}")
            # print(f"prop diversity: {self.prop_diversity}")
            # print(new_categories)
            # print(f"old syllabus topics: {self.syllabus_topics}, new syllabus topics: {new_categories}, old syllabus prop: {self.prop_discipline}, new syllabus prop: {prop_new_syllabus}")
            #new_props = {**prop_new_syllabus, **self.prop_diversity}

            #can change later
            if self.diversity_measure == 'raos_entropy':
                delta = self.raos_entropy(self.prop_diversity, prop_new_syllabus) - original_diversity
                # print(f"original diversity: {original_diversity}, new diversity: {self.raos_entropy(self.prop_diversity, prop_new_syllabus)}")
            elif self.diversity_measure == 'jaccard_score':
                delta = self.jaccard_distance(new_all_topics, self.diversity_topics) - original_diversity
            else:
                delta = 0
                
            # print("DELTA: " + str(delta))

            if delta > 0:
                entropy_list.append((book, delta))
                
        entropy_list.sort(key = lambda x: x[1], reverse = True)
        
        top_n_books = [book for book, _ in entropy_list[:n]]
                
        return top_n_books
    
    
    def recommend_books(self):
        """
        Recommends books based on the syllabus.
        Returns:
            a list of book suggestions.
        """
        
        suggestions = self._get_suggestions(self.syllabus_topics, self.diversity_topics)
        pruned = self._prune_suggestions(suggestions)
        print(pruned)
        return pruned
        

    




            
if __name__ == "__main__":
    # print("Beginning Syllabi Pipeline")
    print("Low Syllabus")
    sp = SyllabiPipeline("../example_syllabi/TEST Syllabi/test1")
    print("low: " + str(sp.diversity_score))
    
    # jaccard = SyllabiPipeline("../example_syllabi/TEST Syllabi/test1", 'jaccard_distance')
    # print("jaccard: " + str(jaccard.diversity_score))
    # print(sp.syllabus_topics)
    # print(sp.syllabus)
    # sp.recommend_books()
    print("Low-Medium Syllabus")
    sp2 = SyllabiPipeline("../example_syllabi/TEST Syllabi/test2", 'jaccard_distance')
    print("low-medium: " + str(sp2.diversity_score))

    print("Medium-High Syllabus")
    sp3 = SyllabiPipeline("../example_syllabi/TEST Syllabi/test3")
    print("medium-high: " + str(sp3.diversity_score))

    
    
    print("High Syllabus")
    sp4 = SyllabiPipeline("../example_syllabi/TEST Syllabi/test4")
    print("high: " + str(sp4.diversity_score))


    diversity_scores = {
    'Test 1 (Low)': sp.diversity_score,
    'Test 2 (Low-Medium)': sp2.diversity_score,
    'Test 3 (Medium-High)': sp3.diversity_score,
    'Test 4 (High)': sp4.diversity_score
}

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(diversity_scores.keys(), diversity_scores.values(), color=['red', 'green', 'blue', 'orange'])

    plt.title('Rao\'s Entropy Score Results')
    plt.xlabel('Syllabi')
    plt.ylabel('Diversity Score')
    plt.ylim(0, 1)  # Add some space at the top
    plt.savefig('diversity_scores.png')