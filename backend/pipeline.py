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
import numpy as np
import tensorflow_hub as hub
import random



import os



class SyllabiPipeline:
    """
    Class for the Syllabi Pipeline. Contains all the functions necessary for the measuring diversity and making recommendations.
    """
    def __init__(self, syllabus_path = '../example_syllabi/test2.csv', diversity_measure = 'raos_entropy'):
    
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_path = os.path.join(self.base_dir, 'library_of_congress/detailed-lcc.json')
        final_merged_path = os.path.join(self.base_dir, 'library_of_congress/final_merged_lcc.json')
    
        with open(final_merged_path) as top_level_file: 
            self.detailed_lcc = json.load(top_level_file) 
            
        self.syllabus = pd.read_csv(syllabus_path, dtype={'isbn': str})   

        self.diversity_measure = diversity_measure     
        self.diversity_topics = ['gay', 'homosexuality', 'lgbt', 'bisexual', 'lesbian', 'transgender', 'queer', 'homophobia', 'same-sex']
        self.diversity_topics2 = ['Human sexuality. Sex. Sexual orientation.', 'Kinsey, Alfred.', 'Bisexuality. General works.', 'Bisexuality. By region or country, A-Z.', 'Homosexuality. Lesbianism. Periodicals. Serials.', 'Homosexuality. Lesbianism. Congresses.', 'Homosexuality. Lesbianism. Societies.', 'Homosexuality. Lesbianism. Dictionaries.', 'Homosexuality. Lesbianism. Computer networks. Electronic information resources (including the Internet and digital libraries).', 'Gay and lesbian studies.', 'Homosexuality. Lesbianism. Biography (Collective).', 'Homosexuality. Lesbianism. Travel.', 'Homosexuality. Lesbianism. Gay parents.', 'Lesbians. Biography. Collective.', 'Lesbians. Biography. Individual, A-Z.', 'Lesbians. General works.', 'Lesbians. Sex instruction.', 'Lesbian mothers.', 'Middle-aged lesbians. Older lesbians.', 'Lesbians. By region or country, A-Z.', 'Gay men. Biography. Collective.', 'Gay men. Biography. Individual, A-Z.', 'Kameny, Frank.', 'Gay men. General works.', 'Gay men. Sex instruction.', 'Gay fathers.', 'Middle-aged gay men. Older gay men.', 'Gay men. By region or country, A-Z.', 'Homosexuality. Lesbianism. General works.', 'Homosexuality. Lesbianism. Juvenile works.', 'Special classes of gay people, A-Z.', 'Special classes of gay people. African Americans.', 'Special classes of gay people. Older gays.', 'Special classes of gay people. Youth.', 'Homosexuality. Lesbianism. By region or country, A-Z.', 'Same-sex relationships. General works.', 'Same-sex relationships. By region or country, A-Z', 'Homophobia. Heterosexism. General works.', 'Homophobia. Heterosexism. By region or country, A-Z.', 'Gay rights movement. Gay liberation movement. Homophile movement. General works.', 'Gay rights movement. Gay liberation movement. Homophile movement. By region or country, A-Z.', 'Gay conservatives.', 'Gay press publications. General works.', 'Gay press publications. By region or country, A-Z', 'Gay and lesbian culture. General works.', 'Gay and lesbian culture. Special topics, A-Z.', 'Gay and lesbian culture. Bathhouses. Saunas. Steam baths.', 'Gay and lesbian culture. Bears.', 'Gay and lesbian culture. Gay pride parades.', 'Gay and lesbian culture. Handkerchief codes.', 'Gay and lesbian culture. Online chat groups.', 'Transvestism. Biography. Collective.', 'Transvestism. Biography. Individual, A-Z.', 'Transvestism. General works.', 'Transvestism. By region or country, A-Z', 'Transsexualism. Biography. Collective.', 'Transsexualism. Biography. Individual, A-Z.', 'Jorgensen, Christine.', 'Transsexualism. General works.', 'Transsexualism. By region or country, A-Z.', 'Parents of gay men or lesbians.', 'Children of gay parents.', 'Same-sex divorce. Gay divorce.', 'Same-sex marriage. General works.', 'Same-sex marriage. By region or country, A-Z.', 'The family. Marriage. Women. Bisexuality in marriage.', 'Developmental psychology. Child psychology. Special topics. Homophobia.']
        self.diversity_topics2 = self._clean_tags(self.diversity_topics2, kind = 'by word')

        self.prop_diversity = self._get_prop_occurrences(self.diversity_topics2, 'by word', top_n = 5)
        self.syllabus_topics = self._get_tags_for_syllabus()
        self.prop_discipline = self._get_prop_occurrences(self.syllabus_topics, 'by word', top_n = 10)

        self.syllabus_books = self._get_books_for_syllabus()
        
 
        if diversity_measure == 'raos_entropy':
            self.diversity_score = self.raos_entropy(self.prop_diversity, self.prop_discipline)

        elif diversity_measure == 'jaccard_distance':
            #self.diversity_score = self.jaccard_distance(self._clean_tags(self.syllabus_topics, 'by word'), self._clean_tags(self.diversity_topics2, 'by word'))
            self.diversity_score = self.jaccard_distance(self.prop_diversity.keys(), self.prop_discipline.keys())
        
        elif diversity_measure == 'relevance_proportion':
            self.diversity_score = self.relevance_proportion(self.syllabus_books)
            
        elif diversity_measure == 'overlap_proportion':
            self.diversity_score = self.overlap_proportion()
            
        self.score = 0.0
        self.delta = 0.0
        self.suggestions = self._get_suggestions(self.syllabus_topics, self.diversity_topics)            
            
    def _get_books_for_syllabus(self):
        """
        Gets the books for a syllabus.
        Returns:
            a list of books for the syllabus.
        """
        books = []

        count = 0
        for isbn in self.syllabus['isbn']:
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'isbn:{isbn})', 
                'fields': 'author_name,title,isbn,subject,lcc',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=20).json()
            book = response['docs'][0] if response['docs'] else None
            if book is not None:
                books.append(book)
        return books
            
    def _get_tags_for_syllabus(self):
        """
        Gets the tags for a syllabus.
        Returns:
            a list of tags for the syllabus.
        """
        topics = []
        for isbn in self.syllabus['isbn']: 
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'isbn:{isbn})', 
                'fields': 'subject',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=20).json()
            topic = response['docs'][0]['subject'] if response['docs'] else None
            if topic is not None:
                topics.append(topic)
        return topics

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
    
    
    def _clean_topics(self, topics_lst, kind = 'by phrase'):

        """
        BROKEN
        Cleans the topics by removing common stop words and Library of Congress stop words.
        Args:
            tags (list): A list of tags to clean.
            kind (string): whether to keep tags as phrases or to only look at them as individual words
        Returns:
            a list of cleaned tags.
        """
            
        stop_words = set(stopwords.words('english'))
        stop_words_path = os.path.join(self.base_dir, 'library_of_congress/lcc_stop_words.txt') 

        lcc_stop = open(stop_words_path, "r").read().split("\n")
        
        all_tags = []

        if type(topics_lst[0]) == str:
            topics_lst = [topics_lst] 

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
                # print(tags)

                all_tags += tags

        return tags
    
    def _clean_tags(self, tag_list, kind = 'by word'):
        """
        Cleans the tags by removing common stop words and Library of Congress stop words.
        Args:
            tags (list): A list of tags to clean.
            kind (string): whether to keep tags as phrases or to only look at them as individual words
        Returns:
            a list of cleaned tags.
        """
        stop_words = set(stopwords.words('english'))
        stop_words_path = os.path.join(self.base_dir, 'library_of_congress/lcc_stop_words.txt') 

        lcc_stop = open(stop_words_path, "r").read().split("\n")
        tag_list = self._flatten_list(tag_list)

        cleaned_tags = []
        if kind == 'by word': 
            for i in tag_list:
                tags = i.split()
                tags = [x.lower() for x in tags]
                tags = [re.sub(r'[^\w\s]', '', tag) for tag in tags]
                tags = [x for x in tags if x not in lcc_stop]
                tags = [x for x in tags if x not in stop_words]
        
            
                tags = ["lesbian" if "lesb" in tag else tag for tag in tags]
                tags = ["gay" if "gay" in tag else tag for tag in tags]
                tags = ["transgender" if "trans" in tag else tag for tag in tags]
                
                cleaned_tags += tags
        elif kind == 'by phrase':
            for i in tag_list:
                tags = i.split(". ")
                tags = [x.lower() for x in tags]
                tags = [re.sub(r'[^\w\s]', '', tag) for tag in tags]
                tags = [x for x in tags if x not in lcc_stop]
                tags = [x for x in tags if x not in stop_words]
        
            
                tags = ["lesbian" if "lesb" in tag else tag for tag in tags]
                tags = ["gay" if "gay" in tag else tag for tag in tags]
                tags = ["transgender" if "trans" in tag else tag for tag in tags]
                
                cleaned_tags += tags
        else:
            cleaned_tags = []
            
        cleaned_tags = [tag for tag in cleaned_tags if tag]
        return cleaned_tags

    def _get_prop_occurrences(self, topics_lst, kind, top_n = 40): 
        """
        Takes in a list of topics from either find_diversity-topics or lookup_meaning.
        Args:
            topics_lst (list): A list of topics to analyze.
            kind (str): 'by phrase' or 'by words'. Default is 'by phrase'.
            top_n (int): The number of top topics to return.
        Returns:
            a dictionary with the key as the topic and the value as the proportion of occurrences in the top_n
        """
 
        all_tags = self._clean_tags(self._flatten_list(topics_lst))
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
            response = requests.get(f'https://openlibrary.org/search.json?q=lcc:{lcc}&subject={topics}&fields={field}&limit={limit}').json()

            if response['docs']: 
                return response['docs']
            else:
                return '' 

        elif topics:
    
            q = f'https://openlibrary.org/search.json?q=lcc:{lcc}&fields={field}&limit={limit}&subject:'
            if exact_string_matching:
                topics = list(map(lambda x: f"\"{x}\"", topics)) 

                
            topics = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), topics)) 

            topics = '+OR+:'.join(topics)

            q += "(" + topics + ")"
            
            # print("Querying Open Library " + q)
            response = requests.get(q, timeout=5).json()
            
            if response['docs']: 
                return response['docs']
            else:
                return '' 

        elif discipline_tags and diversity_tags: 
            
            str_div =  ' OR '.join(diversity_tags)
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'lcc:LB* AND subject:({str_div}) AND language:("eng")', 
                'fields': 'author_name,title,isbn,subject,lcc', 
                'limit': 40,
                'sort': 'random'
                }
        
            response = requests.get(url, params=params, timeout=30).json()
      
            
            if response['docs']: 
                return response['docs']
            else:
                return ''
                
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
        # print("Proportions of Discipline:", prop_discipline, "\n Proportions of Diversity:", prop_diversity)

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
                # cosine distance (1 - cosine similarity)
                distance = distance_matrix[i, j]

                #print("Distance between:", cat_i, " and ", cat_j, ": ", distance)
                
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
        # print(disc_lst, div_lst)

        jd = len(set(disc_lst).intersection(set(div_lst)))/len(set(disc_lst).union(set(div_lst)))

        return 1 - jd
    
    def relevance_proportion(self, books):
        """
        Calculates the proportion of books in syllabus that have at least one subject in the diversity topics.
        
        Returns:
            a float representing the proportion of books in the syllabus that have at least one subject in the diversity topics.
        """
        count = 0
        for book in books:
            if any(sub.lower() in subject.lower() for sub in self.prop_diversity.keys() for subject in book['subject']):
                count += 1
        return count/len(books)
        
        
        
    def overlap_proportion(self, other_subjects = []):
        """
        Calculates the proportion of unique subjects in the syllabus compared to the total number of subjects.
        Args:
        [Optional] other_subjects (list): Other subjects to consider
        """
        topics = self.syllabus_topics + other_subjects

        #topics = self.syllabus_topics
        topics = self._flatten_list(topics)
        
        total = len(topics)
        topics_set = set(topics)
        unique = len(topics_set) 
        
        return unique/total
        
        

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
    
        for book in suggestions:
            try:
                book['isbn'] = next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None) #shorten the list of ISBNs, ISBN-13
            except:
                book.update({'isbn': ''})
            book['subject'] = [x.lower() for x in book['subject']]
            book['author_name'] = ', '.join(book['author_name'])
            
        for book in suggestions:
            if book['title'] in [b['title'] for b in self.syllabus_books]:
                suggestions.remove(book)
            
        return suggestions
    
    def _prune_suggestions(self, suggest, n = 3):
        """
        Prunes the list of suggestions to choose the best n suggestions.
        The first book is chosen based on which one improves diversity of the syllabus the most.
        The remaining books are chosen based on which improves diversity of the new set of books the most.
        Args:
            suggest (list): A list of suggestions.
            n (int): The number of suggestions to return.
        Returns:
            a list of pruned suggestions.
        """
        original_diversity = self.diversity_score
      
        L = []
        #suggest = self.suggestions
        
        # first book based on syllabus
        best_book = self._find_best_book(self.suggestions, self.syllabus_topics, self.diversity_measure, self.diversity_score, self.prop_diversity, self.diversity_topics2)
        L.append(best_book)
        suggest.remove(best_book)

        # remaining books based on new set of books
        set_topics = [L[0]['subject']] #subjects in the first book of the list
        while len(L) < n:
            if len(L) == n - 1:
                last = True
            else:
                last = False
            
            #used to just be original_diversity. if this makes it perform worse, change it back
            best_book = self._find_best_book(self.suggestions, set_topics, self.diversity_measure, self.diversity_score, self.prop_diversity, self.diversity_topics2, last) 
            
            if best_book is None:
                break
            else:
                L.append(best_book) #add the best new book to the list
                suggest.remove(best_book) #remove the book from the dictionary of suggestions
                set_topics += [L[-1]['subject']] #add topics we've already seen to the list
        
        return L
    
    def _find_best_book(self, suggest, current_topics, diversity_measure, original_diversity, prop_diversity=None, diversity_topics=None, last=False):
        """
        Finds the best book, based on how it affects a diversity measure, in a list of suggestions
        Args:
        suggest (dict): A list of book suggestions (as dictionaries)
        current_topics (list): a list of topics in the 
        diversity_measure (str): the metric to use to calculate the best book
        original_diversity (float): the diversity score before making suggestions
        """
        max_improvement = -float('inf')
        best_book = None
        
        for book in suggest:
            
            new_all_topics = current_topics + [book['subject']]
            prop_new_syllabus = self._get_prop_occurrences(new_all_topics, 'by word')

            #There could be a calculation error here

            if self.diversity_measure == 'raos_entropy':
                self.score = self.raos_entropy(prop_diversity, prop_new_syllabus)
                self.delta = self.score - original_diversity
            elif self.diversity_measure == 'jaccard_score':
                self.score = self.jaccard_distance(prop_diversity.keys(), prop_new_syllabus.keys()) #updated to only look at top n
                self.delta = original_diversity - self.score
                #self.rec_delta = -delta #dont make it negative, we will need this for max improvement calculation later on
            elif self.diversity_measure == 'relevance_proportion':
                self.score = self.relevance_proportion([book])
                self.delta = self.score - original_diversity
                #self.rec_delta = delta
            elif self.diversity_measure == 'overlap_proportion':
                self.score = self.overlap_proportion(book['subject'])
                self.delta = original_diversity - self.score
                #self.rec_delta = -delta #dont make it negative, we will need this for max improvement calculation later on
            else:
                self.delta = 0
                
            if self.delta > max_improvement:
                max_improvement = self.delta
                best_book = book
    
        return best_book
    
    
    def recommend_books(self):
        """
        Recommends books based on the syllabus.
        Returns:
            a list of book suggestions.
        """
        
        #suggestions2 = self._get_suggestions(self.syllabus_topics, self.diversity_topics)
        pruned = self._prune_suggestions(self.suggestions)
        return pruned
    
def results(measure):
    """
    Measures and plots diversity score for each syllabus
    """
        
    print("Low Syllabus")
    sp1 = SyllabiPipeline("../example_syllabi/test1.csv", measure)
    print("low: " + str(sp1.diversity_score))
    
    print("Low-Medium Syllabus")
    sp2 = SyllabiPipeline("../example_syllabi/test2.csv", measure)
    print("low-medium: " + str(sp2.diversity_score))

    print("Medium-High Syllabus")
    sp3 = SyllabiPipeline("../example_syllabi/test3.csv", measure)
    print("medium-high: " + str(sp3.diversity_score))

    print("High Syllabus")
    sp4 = SyllabiPipeline("../example_syllabi/test4.csv", measure)
    print("high: " + str(sp4.diversity_score))

    scores = {
    'Test 1 (Low)': sp1.diversity_score,
    'Test 2 (Low-Medium)': sp2.diversity_score,
    'Test 3 (Medium-High)': sp3.diversity_score,
    'Test 4 (High)': sp4.diversity_score
    }

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(scores.keys(), scores.values(), color=['red', 'green', 'blue', 'orange'])
    plt.title(f'{measure.title()} Score Results')
    plt.xlabel('Syllabi')
    plt.ylabel('Diversity Score')
    plt.ylim(0, 1)
    plt.savefig(f'{measure}.png')
    

def rec_results(measure, syll, f): #what is this
    f.write(f"{measure.title()} Recommendations\n")
    sp = SyllabiPipeline(f"../example_syllabi/{syll}.csv")
    f.write(str(sp.recommend_books()))
    # f.write("\n")
    # f.write("--------------------\n")
    
def rec_delta_results(): #what is this
    with open('rec_delta.txt', 'w') as f:
        f.write("Scores Before Recs\n")
        sp = SyllabiPipeline("../example_syllabi/test2.csv", 'raos_entropy')
        f.write(f"raos before recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2.csv", 'jaccard_distance')
        f.write(f"jaccard before recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2.csv", 'relevance_proportion')
        f.write(f"relevance before recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2.csv", 'overlap_proportion')
        f.write(f"overlap before recs: {str(sp.diversity_score)}\n")
        f.write("--------------------\n")
        
        f.write("After Recommending with Rao\n")
        sp = SyllabiPipeline("../example_syllabi/test2 re.csv", 'raos_entropy')
        f.write(f"raos after raos recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 re.csv", 'jaccard_distance')
        f.write(f"jaccard after raos recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 re.csv", 'relevance_proportion')
        f.write(f"relevance after raos recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 re.csv", 'overlap_proportion')
        f.write(f"overlap after raos recs: {str(sp.diversity_score)}\n")
        f.write("--------------------\n")
        
        f.write("After Recommending with Jaccard\n")
        sp = SyllabiPipeline("../example_syllabi/test2 jd.csv", 'raos_entropy')
        f.write(f"raos after jaccard recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 jd.csv", 'jaccard_distance')
        f.write(f"jaccard after jaccard recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 jd.csv", 'relevance_proportion')
        f.write(f"relevance after jaccard recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 jd.csv", 'overlap_proportion')
        f.write(f"overlap after jaccard recs: {str(sp.diversity_score)}\n")
        f.write("--------------------\n")
        
        f.write("After Recommending with Relevance\n")
        sp = SyllabiPipeline("../example_syllabi/test2 rp.csv", 'raos_entropy')
        f.write(f"raos after relevance recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 rp.csv", 'jaccard_distance')
        f.write(f"jaccard after relevance recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 rp.csv", 'relevance_proportion')
        f.write(f"relevance after relevance recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 rp.csv", 'overlap_proportion')
        f.write(f"overlap  after relevance recs: {str(sp.diversity_score)}\n")
        f.write("--------------------\n")
        
        f.write("After Recommending with Overlap\n")
        sp = SyllabiPipeline("../example_syllabi/test2 bp.csv", 'raos_entropy')
        f.write(f"raos after overlap recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 bp.csv", 'jaccard_distance')
        f.write(f"jaccard after overlap recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 bp.csv", 'relevance_proportion')
        f.write(f"relevance after overlap recs: {str(sp.diversity_score)}\n")
        
        sp = SyllabiPipeline("../example_syllabi/test2 bp.csv", 'overlap_proportion')
        f.write(f"overlap before after overlap recs: {str(sp.diversity_score)}\n")
        f.write("--------------------\n")
        
def perturb_data(measure):
    
    sp = SyllabiPipeline("../example_syllabi/test2.csv", measure)
    print("no noise: " + str(sp.diversity_measure))
    
    random_subjects = ['history', 'green', 'america', 'airports', 'cancer', 'biology', 'africa', 'ethnic', 'math', 'ethics', 'politics', 'economics', 'computer science', 'latino', 'perspective', 'hand', 'fish', 'teenager', 'adult', 'geology', 'apartments', 'urban', 'finance', 'adventure', 'mythology', 'technology', 'romance', 'psychology', 'poetry', 'leadership', 'literature', 'social', 'neuroscience', 'fiction', 'fantasy', 'asia', 'abolition', 'library', 'swamp', 'horse', 'bread', 'conservative', 'liberal', 'biography', 'english', 'nonfiction']
    
    no_noise = sp.diversity_score
    
    def perturb_measure(measure, n):
        
        if measure == 'raos_entropy':
            # print(sp.syllabus_topics)
            topics = sp.syllabus_topics + [random.choices(random_subjects, k=n)]
            # print(len(sp._flatten_list(topics)))
            # print(topics)
            sp.prop_discipline = sp._get_prop_occurrences(topics, 'by word', top_n = 10)
            return sp.raos_entropy(sp.prop_diversity, sp.prop_discipline)

        elif measure == 'jaccard_distance':
            topics = sp.syllabus_topics + [random.choices(random_subjects, k=n)]
            return sp.jaccard_distance(sp._clean_tags(topics, 'by word'), sp._clean_tags(sp.diversity_topics2, 'by word'))
            
            
        elif measure == 'relevance_proportion': #this example is a bit shaky
            total = len(sp.syllabus_books)
            num_relevant = sp.relevance_proportion(sp.syllabus_books) * total
            random_relevant = random.randint(0, n) #throwing in random books that aren't relevant to the subject
            return (num_relevant + random_relevant) / (total + n)
            
        elif measure == 'overlap_proportion':
            old_topics = sp.syllabus_topics
            sp.syllabus_topics += random.choices(random_subjects, k=n)
            score = sp.overlap_proportion()
            sp.syllabus_topics = old_topics
            return score
        
    noise_5 = perturb_measure(measure, 5)
    noise_15 = perturb_measure(measure, 10)
    noise_30 =  perturb_measure(measure, 50)


    scores = {
    'No Noise': no_noise,
    '5 Random Subjects': noise_5,
    '10 Random Subjects': noise_15,
    '50 Random Subjects': noise_30
    }

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(scores.keys(), scores.values(), color=['red', 'green', 'blue', 'orange'])
    plt.title(f'{measure.title()} Noise Score Results (Low-Medium Syllabus)')
    plt.xlabel('Level of Noise')
    plt.ylabel('Diversity Score')
    plt.ylim(0, 1)
    plt.savefig(f'{measure}_noise2.png')
    
    
    

            
if __name__ == "__main__":
    
    # Empirical Results!
    #results('raos_entropy')
    #results('jaccard_distance')
    #results('relevance_proportion')
    #results('overlap_proportion')
    

    # Recommendations
    measures = ['raos_entropy', 'jaccard_distance', 'relevance_proportion', 'overlap_proportion']

    with open('../results/recsre.txt', 'w') as f:
        f.write("Recommend Books for low-medium\n")

        for m in measures:
            print(m)
            rec_results(m, 'test2', f)
        
    # Results After Recommendations
    #rec_delta_results()   
    
    
    # Testing Durability
    #perturb_data('raos_entropy')
    #perturb_data('jaccard_distance')
    #perturb_data('relevance_proportion')
    #perturb_data('overlap_proportion')
     

        

    
    
   