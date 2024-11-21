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


class SyllabiPipeline:
    """
    Class for the Syllabi Pipeline. Contains all the functions necessary for the measuring diversity and making recommendations.
    """
    def __init__(self, syllabus_path = '../example_syllabi/TEST Syllabi/test2', diversity_measure = 'raos_entropy'):
    
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_path = os.path.join(self.base_dir, 'library_of_congress/detailed-lcc.json')
        final_merged_path = os.path.join(self.base_dir, 'library_of_congress/final_merged_lcc.json')
    
        with open(final_merged_path) as top_level_file: 
            self.detailed_lcc = json.load(top_level_file) 
            
        self.syllabus = pd.read_csv(syllabus_path)   
        self.diversity_measure = diversity_measure     
        self.diversity_topics = ['gay', 'homosexuality', 'lgbt', 'bisexual', 'lesbian', 'transgender', 'queer', 'homophobia', 'same-sex']
        self.diversity_topics2 = ['Human sexuality. Sex. Sexual orientation.', 'Kinsey, Alfred.', 'Bisexuality. General works.', 'Bisexuality. By region or country, A-Z.', 'Homosexuality. Lesbianism. Periodicals. Serials.', 'Homosexuality. Lesbianism. Congresses.', 'Homosexuality. Lesbianism. Societies.', 'Homosexuality. Lesbianism. Dictionaries.', 'Homosexuality. Lesbianism. Computer networks. Electronic information resources (including the Internet and digital libraries).', 'Gay and lesbian studies.', 'Homosexuality. Lesbianism. Biography (Collective).', 'Homosexuality. Lesbianism. Travel.', 'Homosexuality. Lesbianism. Gay parents.', 'Lesbians. Biography. Collective.', 'Lesbians. Biography. Individual, A-Z.', 'Lesbians. General works.', 'Lesbians. Sex instruction.', 'Lesbian mothers.', 'Middle-aged lesbians. Older lesbians.', 'Lesbians. By region or country, A-Z.', 'Gay men. Biography. Collective.', 'Gay men. Biography. Individual, A-Z.', 'Kameny, Frank.', 'Gay men. General works.', 'Gay men. Sex instruction.', 'Gay fathers.', 'Middle-aged gay men. Older gay men.', 'Gay men. By region or country, A-Z.', 'Homosexuality. Lesbianism. General works.', 'Homosexuality. Lesbianism. Juvenile works.', 'Special classes of gay people, A-Z.', 'Special classes of gay people. African Americans.', 'Special classes of gay people. Older gays.', 'Special classes of gay people. Youth.', 'Homosexuality. Lesbianism. By region or country, A-Z.', 'Same-sex relationships. General works.', 'Same-sex relationships. By region or country, A-Z', 'Homophobia. Heterosexism. General works.', 'Homophobia. Heterosexism. By region or country, A-Z.', 'Gay rights movement. Gay liberation movement. Homophile movement. General works.', 'Gay rights movement. Gay liberation movement. Homophile movement. By region or country, A-Z.', 'Gay conservatives.', 'Gay press publications. General works.', 'Gay press publications. By region or country, A-Z', 'Gay and lesbian culture. General works.', 'Gay and lesbian culture. Special topics, A-Z.', 'Gay and lesbian culture. Bathhouses. Saunas. Steam baths.', 'Gay and lesbian culture. Bears.', 'Gay and lesbian culture. Gay pride parades.', 'Gay and lesbian culture. Handkerchief codes.', 'Gay and lesbian culture. Online chat groups.', 'Transvestism. Biography. Collective.', 'Transvestism. Biography. Individual, A-Z.', 'Transvestism. General works.', 'Transvestism. By region or country, A-Z', 'Transsexualism. Biography. Collective.', 'Transsexualism. Biography. Individual, A-Z.', 'Jorgensen, Christine.', 'Transsexualism. General works.', 'Transsexualism. By region or country, A-Z.', 'Parents of gay men or lesbians.', 'Children of gay parents.', 'Same-sex divorce. Gay divorce.', 'Same-sex marriage. General works.', 'Same-sex marriage. By region or country, A-Z.', 'The family. Marriage. Women. Bisexuality in marriage.', 'Developmental psychology. Child psychology. Special topics. Homophobia.']
        self.diversity_topics2 = self._clean_tags(self.diversity_topics2, kind = 'by word')
        self.prop_diversity = self._get_prop_occurrences(self.diversity_topics2, 'by word', top_n = 5)
        self.syllabus_topics = self._get_tags_for_syllabus()
        self.prop_discipline = self._get_prop_occurrences(self.syllabus_topics, 'by word', top_n = 10)

        
 
        if diversity_measure == 'raos_entropy':
            self.diversity_score = self.raos_entropy(self.prop_diversity, self.prop_discipline)

        elif diversity_measure == 'jaccard_distance':
            self.diversity_score = self.jaccard_distance(self._clean_topics(self.syllabus_topics, 'by words'), self._clean_topics(self.diversity_topics2, 'by words'))


            
            
                        
            
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
                print(tags)

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
            
            print("Querying Open Library " + q)
            response = requests.get(q, timeout=5).json()
            
            if response['docs']: 
                return response['docs']
            else:
                return '' 

        elif discipline_tags and diversity_tags: 
            
            str_div =  ' OR '.join(diversity_tags)
            url = 'https://openlibrary.org/search.json'
            params = {
                'q': f'lcc:L* AND subject:({str_div}) AND language:("eng")', 
                'fields': 'author_name,title,isbn,subject,lcc', 
                'limit': 20
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
        print("Proportions of Discipline:", prop_discipline, "\n Proportions of Diversity:", prop_diversity)

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
    
        for book in suggestions:
            try:
                book['isbn'] = next((i for i in book['isbn'] if re.match(r'^(979|978)\d{10}$', i)), None) #shorten the list of ISBNs, ISBN-13
            except:
                book.update({'isbn': ''})
            book['subject'] = [x.lower() for x in book['subject']]
            book['author_name'] = ', '.join(book['author_name'])
            


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
        
            new_all_topics = self.syllabus_topics + [book['subject']]
            prop_new_syllabus = self._get_prop_occurrences(new_all_topics, 'by phrase')
    

            if self.diversity_measure == 'raos_entropy':
                delta = self.raos_entropy(self.prop_diversity, prop_new_syllabus) - original_diversity
            elif self.diversity_measure == 'jaccard_score':
                delta = self.jaccard_distance(new_all_topics, self.diversity_topics) - original_diversity
            else:
                delta = 0
                

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
    print("Low Syllabus")
    sp = SyllabiPipeline("../example_syllabi/TEST Syllabi/test1")
    print("low: " + str(sp.diversity_score))
    
    print("Low-Medium Syllabus")
    sp2 = SyllabiPipeline("../example_syllabi/TEST Syllabi/test2")
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
    plt.ylim(0, 1)
    plt.savefig('diversity_scores.png')