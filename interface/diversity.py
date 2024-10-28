# ===============
# Import Statements
# ===============

import urllib
from urllib.parse import quote

# Language Identifier 1
import fasttext
from huggingface_hub import hf_hub_download

# Language Identifier 2
from langid.langid import LanguageIdentifier, model

import requests
import re
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import csv
from os import listdir
import json

# ===============
# Configurations
# ===============

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model_ft = fasttext.load_model(model_path)
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True) #instantiate identifier

#testing Open Library API
# r = requests.get('https://openlibrary.org/search.json?q=subject:("dogs"+OR+"cats")+subject:("Juvenile fiction"+OR+"Juvenile literature")&fields=subject')
# r = r.json()
# subs = [d['subject'] for d in r['docs']] #gets the list, AKA value from k:v in subject:list dictionary
#print(subs[0])

#discipline tags is a list
#diversity tags is a list
#k is the number of items to return

# =================
# Helper Functions 
# =================

#finds results that match ANY of the first list of tags and ANY of the second list of tags
def search_recs(discipline_tags, diversity_tags, k):
    #encode URI
    discipline_tags, diversity_tags = list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), discipline_tags)), list(map(lambda x: urllib.parse.quote(x.encode("utf-8")), diversity_tags))
    #if this ever throws errors, maybe we need to specify unicode

    #exact string matching
    discipline_tags, diversity_tags = list(map(lambda x: f"\"{x}\"", discipline_tags)), list(map(lambda x: f"\"{x}\"", diversity_tags))

    #match any of the tags
    str_disc, str_div = '+OR+'.join(discipline_tags), '+OR+'.join(diversity_tags)

    print(f'https://openlibrary.org/search.json?q=subject:({str_disc})+subject:({str_div})&fields=subject&limit={k}')
    return requests.get(f'https://openlibrary.org/search.json?q=subject:({str_disc})+subject:({str_div})&fields=author_name,title,isbn,subject&limit={k}', timeout=10).json()





#book = get_books(syllabus); takes in a list of ISBNs
def get_tags(books):
    
    tags = []
    for isbn in books:
        data = requests.get(f'https://openlibrary.org/search.json?q=isbn:{isbn}&fields=subject').json()
        
        if 'docs' in data and len(data['docs']) > 0 and 'subject' in data['docs'][0]:
            tags.extend(data['docs'][0]['subject'])
        else:
            print(f"ISBN {isbn} not found")

        
    return tags

#takes in a list
def clean_tags(tags):
    # for idx, l in enumerate(tags): #index, list of lists

    #lowercase
    tags = [s.lower() for s in tags]

    #language identifier
    #We can either keep a tag if both methods AGREE that it is english OR only use one and set a probability threshold for english likelihood
    tags = [s for s in tags if model_ft.predict(s)[0][0] == '__label__eng_Latn'] #if english, using fast text; https://aclanthology.org/E17-2068/
    #if english, using langid
    tags = [s for s in tags if identifier.classify(s)[0] == 'en'] #off a cursory glance, performs better than the fasttext one, but still not as robust as using both; http://www.aclweb.org/anthology/P12-3005

    #remove mentions of "fiction" to prevent stripped pertinent information due to commas later on
    tags = [s.split('in fiction')[0] for s in tags] #remove any mention of 'fiction'
    tags = [s.split(', fiction')[0] for s in tags] #remove any mention of 'fiction'
    tags = [s.split('fiction, ')[0] for s in tags] #remove any mention of 'fiction'

    #clean for extraness
    tags = [s.split(',')[0]  for s in tags] #remove anything after a comma
    tags = [s.split('--')[0]  for s in tags] #remove anything with the --
    tags = [s.split('(')[0]  for s in tags] #remove parenthesis and anything within it
    tags = [s.split('[')[0]  for s in tags] #remove parenthesis and anything within it
    tags = [s.split('{')[0]  for s in tags] #remove parenthesis and anything within it
    tags = [s.split('/')[0]  for s in tags] #look at info before slash
    tags = [s.split('"')[0]  for s in tags] #remove quotes
    tags = [s for s in tags if ":" not in s] #remove anything with parentheses
    tags = [s for s in tags if "reading level" not in s] #remove any mention of reading level

    #remove other uninformative tags
    tags = [s for s in tags if "translations" not in s]
    tags = [s for s in tags if "staff" not in s] #staff picks
    tags = [s for s in tags if "language materials" not in s] #language materials

    #remove dewey system stuff until further notice
    tags = [s for s in tags if not s.isdigit()]

    #ampersand in the tags is causing problems

    #remove whitespace
    tags = [s.strip(' \t\n\r') for s in tags]

    #remove empty string
    tags = [s for s in tags if bool(s) != False]

    #make unique, update list
    # tags[idx] = list(set(tags))

    return list(set(tags)) #list




# =================
# Logic Functions 
# =================



# Function to categorize tags
# def categorize_tags(tags, desired_diversity ='gender representation', overarching_discipline='african studies'):
#     # Load model for embeddings
#     # Load model for embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Desired diversity area and overarching discipline
#     desired_diversity = 'gender representation'
#     overarching_discipline = 'african studies'
#     categorized_tags = []
#     for tag_list in tags:
#         # Generate embeddings for tags, diversity area, and discipline
#         tag_embeddings = model.encode(tag_list)
#         diversity_embedding = model.encode([desired_diversity])
#         discipline_embedding = model.encode([overarching_discipline])
#         cat_tags = {}
#         for i, tag in enumerate(tag_list):
#             # Calculate cosine similarity to both diversity and discipline
#             diversity_sim = cosine_similarity([tag_embeddings[i]], diversity_embedding)[0][0]
#             discipline_sim = cosine_similarity([tag_embeddings[i]], discipline_embedding)[0][0]

#             # Categorize based on higher similarity
#             if (diversity_sim > .25) or (discipline_sim > .15):
#                 if diversity_sim > discipline_sim:
#                     cat_tags[tag] = 'Diversity'
#                 elif discipline_sim > diversity_sim:
#                     cat_tags[tag] = 'Discipline'
#             else:
#                 cat_tags[tag] = 'Neither'
#         categorized_tags.append(cat_tags)

#     return categorized_tags
        

#     return cat_alpha, cat_beta

# # Define Rao's entropy formula
# def raos_entropy(cat_alpha, cat_beta):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     # Generate embeddings for topics
#     alpha_embeddings = model.encode(cat_alpha)
#     beta_embeddings = model.encode(cat_beta)

#     # Full list of syllabus topics
#     syllabus_topics = cat_alpha + cat_beta

#     # Calculate proportions (p_i and p_j)
#     topic_counts = Counter(syllabus_topics)
#     total_topics = len(syllabus_topics)
#     p_alpha = np.array([topic_counts[topic] / total_topics for topic in cat_alpha])
#     p_beta = np.array([topic_counts[topic] / total_topics for topic in cat_beta])
#     entropy = 0.0
#     # Calculate pairwise cosine distances between topics
#     distance_matrix = cosine_distances(alpha_embeddings, beta_embeddings)
    
#     # Sum over all topic pairs
#     for i in range(len(cat_alpha)):
#         for j in range(len(cat_beta)):
#             entropy += p_alpha[i] * p_beta[j] * distance_matrix[i, j]
    
#     return entropy


# def calculate_diversity_metrics(isbns):
#     tags = get_tags(isbns)
#     cleaned_tags = clean_tags(tags)

#     cat_alpha, cat_beta = categorize_tags('gender representation', 'african studies', cleaned_tags)

#     entropy_score = raos_entropy(cat_alpha, cat_beta)

    
#     print(f"Rao's Entropy (Diversity): {entropy}")
#     return entropy_score

def get_suggestions(cat_alpha, cat_beta):
    
    # construct the query
    alpha_tags = ' OR '.join(cat_alpha)
    beta_tags =  ' OR '.join(cat_beta)
    url = 'https://openlibrary.org/search.json'
    params = {
        'q': f'subject:({alpha_tags}) AND subject:({beta_tags}) AND language:("eng")', 
        'fields': 'author_name,title,isbn,subject', 
        'limit': 3
        }
    
    response = requests.get(url, params=params, timeout=30)
    
    # clean the response
    books = response.json()
    del books['numFound'], books['start'], books['numFoundExact'], books['num_found'], books['offset'], data['q']
    for book in books['docs']:
        book['isbn'] = book['isbn'][0]
        book['author_name'] = book['author_name'][0]

def ia_select(tags, k):
    
    
    cleaned_tags = clean_tags(tags)

    categorized_tags = categorize_tags('gender representation', 'african studies', cleaned_tags)

    cat_alpha, cat_beta = [], []
    for lst in categorized_tags:
        for tag in lst:
            if lst[tag] == 'Discipline':
                cat_alpha.append(tag)
            elif lst[tag] == 'Diversity':
                cat_beta.append(tag)
    
    # go through what each c means
    L = []
    U = {c: P(c) for c in (cat_alpha + cat_beta)}
    
    max_score = 0
    best_book = None
    
    books = get_suggestions(cat_alpha, cat_beta)

    while len(L) < k:
        for book in books:
            score = sum([U[c] * V(book, c) for c in (cat_alpha + cat_beta)])
            
            if score > max_score:
                max_score = score
                best_book = book
                
        L.append(best_book)
        
        for c in (best_book['subject']): #change to use the correct c
            U[c] = (1 - V(book)) * U(c, L) if len(L) > 0 else (1 - V(book)) * U[c]
        
        books.remove(best_book)
        
    return L
            
            
def P(category):
    pass
    
def U(book, category):
    pass

def V(book, cat_alpha):
    pass
        

    
def get_tags_for_categories(dir):

    category_tags = []
    
    for file in listdir(dir):
        if file.endswith('.csv'):
            
            tags = []
            with open(f"{dir}/{file}", 'r', encoding='utf-8-sig', errors='replace') as f:
                print(f"Reading {file}")
                content = f.read().replace('\xa0', ' ')  # replace non-breaking space (temporary fix)
                reader = csv.DictReader(content.splitlines())
                isbns = []
                for row in reader:
                    # print(row)
                    isbns.append(row['isbn'])
                tags = (sum(get_tags(isbns), []))
            # print(tags)
            dict = {file[:-3]: tags}
            print(dict)
            category_tags.append(dict)
                
    # write tags to a new json file as a new key
    with open('tags.json', 'w') as f:
        json.dump(category_tags, f)
    
         
            


if __name__ == '__main__':
    # get_tags_for_categories("../example_syllabi")
    
    

    # Cleaning tags from the two isbns defined above
    # tags = get_tags([9780192832696, 9780451015594])

    # cleaned_tags = clean_tags(tags)
    # print(cleaned_tags)
    # categorized_tags = categorize_tags(cleaned_tags)
    # print(categorized_tags)
    
    print(calculate_diversity_metrics([9780192832696, 9780451015594]))
