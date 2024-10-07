# for rating stuff

#import csv
import urllib
from urllib.parse import quote

#language identifier 1
# pip -q install fasttext
import fasttext
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model_ft = fasttext.load_model(model_path)

#language identifier 2
# pip -q install langid
from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True) #instantiate identifier

import requests
import re

# Importing other dependencies
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter

#testing Open Library API
# r = requests.get('https://openlibrary.org/search.json?q=subject:("dogs"+OR+"cats")+subject:("Juvenile fiction"+OR+"Juvenile literature")&fields=subject')
# r = r.json()
# subs = [d['subject'] for d in r['docs']] #gets the list, AKA value from k:v in subject:list dictionary
#print(subs[0])

#discipline tags is a list
#diversity tags is a list
#k is the number of items to return

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
    return requests.get(f'https://openlibrary.org/search.json?q=subject:({str_disc})+subject:({str_div})&fields=author_name,title,isbn,subject&limit={k}').json()

#book = get_books(syllabus); takes in a list of ISBNs
def get_tags(books):
    r = [requests.get(f'https://openlibrary.org/search.json?q=isbn:{isbn}&fields=subject').json()['docs'][0]['subject'] for isbn in books] #just give me the list of subjects plz
    return r

# lst = get_tags([9780192832696, 9780451015594])
#print(lst)

#takes in a list of lists
def clean_tags(tags):
    for idx, l in enumerate(tags): #index, list of lists

        #lowercase
        l = [s.lower() for s in l]

        #language identifier
        #We can either keep a tag if both methods AGREE that it is english OR only use one and set a probability threshold for english likelihood
        l = [s for s in l if model_ft.predict(s)[0][0] == '__label__eng_Latn'] #if english, using fast text; https://aclanthology.org/E17-2068/
        #if english, using langid
        l = [s for s in l if identifier.classify(s)[0] == 'en'] #off a cursory glance, performs better than the fasttext one, but still not as robust as using both; http://www.aclweb.org/anthology/P12-3005

        #remove mentions of "fiction" to prevent stripped pertinent information due to commas later on
        l = [s.split('in fiction')[0] for s in l] #remove any mention of 'fiction'
        l = [s.split(', fiction')[0] for s in l] #remove any mention of 'fiction'
        l = [s.split('fiction, ')[0] for s in l] #remove any mention of 'fiction'

        #clean for extraness
        l = [s.split(',')[0]  for s in l] #remove anything after a comma
        l = [s.split('--')[0]  for s in l] #remove anything with the --
        l = [s.split('(')[0]  for s in l] #remove parenthesis and anything within it
        l = [s.split('[')[0]  for s in l] #remove parenthesis and anything within it
        l = [s.split('{')[0]  for s in l] #remove parenthesis and anything within it
        l = [s.split('/')[0]  for s in l] #look at info before slash
        l = [s.split('"')[0]  for s in l] #remove quotes
        l = [s for s in l if ":" not in s] #remove anything with parentheses
        l = [s for s in l if "reading level" not in s] #remove any mention of reading level

        #remove other uninformative tags
        l = [s for s in l if "translations" not in s]
        l = [s for s in l if "staff" not in s] #staff picks
        l = [s for s in l if "language materials" not in s] #language materials

        #remove dewey system stuff until further notice
        l = [s for s in l if not s.isdigit()]

        #ampersand in the tags is causing problems

        #remove whitespace
        l = [s.strip(' \t\n\r') for s in l]

        #remove empty string
        l = [s for s in l if bool(s) != False]

        #make unique, update list
        tags[idx] = list(set(l))

    return tags #list of lists

# Cleaning tags from the two isbns defined above
# cleaned_tags = clean_tags(lst)



# Function to categorize tags
def categorize_tags(desired_diversity, overarching_discipline, tags):
    # Load model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Desired diversity area and overarching discipline
    desired_diversity = 'gender representation'
    overarching_discipline = 'african studies'
    categorized_tags = []
    for tag_list in tags:
        # Generate embeddings for tags, diversity area, and discipline
        tag_embeddings = model.encode(tag_list)
        diversity_embedding = model.encode([desired_diversity])
        discipline_embedding = model.encode([overarching_discipline])
        cat_tags = {}
        for i, tag in enumerate(tag_list):
            # Calculate cosine similarity to both diversity and discipline
            diversity_sim = cosine_similarity([tag_embeddings[i]], diversity_embedding)[0][0]
            discipline_sim = cosine_similarity([tag_embeddings[i]], discipline_embedding)[0][0]

            # Categorize based on higher similarity
            if (diversity_sim > .25) or (discipline_sim > .15):
                if diversity_sim > discipline_sim:
                    cat_tags[tag] = 'Diversity'
                elif discipline_sim > diversity_sim:
                    cat_tags[tag] = 'Discipline'
            else:
                cat_tags[tag] = 'Neither'
        categorized_tags.append(cat_tags)

    return categorized_tags

# Categorize tags
categorized_tags = categorize_tags(desired_diversity, overarching_discipline, cleaned_tags)

# Splitting the tags into their respective catogories
cat_alpha, cat_beta = [], []
for lst in categorized_tags:
    for tag in lst:
        if lst[tag] == 'Discipline':
            cat_alpha.append(tag)
        elif lst[tag] == 'Diversity':
            cat_beta.append(tag)
            
            # Load pre-trained model for topic embeddings


# Define Rao's entropy formula
def raos_entropy(cat_alpha, cat_beta):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings for topics
    alpha_embeddings = model.encode(cat_alpha)
    beta_embeddings = model.encode(cat_beta)

    # Full list of syllabus topics
    syllabus_topics = cat_alpha + cat_beta

    # Calculate proportions (p_i and p_j)
    topic_counts = Counter(syllabus_topics)
    total_topics = len(syllabus_topics)
    p_alpha = np.array([topic_counts[topic] / total_topics for topic in cat_alpha])
    p_beta = np.array([topic_counts[topic] / total_topics for topic in cat_beta])
    entropy = 0.0
    # Calculate pairwise cosine distances between topics
    distance_matrix = cosine_distances(alpha_embeddings, beta_embeddings)
    
    # Sum over all topic pairs
    for i in range(len(cat_alpha)):
        for j in range(len(cat_beta)):
            entropy += p_alpha[i] * p_beta[j] * distance_matrix[i, j]
    
    return entropy

# Calculate diversity using Rao's entropy
entropy = raos_entropy(cat_alpha, cat_beta)
print(f"Rao's Entropy (Diversity): {entropy}")