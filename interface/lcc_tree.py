import json
import string
from collections import defaultdict, deque

def create_loc_graph():
    with open('../backend/detailed-lcc.json') as detailed_file, open('../backend/top-level-lcc.json') as top_level_file:
        detailed_data = json.load(detailed_file)
        top_level_data = json.load(top_level_file)

    loc_tree = defaultdict(list)
    
    # Add the root node and all the top-level nodes to the tree (e.g., A, B, C, D, etc.)
    # Algins with the top-level nodes in top-level-lcc.json
    loc_tree["root"] = list(string.ascii_uppercase)
    for letter in string.ascii_uppercase:
        loc_tree["Class " + letter] = []
        
    # Add all second-level nodes to the tree (e.g., AC, AE, AD, etc.)
    # These are top-level in detailed-lcc.json
    for prefix in detailed_data.keys():
        loc_tree[prefix] = [] # for some reason, there is a non top-level A node
        loc_tree["Class " + prefix[0]].append(prefix)
        
    # Add all subsequent nodes to the tree
    # These are second-level in detailed-lcc.json
    for prefix, nodes in detailed_data.items():
        for node in nodes:
            
            node_id = node["id"]
            if node_id == prefix: 
                break
                    
            # if no parents, add prefix (e.g., AC, AE, AD, etc.) as parent
            parents = node["parents"] if node["parents"] else [prefix]
            
            # Add the node to its direct parent's list of children
            loc_tree[parents[-1]].append(node_id) # last index is closest
        
    print(loc_tree["Class B"])
    
            
    
    def print_tree(node, graph, file, level=0):
        # Print the current node with indentation based on the level
        file.write(" " * (level * 4) + node + "\n")
        
        # Recur for each child of the current node
        for child in graph.get(node, []):
            print_tree(child, graph, file, level + 1)

    #Example Usage: Start printing from the root node "root"
    file = open("lcc_tree.txt", "w")
    print_tree("root", loc_tree, file)
    file.close()
    
  


if __name__ == "__main__":
    create_loc_graph()