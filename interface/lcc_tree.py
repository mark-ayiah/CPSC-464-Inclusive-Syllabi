import json
import string
from collections import defaultdict, deque

def get_loc_tree():
    with open('../backend/detailed-lcc.json') as detailed_file, open('../backend/top-level-lcc.json') as top_level_file:
        detailed_data = json.load(detailed_file)
        top_level_data = json.load(top_level_file)

    loc_tree = defaultdict(list)
    
    # Add the root node and all the top-level nodes to the tree (e.g., A, B, C, D, etc.)
    # Algins with the top-level nodes in top-level-lcc.json
    loc_tree["root"] = []
    
    for prefix in top_level_data.keys():
        loc_tree["Class " + prefix] = []
        loc_tree["root"].append("Class " + prefix)
    
        
    # Add all second-level nodes to the tree (e.g., AC, AE, AD, etc.)
    # These are top-level in detailed-lcc.json
    for prefix in detailed_data.keys():
        loc_tree["Subclass " + prefix] = [] # for some reason, there is a non top-level A node
        loc_tree["Class " + prefix[0]].append("Subclass " + prefix)
        
    # print(loc_tree)
    # Add all subsequent nodes to the tree
    # These are second-level in detailed-lcc.json
    for prefix, nodes in detailed_data.items():
        for node in nodes:
            
            node_id = node["id"]
    
            # if no parents, add prefix (e.g., AC, AE, AD, etc.) as parent
            # parent = node["parents"] if node["parents"] else [prefix]
            if node["parents"]:
                parent = node["parents"][-1]
            else:
                parent = "Subclass " + prefix
            if node_id == "A":
                print(parent)
            # Add the node to its direct parent's list of children
            loc_tree[parent].append(node_id) # last index is closest
            

    return loc_tree
        
   
    
    
    
            
    
def write_tree(node, graph, file, level=0):
    # Print the current node with indentation based on the level
    file.write(" " * (level * 4) + node + "\n")
    
    # Recur for each child of the current node
    for child in graph.get(node, []):
        write_tree(child, graph, file, level + 1)


# def calculate_distance(graph, node1, node2):
    

    
  


if __name__ == "__main__":
    loc_tree = get_loc_tree()
    with open("lcc_tree.txt", "w") as file:
        write_tree("root", loc_tree, file)
    