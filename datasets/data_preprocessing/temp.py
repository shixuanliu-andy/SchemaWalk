import pandas as pd
import numpy as np
from typetree import typetree
from time import time

def get_parents(child, tax):
    index = (tax['child']==child)
    return tax[index]['parent']

def get_ancestors_(childs, tax, acceptable_types_occurence, MAXTYPE=2):
    time0 = time()
    acceptable_types = acceptable_types_occurence.keys()
    ancestors = set()
    while len(childs)>0:
        child = childs.pop(-1)
        if child in acceptable_types:
            ancestors.add(child)
            continue
        index = (tax['child']==child)
        parents = set(tax[index]['parent'])
        intersection = parents.intersection(acceptable_types)
        if intersection:
            ancestors.update(intersection)
            childs.extend(list(parents))
        else:
            childs.extend(list(parents))
        childs = sorted(list(set(childs)), key=lambda i:len(i))
    ret_types = list(set(ancestors).intersection(acceptable_types))
    ret_types = sorted(ret_types, key=lambda i:acceptable_types_occurence[i], reverse=True)[:MAXTYPE]
    time1 = time()
    print (time1-time0)
    return ret_types

def get_ancestors(childs, tax, acceptable_types_occurence, MAXTYPE=3):
    time0 = time()
    counter = 0
    acceptable_types = set(acceptable_types_occurence.keys())
    ancestors = childs.intersection(acceptable_types)
    while len(childs-ancestors)>0:
        child_df = tax[tax['child'].isin(childs)]
        child_df = child_df[~child_df['parent'].isin(ancestors)]
        childs = set([i for i in child_df['parent'] if 'wordnet' in i or 'wikicat' in i])
        ancestors.update(childs & acceptable_types)
        wordnets = set([i for i in childs if 'wordnet' in i]) - acceptable_types
        childs -= wordnets
        if all(['wordnet' in i for i in childs]) or counter>=5:
            break
        counter += 1
    # ret_types = list(set(ancestors).intersection(acceptable_types))
    ret_types = sorted(list(ancestors), key=lambda i:acceptable_types_occurence[i], reverse=True)[:MAXTYPE]
    time1 = time()
    print(counter)
    print (time1-time0)
    return ret_types

def gen_acceptable_types_occurence(simple_tax, ent_types):
    simple_tax = simple_tax[simple_tax['rel']=='rdfs:subClassOf']
    types = [i for i in simple_tax[simple_tax['parent']=='owl:Thing']['child'] if 'wordnet' in i]
    for i in types.copy():
        child_type = [i for i in simple_tax[simple_tax['parent']==i]['child'] if 'wordnet' in i]
        types.extend(child_type)
    occurence = ent_types.stack().value_counts()
    return {i:occurence[i] for i in types if i in occurence}


data_name = 'yago_new'
input_dir_base = '../data_preprocessed/{}/original/'.format(data_name)
print ('loading yagoSimpleTaxonomy.tsv')
simple_tax = pd.read_csv(input_dir_base+'yagoSimpleTaxonomy.tsv',
                         sep='\t', header=0,names=['child','rel','parent'], usecols=[1,2,3])
print ('loading yagoSimpleTypes.tsv')
ent_types = pd.read_csv(input_dir_base+'yagoSimpleTypes.tsv',
                        sep='\t', header=0, names=['type'], usecols=[3])
acceptable_types_occurence = gen_acceptable_types_occurence(simple_tax, ent_types)
print ('loading yagoTaxonomy.tsv')
true_tax = pd.read_csv(input_dir_base+'yagoTaxonomy.tsv',
                        sep='\t', header=0,names=['child','parent'], usecols=[1,3])
# tree=typetree(input_dir_base)
new_list = ['<wikicat_Alumni_of_Keble_College,_Oxford>',
  '<wikicat_English_cricketers>',
  '<wikicat_Living_people>',
  '<wikicat_Mashonaland_cricketers>',
  '<wikicat_Nottinghamshire_cricketers>',
  '<wikicat_Oxford_University_cricketers>',
  "<wikicat_People_educated_at_King_Edward's_School,_Birmingham>",
  '<wikicat_Warwickshire_cricketers>']
# res=tree.filter_typelist(new_list)
ancestors = get_ancestors(set(new_list),
                          true_tax, acceptable_types_occurence)