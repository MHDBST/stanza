import stanza
from datasets import load_dataset

dataset = load_dataset("sick")
mdl_pth_out='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/saved_models/constituency/en_out_constituency.pt'
mdl_pth_my='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/stanza/saved_models/constituency/en_my_constituency.pt'
nlp = stanza.Pipeline('en', processors='lemma,tokenize,pos,depparse')
sAs=dataset['validation']['sentence_A']
sBs=dataset['validation']['sentence_B']

a_docs = [stanza.Document([], text=d) for d in sAs]
b_docs = [stanza.Document([], text=d) for d in sBs]
Adoc = nlp(a_docs)
Bdoc = nlp(b_docs)


def get_dep_tree_connections(mapp,entity):

    connection_arr=[]
    if entity in mapp:
        
        for arr in mapp[entity]:
            current_word = entity
            deprel = arr[0]
            head = arr[1]
            connection_arr.append([current_word,deprel,head])
            while head != 'root' and head.lower() != current_word:
                current_word = head.lower()
                if current_word not in mapp:
                    break
                for arr1 in mapp[current_word]:
                    deprel = arr1[0]
                    head = arr1[1]
                    connection_arr.append([current_word,deprel,head])
                    if [current_word,deprel,head] in connection_arr:
                        break

                if connection_arr.count([current_word,deprel,head]) > 2:
                    break
                    

        return connection_arr
    return None
for i,(Asent,Bsent) in enumerate(zip(Adoc,Bdoc)):
    print('processing entry at %d'%i)
    Asent=Asent.sentences[0]
    Bsent=Bsent.sentences[0]
    print(sAs[i])
    print(sBs[i])
    a_ents=[]
    b_ents=[]
    a_map={}
    b_map={}
    for word in Asent.words:
        if word.text.lower() not in a_map:
            a_map[word.text.lower()]=[[word.deprel,Asent.words[word.head-1].text if word.head>0 else "root"]]
        else:
            a_map[word.text.lower()].append([word.deprel,Asent.words[word.head-1].text if word.head>0 else "root"])
        if word.deprel=='nsubj' or word.deprel=='obj':
            a_ents.append(word.text.lower())
        
    print('amab:',a_map)
    for word in Bsent.words:
        if word.text.lower() not in b_map:
            b_map[word.text.lower()]=[[word.deprel,Bsent.words[word.head-1].text if word.head>0 else "root"]]
        else:
            b_map[word.text.lower()].append([word.deprel,Bsent.words[word.head-1].text if word.head>0 else "root"])
        if word.deprel=='nsubj'  or word.deprel=='obj':
            b_ents.append(word.text.lower())    
    print('b_map:',b_map)
    if len(a_ents)==0 or len(b_ents)==0:
        continue
    a_connection_arrs=[]
    b_connection_arrs=[]
    for ent in a_ents:
        a_connection_arr=get_dep_tree_connections(a_map,ent)
        a_connection_arrs.append(a_connection_arr)
    for ent in b_ents:
        b_connection_arr=get_dep_tree_connections(b_map,ent)
        b_connection_arrs.append(b_connection_arr)
    print('a_connection_arr:',a_connection_arrs)
    print('b_connection_arr:',b_connection_arrs)
    if i ==1:
        break