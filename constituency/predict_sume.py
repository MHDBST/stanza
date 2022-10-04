#%%
import stanza
import pandas as pd
import pickle
#%%
ind = -1
input_csv = pd.read_csv('/home/mbastan/XP/ie_t5/best_model/afterpretrain_pretrain_all_4_8/postprocess_out_v1.csv')
true_exp=list(input_csv['True_Exp'])[:ind]
pred_exp=list(input_csv['BM_Exp'])[:ind]
true_regs=list(input_csv['True_reg'])[:ind]
true_eles=list(input_csv['True_ele'])[:ind]
mdl_pth_out='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/saved_models/constituency/en_out_constituency.pt'
mdl_pth_my='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/stanza/saved_models/constituency/en_my_constituency.pt'
mdl_pth = mdl_pth_out
p_tree =[]
t_tree =[]
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',model_path=mdl_pth)
# stanza.download('en', package='craft')
# t_docs = true_exp[0].replace('<el>','').replace('<le>','').replace('<re>','').replace('<er>','')#[stanza.Document([], text=d) for d in true_exp]
# print(t_docs)
# p_docs = pred_exp[0].replace('<el>','').replace('<le>','').replace('<re>','').replace('<er>','')#[stanza.Document([], text=d) for d in pred_exp]
# print(p_docs)
def clean_doc(d):
    return(d.replace('<el>','').replace('<le>','').replace('<re>','').replace('<er>',''))



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
#%%
def create_files():
 nlp = stanza.Pipeline('en', processors='lemma,tokenize,pos,depparse',package='craft')
 t_docs = [stanza.Document([], text=clean_doc(d)) for d in true_exp if pd.notna(d)]
 p_docs = [stanza.Document([], text=clean_doc(d)) for d in pred_exp if pd.notna(d)]
 tdoc = nlp(t_docs)
 pdoc = nlp(p_docs)
 with open('t_regs.pk','wb') as f_t_regs:
    with open('t_eles.pk','wb') as f_t_eles:
        with open('p_eles.pk','wb') as f_p_eles:
            with open('p_regs.pk','wb') as f_p_regs:
                # for i,(tsent,psent) in enumerate(zip(tdoc.sentences,pdoc.sentences)):
                t_connection_arr_regs=[]
                t_connection_arr_eles=[]
                p_connection_arr_regs=[]
                p_connection_arr_eles=[]
                for i,(tsent,psent) in enumerate(zip(tdoc,pdoc)):
                    print('processing entry at %d'%i)
                    t_map ={}
                    p_map={}
                    tsent=tsent.sentences[0]
                    psent=psent.sentences[0]
                    for word in tsent.words:
                        if word.text.lower() not in t_map:
                            t_map[word.text.lower()]=[[word.deprel,tsent.words[word.head-1].text if word.head>0 else "root"]]
                        else:
                            t_map[word.text.lower()].append([word.deprel,tsent.words[word.head-1].text if word.head>0 else "root"])
                    t_connection_arr_reg=get_dep_tree_connections(t_map,true_regs[i].lower())
                    t_connection_arr_ele=get_dep_tree_connections(t_map,true_eles[i].lower())

                    if not t_connection_arr_reg or not t_connection_arr_ele:
                        continue
                    # print('reg connection:',t_connection_arr_regs)
                    # print('ele connection:',t_connection_arr_ele)
                    # print('---------')
                    for word in psent.words:
                        if word.text.lower() not in p_map:
                            p_map[word.text.lower()]=[[word.deprel,psent.words[word.head-1].text if word.head>0 else "root"]]
                        else:
                            p_map[word.text.lower()].append([word.deprel,psent.words[word.head-1].text if word.head>0 else "root"])
                    p_connection_arr_reg=get_dep_tree_connections(p_map,true_regs[i].lower())
                    p_connection_arr_ele=get_dep_tree_connections(p_map,true_eles[i].lower())

                    if not p_connection_arr_reg or not p_connection_arr_ele:
                        continue
                    t_connection_arr_regs.append(t_connection_arr_reg)
                    t_connection_arr_eles.append(t_connection_arr_ele)
                    p_connection_arr_regs.append(p_connection_arr_reg)
                    p_connection_arr_eles.append(p_connection_arr_ele)
                    # print('reg connection:',p_connection_arr_regs)
                    # print('ele connection:',p_connection_arr_ele)
                    # print('<--------->')
                print(len(t_connection_arr_regs))
                print(len(t_connection_arr_eles))
                print(len(p_connection_arr_regs))
                print(len(p_connection_arr_eles))
                pickle.dump(t_connection_arr_regs,f_t_regs)
                pickle.dump(t_connection_arr_eles,f_t_eles)
                pickle.dump(p_connection_arr_regs,f_p_regs)
                pickle.dump(p_connection_arr_eles,f_p_eles)
                return(t_connection_arr_regs,t_connection_arr_eles,p_connection_arr_eles,p_connection_arr_regs)



try:
    # a=b
    with open('t_regs.pk','rb') as f_t_regs:
        t_connection_arr_regs=pickle.load(f_t_regs)
    with open('t_eles.pk','rb') as f_t_eles:
        t_connection_arr_eles=pickle.load(f_t_eles)
    with open('p_eles.pk','rb') as f_p_eles:
        p_connection_arr_eles=pickle.load(f_p_eles)
    with open('p_regs.pk','rb') as f_p_regs:
        p_connection_arr_regs=pickle.load(f_p_regs)
    print('file loaded')
except:
    print('creating files')
    t_connection_arr_regs,t_connection_arr_eles,p_connection_arr_eles,p_connection_arr_regs=create_files()
print(t_connection_arr_regs[0])
print(t_connection_arr_eles[0])
print(p_connection_arr_eles[0])
print(p_connection_arr_regs[0])