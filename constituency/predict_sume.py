import stanza
import pandas as pd

input_csv = pd.read_csv('/home/mbastan/XP/ie_t5/best_model/afterpretrain_pretrain_all_4_8/postprocess_out_v1.csv')
true_exp=list(input_csv['True_Exp'])
pred_exp=list(input_csv['BM_Exp'])
mdl_pth_out='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/saved_models/constituency/en_out_constituency.pt'
mdl_pth_my='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/stanza/saved_models/constituency/en_my_constituency.pt'
mdl_pth = mdl_pth_out
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',model_path=mdl_pth)
# stanza.download('en', package='craft')
nlp = stanza.Pipeline('en', package='craft')
for tsent,psent in zip(true_exp,pred_exp):
    

    doc = nlp(tsent)
    print(doc.sentences[0].deprel)
    break
    # for sentence in doc.sentences:
    #     print('original sentence:',sent)
    #     print(sentence.constituency)