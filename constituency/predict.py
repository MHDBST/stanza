import stanza

# nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
# doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on the input text
# print(doc) # Look at the result
# pipe=stanza.Pipeline('en')
# print(pipe.processors['constituency'].vocab['constituency'])
# print(list(pipe.processors['constituency'].vocab['deprel']._unit2id.keys()))
# stanza.Pipeline.processors['ner'].get_known_tags()
mdl_pth_out='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/saved_models/constituency/en_out_constituency.pt'
mdl_pth_my='/home/mbastan/structuralDecoding/neurologic_decoding/stanza/stanza-train/stanza/saved_models/constituency/en_my_constituency.pt'
sent='Barack Obama was born in Hawaii.'
mdl_pth = mdl_pth_my
while True:
    sent = input("Enter the input text \n")
    model_selection = input("Select model: 1 for general model, 2 for model trained with modified data, 3 for default model \n")
    if '2' in model_selection:
        mdl_pth = mdl_pth_out
    elif '1' in model_selection:
        mdl_pth = mdl_pth_my
    else:
        mdl_pth =None
    # print('using model : %s'%(mdl_pth.split('/')[-1]))


    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',model_path=mdl_pth)
    doc = nlp(sent)
    for sentence in doc.sentences:
        print('original sentence:',sent)
        print(sentence.constituency)