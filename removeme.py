import stanza
pipe = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package={'constituency': 'wsj_bert'})

print(pipe.processors["constituency"].get_constituents())
