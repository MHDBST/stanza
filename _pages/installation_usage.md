---
title: Installation & Getting Started
keywords: installation-download
permalink: '/installation_usage.html'
---

To use Stanza Neural Pipeline, you first need to install the package and download the model for the language you want to use. Then you can build the pipeline with downloaded models. Once the pipeline is built, you can process the document and get annotations.

For the usage information of Stanford CoreNLP Client, you can check out [here](corenlp_client.md).

## Installation

Stanza supports Python 3.6 or later. We strongly recommend that you install Stanza from PyPI. This should also help resolve all of the dependencies of Stanza, for instance [PyTorch](https://pytorch.org/) 1.0.0 or above. If you already have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run:
```bash
pip install stanza
```

If you currently have a previous version of `stanza` installed, use:
```bash
pip install stanza -U
```

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of Stanza and training your own models. For this option, run:
```bash
git clone https://github.com/stanfordnlp/stanza.git
cd stanza
pip install -e .
```

## Pipeline Building

Stanza provides simple, flexible, and unified interfaces for downloading and loading various [Processor](pipeline.md#processors)s. You can easily build the desired [Pipeline](pipeline.md#pipeline) (containing a list of [Processor](pipeline.md#processors)s) to annotate documents. Note that loading has the same interface as downloading, but allowing more options that can control devices (CPU or GPU), use pretokenized text, specify model path, etc. A full list of available options can be found for [downloading](models#downloading-and-using-models) and [loading](pipeline.md#pipeline). Here we provide some intuitive examples covering most use cases:

Download and load the default [Processor](pipeline.md#processors)s for English:
```python
>>> stanza.download('en')
>>> nlp = stanza.Pipeline('en')
```

Download and load the `default` [TokenizeProcessor](tokenize.md) and [POSProcessor](pos.md) for Chinese:
```python
>>> stanza.download('zh', processors='tokenize,pos')
>>> nlp = stanza.Pipeline('zh', processors='tokenize,pos')
```

Download and load the [TokenizeProcessor](tokenize.md) and [MWTProcessor](mwt.md) trained on `GSD` dataset for German:
```python
>>> stanza.download('de', processors='tokenize,mwt', package='gsd')
>>> nlp = stanza.Pipeline('de', processors='tokenize,mwt', package='gsd')
```

Download and load the [NERProcessor](ner.md) trained on `CoNLL03` dataset and all other `default` processors for Dutch:
```python
>>> stanza.download('nl', processors={'ner': 'conll03'})
>>> nlp = stanza.Pipeline('nl', processors={'ner': 'conll03'})
```

Download and load the [NERProcessor](ner.md) trained on `WikiNER` dataset, and other processors trained on `PADT` dataset for Arabic:
```python
>>> stanza.download('ar', processors={'ner': 'wikiner'}, package='padt')
>>> nlp = stanza.Pipeline('ar', processors={'ner': 'wikiner'}, package='padt')
```

Download and load the [TokenizeProcessor](tokenize.md) trained on `GSD` dataset, [POSProcessor](pos.md) trained on `Spoken` dataset, [NERProcessor](ner.md) trained on `CoNLL03` dataset, and `default` [LemmaProcessor](lemma.md) for French:
```python
>>> stanza.download('fr', processors={'tokenize': 'gsd', 'pos': 'spoken', 'ner': 'conll03', 'lemma': 'default'}, package=None)
>>> nlp = stanza.Pipeline('fr', processors={'tokenize': 'gsd', 'pos': 'spoken', 'ner': 'conll03', 'lemma': 'default'}, package=None)
```

Download and load the `default` [Processor](pipeline.md#processors)s for English from current working directory, and print all the information for debugging:
```python
>>> stanza.download('en', dir='.', logging_level='DEBUG')
>>> nlp = stanza.Pipeline('en', dir='.', logging_level='DEBUG')
```

## Document Annotation

Once the [Pipeline](pipeline.md#pipeline) is loaded, you can simply pass the text to the [Pipeline](pipeline.md#pipeline) and get the annotated [Document](data_objects#document) instance:

```python
>>> doc = nlp('Barack Obama was born in Hawaii.')
```

Within a [Document](data_objects#document), annotations are further stored in [Sentence](data_objects#sentence)s, [Token](data_objects#token)s, [Word](data_objects#word)s, [Span](data_objects#span)s in a top-down fashion. A List of all annotations and functions can be found in the [Data Objects](data_objects#document) page.

Print the text and POS tag of each word in the document:
```python
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.pos)
```

Print all entities and dependencies in the document:
```python
for sentence in doc.sentences:
    print(sentence.entities)
    print(sentence.dependencies)
```