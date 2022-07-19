"""
Test some pieces of the depparse dataloader
"""
import pytest

import os
import tempfile

from stanza.models import tagger
from stanza.models.common import pretrain
from stanza.models.pos.data import DataLoader
from stanza.models.pos.trainer import Trainer
from stanza.models.pos.vocab import WordVocab, XPOSVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory
from stanza.utils.conll import CoNLL

from stanza.tests import TEST_WORKING_DIR

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

EN_EXAMPLE="""
1	Sh'reyan	Sh'reyan	PROPN	NNP%(tag)s	Number=Sing	3	nmod:poss	3:nmod:poss	_
2	's	's	PART	POS%(tag)s	_	1	case	1:case	_
3	antennae	antenna	NOUN%(tag)s	NNS	Number=Plur	6	nsubj	6:nsubj	_
4	are	be	VERB	VBP%(tag)s	Mood=Ind|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
5	hella	hella	ADV	RB%(tag)s	_	6	advmod	6:advmod	_
6	thicc	thicc	ADJ	JJ%(tag)s	Degree=Pos	0	root	0:root	_
"""

EMPTY_TAG = lambda x: ""
DASH_TAGS = lambda x: "-%d" % x

def build_doc(iterations, suffix):
    """
    build N copies of the english text above, with a lambda function applied for the tag suffices

    for example:
      lambda x: "" means the suffices are all blank (NNP, POS, NNS, etc) for each iteration
      lambda x: "-%d" % x means they go (NNP-0, NNP-1, NNP-2, etc) for the first word's tag
    """
    texts = [EN_EXAMPLE % {"tag": suffix(i)} for i in range(iterations)]
    text = "\n\n".join(texts)
    doc = CoNLL.conll2doc(input_str=text)
    return doc

def build_data(iterations, suffix):
    """
    Same thing, but passes the Doc through a POS Tagger DataLoader
    """
    doc = build_doc(iterations, suffix)
    data = DataLoader.load_doc(doc)
    return data
    

def test_basic_en_ewt():
    """
    en_ewt is currently the basic vocab

    note that this may change if the dataset is drastically relabeled in the future
    """
    data = build_data(1, EMPTY_TAG)
    vocab = xpos_vocab_factory(data, "en_ewt")
    assert isinstance(vocab, WordVocab)


def test_basic_en_unknown():
    """
    With only 6 tags, it should use a basic vocab for an unknown dataset
    """
    data = build_data(10, EMPTY_TAG)
    vocab = xpos_vocab_factory(data, "en_unknown")
    assert isinstance(vocab, WordVocab)



def test_dash_en_unknown():
    """
    With this many different tags, it should choose to reduce it to the base xpos removing the -
    """
    data = build_data(10, DASH_TAGS)
    vocab = xpos_vocab_factory(data, "en_unknown")
    assert isinstance(vocab, XPOSVocab)
    assert vocab.sep == "-"

def check_reload(pt, shorthand, iterations, suffix, expected_vocab):
    """
    Build a Trainer (no actual training), save it, and load it back in to check the type of Vocab restored

    TODO: This test may be a bit "eager" in that there are no other
    tests which check building, saving, & loading a pos trainer.
    Could add tests to test_trainer.py, for example
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
        args = tagger.parse_args(["--batch_size", "1", "--shorthand", shorthand])
        train_doc = build_doc(iterations, suffix)
        train_batch = DataLoader(train_doc, args["batch_size"], args, pt, evaluation=False)
        vocab = train_batch.vocab
        assert isinstance(vocab['xpos'], expected_vocab)

        trainer = Trainer(args=args, vocab=vocab, pretrain=pt, use_cuda=False)

        model_file = os.path.join(tmpdirname, "foo.pt")
        trainer.save(model_file)

        new_trainer = Trainer(model_file=model_file, pretrain=pt)
        assert isinstance(new_trainer.vocab['xpos'], expected_vocab)

@pytest.fixture(scope="module")
def pt():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)
    return pt

def test_reload_word_vocab(pt):
    """
    Test that building a model with a known word vocab shorthand, saving it, and loading it gets back a word vocab
    """
    check_reload(pt, "en_ewt", 10, EMPTY_TAG, WordVocab)

def test_reload_unknown_word_vocab(pt):
    """
    Test that building a model with an unknown word vocab, saving it, and loading it gets back a word vocab
    """
    check_reload(pt, "en_unknown", 10, EMPTY_TAG, WordVocab)

def test_reload_unknown_xpos_vocab(pt):
    """
    Test that building a model with an unknown xpos vocab, saving it, and loading it gets back an xpos vocab
    """
    check_reload(pt, "en_unknown", 10, DASH_TAGS, XPOSVocab)
