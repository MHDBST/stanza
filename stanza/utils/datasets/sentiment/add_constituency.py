"""
For a dataset produced by prepare_sentiment_dataset, add constituency parses.

Obviously this will only work on languages that have a constituency parser
"""

import argparse
import os

import stanza
from stanza.models.classifiers.data import read_dataset
from stanza.models.classifiers.utils import WVType
from stanza.models.mwt.utils import resplit_mwt
from stanza.utils.datasets.sentiment import prepare_sentiment_dataset
from stanza.utils.datasets.sentiment import process_utils
import stanza.utils.default_paths as default_paths

SHARDS = ("train", "dev", "test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="Dataset to process")
    parser.add_argument('--constituency_package', type=str, default=None, help="Constituency model to use for parsing")
    args = parser.parse_args()

    paths = default_paths.get_default_paths()
    expected_files = [os.path.join(paths['SENTIMENT_DATA_DIR'], '%s.%s.json' % (args.dataset, shard)) for shard in SHARDS]
    for filename in expected_files:
        if not os.path.exists(filename):
            print("Cannot find expected dataset file %s - rebuilding dataset" % filename)
            prepare_sentiment_dataset.main(args.dataset)
            break

    lang, shortname = args.dataset.split("_", 1)

    pipeline_args = {"lang": lang,
                     "processors": "tokenize,pos,constituency",
                     "tokenize_pretokenized": True,
                     "pos_tqdm": True,
                     "constituency_tqdm": True}
    package = {}
    if args.constituency_package is not None:
        package["constituency"] = args.constituency_package
    if package:
        pipeline_args["package"] = package
    pipe = stanza.Pipeline(**pipeline_args)

    if "mwt" in pipe.processors:
        print("This language has MWT.  Because the sentiment dataset was prepared with *tokens*, whereas the constituency uses *words*, we must resplit.")
        mwt_pipe = stanza.Pipeline(lang=lang, processors="tokenize,mwt")

    for filename in expected_files:
        dataset = read_dataset(filename, WVType.OTHER, 1)
        text = [x.text for x in dataset]
        if "mwt" in pipe.processors:
            print("Resplitting MWT in %d sentences from %s" % (len(dataset), filename))
            doc = resplit_mwt(text, mwt_pipe)
            print("Parsing %d sentences from %s" % (len(dataset), filename))
            doc = pipe(doc)
        else:
            print("Parsing %d sentences from %s" % (len(dataset), filename))
            doc = pipe(text)

        assert len(dataset) == len(doc.sentences)
        for datum, sentence in zip(dataset, doc.sentences):
            datum.constituency = sentence.constituency

        process_utils.write_list(filename, dataset)

if __name__ == '__main__':
    main()
