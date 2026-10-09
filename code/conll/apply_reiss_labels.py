"""Run in the isolated tp env. Apply Reiss et al.'s label corrections (Tag, Wrong, Span, Both, Missing) to eng.testb
using their own released code; Token and Sentence corrections are not applied (they change tokenization)."""
import sys
sys.path.insert(0, sys.argv[1])            # data/raw/conll/reiss/scripts
from download_and_correct_corpus import process_label_file
process_label_file("test", sys.argv[2], sys.argv[3], target_file=sys.argv[4])
