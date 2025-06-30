# xh_lexicographic_taggers

Lexicographic POS tagging experiment for isiXhosa.

Data is from [here](https://repo.sadilar.org/items/eb524a86-8953-4d2b-89d9-f40f5860e36c):
"Linguistically enriched corpora for conjunctively written South African languages" by Martin Puttkammer and Tanja 
Gaustad.

## Results

Best model (ours) is bolded:

|Model                            |Accuracy (including noun classes)|Accuracy (not including noun classes)|
|---------------------------------|---------------------------------|-------------------------------------|
|[Du Toit & Puttkammer](https://www.mdpi.com/2078-2489/12/12/520)             | 74.3590%                        | 75.6410%                            |
|Ours                             | **82.0513%**                        | **87.1795%**                            |
|[NLAPOST21 shared task winner](https://upjournals.up.ac.za/index.php/dhasa/article/view/3865), crf, comp sum, bigram | (does not attempt to tag noun classes)                        | 64.1026%                            |
|[NLAPOST21 shared task winner](https://upjournals.up.ac.za/index.php/dhasa/article/view/3865), lstm, comp sum, bigram| (does not attempt to tag noun classes)                      | 60.2564%                            |

**NB:** It is very important to note that this is on words as they would appear in a dictionary, e.g. on [IsiXhosa.click](https://isixhosa.click). This means it is an 'unfair' comparison for these taggers, which are trained on a sentence level. These results do not mean that our tagger is better for _any_ usecase _other_ than dictionary entries formatted like ours are on IsiXhosa.click.

## State of the codebase

I had to hack this together a bit as a proof-of-concept so the code is in a not-so-great
state. If you want to actually _use_ this, please contact me and I'll gladly clean it up for you.
This is mostly here for posterity + transparency and is released as-is.

## Licensing

All code is licensed under Apache 2.0.
Data retains its original licensing, and we merely redistribute it for convenience.
