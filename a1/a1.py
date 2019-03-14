import nltk
from collections import Counter
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download("universal_tagset")

# Task 1 (1 mark)
def word_counts(text, words):
    """Return a vector that represents the counts of specific words in the text
    >>> word_counts("Here is sentence one. Here is sentence two.", ['Here', 'two', 'three'])
    [2, 1, 0]
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> word_counts(emma, ['the', 'a'])
    [4842, 3001]
    """
    vector = []
    final = []  # Create arrays to put words, and final count in
    for s in nltk.sent_tokenize(text):
        for w in nltk.word_tokenize(s): # tokenize the text into words
            vector += [w] # Add each word to the array
    wordCount = Counter(vector) # Count the words
    for word in words:
         final.append(wordCount[word]) # Specifically count the words asked
    return final

# Task 2 (1 mark)
def pos_counts(text, pos_list):
    """Return the sorted list of distinct words with a given part of speech
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> pos_counts(emma, ['DET', 'NOUN'])
    [14352, 32029]
    """
    temp = []
    answer = [] # Arrays for count and final answer
    pos = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)] # tokenize the text
    tagged = nltk.pos_tag_sents(pos, tagset="universal") # Tag each word
    for tag in tagged: 
        for word in tag:
            temp.append(word[1]) # Select only the tagged part
    posCounter = Counter(temp) 
    for p in pos_list:
        answer.append(posCounter[p]) #count the amount of PoS that is specified
    return answer

# Task 3 (1 mark)
import re
VC = re.compile('[aeiou]+[^aeiou]+', re.I)
def count_syllables(word):
    return len(VC.findall(word))

def compute_fres(text):
    """Return the FRES of a text.
    >>> emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    >>> compute_fres(emma) # doctest: +ELLIPSIS
    99.40...
    """
    sentence = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]
    totalSent = len(sentence) # Calculate the amount of sentences
    
    temp = []
    totalSyl = 0
    for s in nltk.sent_tokenize(text):
        for w in nltk.word_tokenize(s): #tokenize the text into words
            temp += [w] 
            totalSyl += count_syllables(w) #count the amount of syllables
    totalWord = len(temp) # Count the amount of words
    
    fres = 206.835-1.015*(totalWord/totalSent)-84.6*(totalSyl/totalWord) #Calculate FRES
    return fres

# Task 4 (2 marks)
import re
regexp = re.compile('.*(rst|ond|ird|fth|xth|nth|hth)$')
def annotateOD(listoftokens):
    """Annotate the ordinal numbers in the list of tokens
    >>> annotateOD("the second tooth".split())
    [('the', ''), ('second', 'OD'), ('tooth', '')]
    """
    result = []
    for t in listoftokens:
        if regexp.match(t):
            result.append((t, 'OD'))
        else:
            result.append((t, ''))
    return result
    
# DO NOT MODIFY THE CODE BELOW

def compute_f1(result, tagged):
    assert len(result) == len(tagged) # This is a check that the length of the result and tagged are equal
    correct = [result[i][0] for i in range(len(result)) if result[i][1][:2] == 'OD' and tagged[i][1][:2] == 'OD']
    numbers_result = [result[i][0] for i in range(len(result)) if result[i][1][:2] == 'OD']
    numbers_tagged = [tagged[i][0] for i in range(len(tagged)) if tagged[i][1][:2] == 'OD']
    if len(numbers_tagged) > 0:
        r = len(correct)/len(numbers_tagged)
    else:
        r = 0.0
    if len(numbers_result) > 0:
        p = len(correct)/len(numbers_result)
    else:
        p = 0.0
    return 2*r*p/(r+p)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    nltk.download('brown')
    tagged = nltk.corpus.brown.tagged_words(categories='news')
    words = [t for t, w in tagged]
    result = annotateOD(words)
    f1 = compute_f1(result, tagged)
    print("F1 score:", f1)
