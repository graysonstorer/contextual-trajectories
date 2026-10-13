import pandas as pd
import shlex

def incrementize(sentence):
    words = []
    temp_word = []
    for i in range(len(sentence)):
        if sentence[i] != ' ':
            temp_word.append(sentence[i])
        else:
            words.append(temp_word)
            temp_word = []
    if temp_word:
        words.append(temp_word)
    individual_words = []
    for word in words:
        individual_words.append(''.join(word))
    final = []
    for i in range(len(individual_words)):
        temp = " ".join(individual_words[0:i + 1])
        final.append(temp)
    return final
print("example of f_1 incrementizing the sentence 'I took the dog for a walk': ")
print(incrementize('I took the dog for a walk. '))


def incrementize_2(sentence):
    words = []
    temp_word = []
    for i in range(len(sentence)):
        if sentence[i] != ' ':
            temp_word.append(sentence[i])
        else:
            words.append(temp_word)
            temp_word = []
    if temp_word:
        words.append(temp_word)
    individual_words = []
    for word in words:
        individual_words.append(''.join(word))
    final = []
    for i in range(len(individual_words)):
        temp = " ".join(individual_words[0:i + 1])
        if i == 0 or i == len(individual_words) - 1:
            final.append(temp)
    return final


print("example of f_2 incrementizing the sentence 'I took the dog for a walk': ")
print(incrementize_2('I took the dog for a walk. '))


def incrementize_3(sentence):
    words = []
    temp_word = []
    for i in range(len(sentence)):
        if sentence[i] != ' ':
            temp_word.append(sentence[i])
        else:
            words.append(temp_word)
            temp_word = []
    if temp_word:
        words.append(temp_word)
    individual_words = []
    for word in words:
        individual_words.append(''.join(word))
    final = []
    for i in range(len(individual_words)):
        temp = " ".join(individual_words[0:i + 1])
        if i == 0 or i == len(individual_words) - 1 or i == int(len(individual_words) / 2):
            final.append(temp)
    return final


print("example of f_3 incrementizing the sentence 'I took the dog for a walk': ")
print(incrementize_3('I took the dog for a walk. '))