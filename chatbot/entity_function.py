# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:36:48 2021

@author: luism
"""

import spacy
def getEntities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities_list = [(X.text, X.label_) for X in doc.ents]
    if len(entities_list)>1:
        # es una lista si hay más de una entidad
        entity = 1
    else:
        # es una tuple si hay valor único
        try:
            entity = entities_list[0]
        except IndexError:
            entity = 0
    return entity