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
        # si hay más de una entidad, de vuelve 1 que será identificado como tipo I de error
        entity = 1
    else:
        # es una tuple si hay valor único, si no encuentra ninguna devolverá 0 que será tipo II de error
        try:
            entity = entities_list[0]
        except IndexError:
            entity = 0
    return entity
