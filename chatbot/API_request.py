
import wikipedia as wiki
wiki.set_lang("en")

num_resultados = 1
sugerencias = False
next = False
while next == False:
    keyboard = input('\nHi, What do you want me to look up in Wikipedia? \n')
    try:
        #print("He encontrado: .{}".format(wikipedia.search(keyboard, results)))
        print(wiki.summary(keyboard))
    except:
        print("\nI have not found any coincidence. \n")
        if wiki.suggest(keyboard) != None:
            print("Did you mean {} \n".format(wiki.suggest(keyboard)))
        else:
            print("Try again")



