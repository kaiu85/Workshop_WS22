def lade_fragen_liste(filename):

    fragen = [];
    
    with open(filename) as f:
        content = f.readlines()
        #remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 
        
    print('%d Zeilen eingelesen, entspricht %d Fragen.' % (len(content),len(content)/6))
        
    for i in range(int(len(content)/6)):
        
        frage = {
            "frage": content[i*6],
            "antworten": content[i*6+1:i*6+5],
            "korrekt": int(content[i*6+5]) 
        }
        
        fragen.append(frage)
        
    return fragen
        
        
        
    