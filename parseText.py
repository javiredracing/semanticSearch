def processText(input):
    text = ""
    splitted_text = []
    file_content = input.readlines()
    for line in file_content:
        #line = line.decode("utf-8")
        aux = line.strip()
        if len(aux) == 0:
            #line = PARSER_KEY
            if len(text) > 0:                
                splitted_text.append(text.strip())
                text = ""
            continue
        elif len(aux) > 0 and not aux.endswith(".") and not aux.endswith(":"):
           line = aux + " "
        text = text + line
    return splitted_text

text = []
with open("metropolitanoRaw.txt", encoding="utf8") as input:
    text = processText(input)

#print(text)
with open("docs/salida.txt", "w",encoding="utf-8") as text_file:
    for i in text:
        text_file.write(i)
        text_file.write("\n\n")

