text = ""
with open("conveniotext1.txt", encoding="utf8") as input:
    file_content = input.readlines()
    for line in file_content:
        if len(line) > 1 and line[-2] != ".":
            line = line[:-1]
        text = text + line
print(text.split("\n\n"))