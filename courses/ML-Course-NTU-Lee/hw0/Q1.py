with open('words.txt', 'r') as file:
    
    word_list = file.read().split(' ')

word = []
position = []
count = {}
for i in range(len(word_list)):
    if word_list[i] not in word:
        word.append(word_list[i].strip('\n'))
        position.append(i)
        count[word_list[i].strip('\n')] = 1
    else:
        rep = word_list[i].strip('\n')
        count[rep] += 1

result = []
for i in range(len(word)):
    a = [word[i], position[i], count[word[i]]]
    result.append(a)

for i in range(len(result)):
    print(result[i])


