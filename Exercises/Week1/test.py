print('what is your name?')
myName = input()
file = open("guest.txt", 'w')
file.write(myName)
file.close()

i = 0
while i < 3:
    print(i)
    i +=1
