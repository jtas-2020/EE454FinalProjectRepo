# use pound signs to define a comment for a line
""" This is how comments
can go across multiple
lines so you don't have
to keep putting hastags
every line of an essay"""

print("Hello world!\n")
print("Who are you?")
first_variable = input()
print("Was " + first_variable + " your input?")
print("Whatever...")

''' Indentation in Python is the 
equivalent of using {} in Java. No
need to use semicolons to terminate
every command'''

# for loop example (credit: google search)
my_string = "python"
x = 0
for i in my_string:
    x = x + 1
    print(my_string[0:x])

for i in my_string:
    x = x - 1
    print(my_string[0:x])

for i in range(0,len(my_string)):
    print(my_string[i] + " "),

''' output:
p
py
pyt
pyth
pytho
python
pytho
pyth
pyt
py
p
'''

print("\n")