import itertools
"""
没看懂
"""
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, name in itertools.groupby(names, first_letter):
    print(letter, list(name))
