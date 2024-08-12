import random

def create_strings_from_file(filename, count, length):
    """
        Create all strings by reading lines in specified files
    """
    strings = []

    with open('corpus/' + filename, 'r', encoding="utf8") as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [" ".join(l.split()[::-1][0:random.randint(1, length)]) for l in lines]
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        while len(strings) < count:
            # if lang == 'fa'
            if len(lines) >= count - len(strings):
                strings.extend(lines[0:count - len(strings)])
            else:
                strings.extend(lines)
    return strings

def create_strings_from_dict(length, allow_variable, count, lang_dict, lang):
    """
        Create all strings by picking X random word in the dictionnary
    """
    print(lang)

    dict_len = len(lang_dict)
    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            current_string = lang_dict[random.randrange(dict_len)] + current_string
            current_string = ' ' + current_string
        strings.append(current_string)
    return strings
