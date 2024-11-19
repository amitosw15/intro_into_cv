# %% [markdown]
# # EX 1.1- basic python: Pyramid case
# Implement a function that get a string input and outputs the same word, only each odd char is lower
# case and each even letter is upper case
# You can assume that the input is a valid string which contains only english letters.
# %%


def pyramid_case(in_word):
    # TODO: return the pyramid case word.
    result = ''
    for index, c in enumerate(in_word):
        if index % 2 == 0:  # even index (odd position)
            result += c.lower()
        else:  # odd index (even position)
            result += c.upper()
    return result
# %%


def pyramid_case_one_liner(in_word):
    # TODO: ~~~BONUS~~~
    # return the pyramid case word in one line of code inside the function.
    # DO NOT USE ";" IN YOUR CODE.
    return ''.join(c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(in_word)) if in_word else ''


# %%
# test functions here
input_words = ["hello", "world", "", "I", "am", "LEARNING", "Python"]

print("==== pyramid_case() results:")
for word in input_words:
    print(pyramid_case(word))

print("\n==== pyramid_case_one_liner() results:")
for word in input_words:
    print(pyramid_case_one_liner(word))


# %%
