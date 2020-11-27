import datetime

def write_anagrams_palindromes_to_file(original_word, anagram_list, palindromes_list):
    file = original_word + " -" + datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    filename = '%s.txt.'%file

    with open(filename, "w") as write_ana_pali:
        write_ana_pali.write("anagrams: \n")
        for anagram in anagram_list:
            write_ana_pali.write(anagram+"\n")
        write_ana_pali.write(" \n")
        write_ana_pali.write("palindromes: \n")
        for palindromes in palindromes_list:
            write_ana_pali.write(palindromes+"\n")

original_word = "test"

anagram_list = ["1", "2", "3"]
palindromes_list = ["8", "9"]
print(datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
write_anagrams_palindromes_to_file(original_word, anagram_list, palindromes_list)
