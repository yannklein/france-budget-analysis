
import re

def extract_numbers_from_txt(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    return set(map(int, re.findall(r'\b\d+\b', content)))

def extract_numbers_from_csv(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    return set(map(int, re.findall(r'\b\d+\b', content)))

def find_missing_number(txt_numbers, csv_numbers):
    for number in sorted(txt_numbers):
        if number not in csv_numbers:
            return number
    return None

if __name__ == "__main__":
    txt_filepath = "/Users/kleinyann/github/BudgetHorizon/account_name.txt"
    csv_filepath = "/Users/kleinyann/github/BudgetHorizon/account_name.csv"

    txt_numbers = extract_numbers_from_txt(txt_filepath)
    csv_numbers = extract_numbers_from_csv(csv_filepath)

    missing_number = find_missing_number(txt_numbers, csv_numbers)
    if missing_number is not None:
        print(f"The first number in the txt file that is not in the csv file is: {missing_number}")
    else:
        print("All numbers in the txt file are present in the csv file.")