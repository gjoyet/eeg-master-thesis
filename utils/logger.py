def log(statement: str) -> None:
    delimiter = '-' * (len(statement) + 8)
    print('\n{}\nLOGGER: {}\n{}\n'.format(delimiter, statement, delimiter))
