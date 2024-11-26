def log(statement: str) -> None:
    delimiter = '-' * (len(statement) + 8)
    print('\n{}\nLOGGER: {}\n{}'.format(delimiter, statement, delimiter))
