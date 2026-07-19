from typing import Iterable

class TSVParser:
    def __init__(self):
        self.content: list[tuple[str, ...]] = []

    def readfile(self, 
                 path: str, 
                 columns: tuple[int, ...] = (0,),
                 header: bool = False,
                 encoding: None | str = 'UTF-8') -> list[tuple[str, ...]]:
        if not isinstance(columns, Iterable):
            raise ValueError("`columns` must be either 'all' or a sequence of integers")
        with open(path, 'r', encoding=encoding) as tsv:
            if header == True:
                tsv.readline()
            tmp = tsv.read().split('\n')
            self.content = [tuple(row.split('\t')[c] for c in columns) for row in tmp]
            del tmp
        return self.content


class Tensorifier:
    ...

if __name__ == '__main__':
    tsv_parser = TSVParser()
    tsv_parser.readfile('data/english-swahili.tsv', (1, 3))
    print(*tsv_parser.content[:5], sep='\n')