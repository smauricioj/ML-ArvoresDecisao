# -*- coding: utf-8 -*-

# Autor: Sergio P
# Data: 08/09/2022

# ---------------------------------------------------------------
# IMPORTS

from json import load
from pathlib import Path
from pandas import read_csv
from sys import exit

from algoritmos import ID3, CART

# ---------------------------------------------------------------
# MAIN


def main():
    # Dicionário de configurações
    with open(Path.cwd()/'conf.json',
              encoding='utf-8') as conf_file:
        conf = load(conf_file)

    # Diretórios
    conf['main_dir'] = Path.cwd().parent
    Path(conf['main_dir']/'Dados').mkdir(exist_ok=True)

    # Dados de exemplos
    if conf['dados'] != 'all':
        instancias = [conf['main_dir']/'Dados'/conf['dados']]
    else:
        instancias = [x for x in Path(conf['main_dir']/'Dados').iterdir()]

    for instancia in instancias:
        with open(instancia) as data_file:
            df = read_csv(data_file, encoding='utf-8')

        # Resultados
        out_path = Path(
            conf['main_dir']/'Resultados'/conf['algoritmo']/instancia.stem
        ).with_suffix('.txt')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Procedimentos
        args = (df, df.columns[-1])
        if conf['algoritmo'] == 'ID3':
            dtc = ID3(*args)
        elif conf['algoritmo'] == 'CART':
            dtc = CART(*args)
        dtc.run()
        with open(out_path, 'w') as out_file:
            out_file.writelines(dtc.__repr__())


if __name__ == '__main__':
    main()
