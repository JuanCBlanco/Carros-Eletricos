import requests
import shutil

def main():
    #  Variáveis do download
    PATH = '../Dados/Bruto/dados.zip'
    URL = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tb9yrptydn-2.zip'
    
    #  Donwload de arquivos 
    with requests.get(URL, stream=True) as r:
        with open(PATH, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    
    #  Descompactação de arquivos e remoção do zip mantendo os demais
    shutil.unpack_archive(PATH, '../Dados/Bruto/')
    shutil.os.remove(PATH)
    
if __name__ == '__main__':
    main()