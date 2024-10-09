# importar as bibliotecas
import os
import cv2
import numpy as np

def captura(largura, altura):
    #classificadores
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')
    #abre camera
    camera = cv2.VideoCapture(0) #0 camera padrão do notebook

    #amostrar da imegem do usuario
    amostra = 1    
    n_amostras =25
    
    # recebe o id do Usuário
    id = input('Digite o ID do Usuário: ')

    #mensagem indicando captura de imagens
    print("Capturando as imagens ...")

    #loop
    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        print(np.average(imagem_cinza))
        faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150,150))

        # identifica a geometria das faces
        for (x, y, l, a) in faces_detectadas:
            cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0,255), 2)
            regiao = imagem[y:y + a, x:x +l]
            regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho)
        # Identifica a geometria dos olhos
            for (ox, oy,ol, oa) in olhos_detectados:
                cv2.rectangle(regiao, (ox, oy), (ox+ol, oy+oa), (0, 255, 0), 2)
                '''if cv2.waitKey(1) and 0xFF == ord("c"):
                    if np.average(imagem_cinza) > 110:
                        while amostra <= n_amostras:
                            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                            cv2.imwrite(f'fotos/pessoa.{str(id)}.{str(amostra)}.jpg', imagem_face)
                            print(f'[foto] {str(amostra)} capturada com sucesso!')
                            amostra += 1
                            '''
            if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite(f'fotos/pessoa.{str(id)}.{str(amostra)}.jpg', imagem_face)
                print(f'[foto] {str(amostra)} capturada com sucesso.')
                amostra += 1
            # NOTE: FUNCIONA COM O "C"
            
        cv2.imshow("Detectar Faces", imagem)
        cv2.waitKey(1)

        if (amostra >= n_amostras + 1):
            print("Faces capturadas.")
            break
        elif cv2.waitKey(1) == ord("q"):
            print("Câmera encerrada!")
            break

    # encerra a captura
    camera.release()
    cv2.destroyAllWindows()
    # fim da função
def get_imagem_com_id():
    caminhos = [os.path.join("fotos", f) for f in os.listdir("fotos")]
    faces = []
    ids = []

    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)

    return np.array(ids), faces

def treinamento():
    #cria os elementos de reconhecimento necessário
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()

    ids, faces = get_imagem_com_id()

    #trinando o algoritmo do programa
    print("Treinando... ")
    eigenface.train(faces, ids)    
    eigenface.write("classificadorEigen.yml")
    fisherface.train(faces, ids)    
    fisherface.write("classificadorFisher.yml")
    lbph.train(faces, ids)
    lbph.write("classificadorLBPH.yml")

    #finaliza o trinamento
    print("Treinamento finalizado com sucesso!")

def reconhecedor_eigenfaces(largura, altura):
    detector_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classificadorEigen.yml")
    fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

    camera = cv2.VideoCapture(0)
    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces_detectadas = detector_faces.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30,30))

        for (x, y, l, a) in faces_detectadas:
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
            id, confianca = reconhecedor.predict(imagem_face)
            cv2.putText(imagem, str(id), (x,y+(a+30)), fonte, 2, (0,0,255))

        #encerra a camera
        cv2.imshow('Reconhecer faces', imagem)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


# programa principal
if __name__ == "__main__":
    # definir o tamanho da câmera
    largura = 220
    altura = 220

    while True:
        # menu
        print('0 - Sair do programa.')
        print('1 - Capturar imagem do usuário.')
        print('2 - Treinar sistema.')
        print('3 - Detecção de faces.')

        opcao = input('Opção desejada: ')
        match opcao:
            case '0':
                print("Programa encerrado!")
                break
            case '1':
                captura(largura, altura)
                continue
            case '2':
                treinamento()
                continue
            case '3':
                reconhecedor_eigenfaces(largura, altura)
                continue 
            case _:
                print("Opção Inválida!")
                continue
