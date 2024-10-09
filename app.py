import os
import cv2
import numpy as np
import flet as ft

# Funções de captura, treinamento e reconhecimento
def captura(largura, altura, id):
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    amostra = 1    
    n_amostras = 25
    msg = "Capturando as imagens..."

    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150,150))

        for (x, y, l, a) in faces_detectadas:
            cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)
            if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite(f'fotos/pessoa.{str(id)}.{str(amostra)}.jpg', imagem_face)
                msg += f'\n[foto] {str(amostra)} capturada com sucesso!'
                amostra += 1

        cv2.imshow("Detectar Faces", imagem)
        cv2.waitKey(1)

        if amostra >= n_amostras + 1:
            msg += "\nFaces capturadas."
            break

    camera.release()
    cv2.destroyAllWindows()
    return msg

def get_imagem_com_id():
    caminhos = [os.path.join("fotos", f) for f in os.listdir("fotos")]
    faces, ids = [], []

    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)

    return np.array(ids), faces

def treinamento():
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()
    ids, faces = get_imagem_com_id()
    
    eigenface.train(faces, ids)    
    eigenface.write("classificadorEigen.yml")
    fisherface.train(faces, ids)    
    fisherface.write("classificadorFisher.yml")
    lbph.train(faces, ids)
    lbph.write("classificadorLBPH.yml")
    
    return "Treinamento finalizado com sucesso!"

def reconhecedor_eigenfaces(largura, altura):
    detector_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classificadorEigen.yml")
    
    camera = cv2.VideoCapture(0)
    msg = "Reconhecimento iniciado..."

    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces_detectadas = detector_faces.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30,30))

        for (x, y, l, a) in faces_detectadas:
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
            id, confianca = reconhecedor.predict(imagem_face)
            cv2.putText(imagem, str(id), (x,y+(a+30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

        cv2.imshow('Reconhecer faces', imagem)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    return msg

# Função principal para interface com Flet
def main(page: ft.Page):
    page.title = "Reconhecimento Facial"
    
    def on_capture_click(e):
        id = id_input.value
        if id:
            message = captura(largura=220, altura=220, id=id)
            message_box.value = message
            page.update()
        else:
            message_box.value = "Por favor, insira um ID válido."
            page.update()

    def on_train_click(e):
        message = treinamento()
        message_box.value = message
        page.update()

    def on_recognize_click(e):
        message = reconhecedor_eigenfaces(largura=220, altura=220)
        message_box.value = message
        page.update()

    id_input = ft.TextField(label="ID do Usuário", autofocus=True)
    capture_button = ft.ElevatedButton(text="Capturar Imagem", on_click=on_capture_click)
    train_button = ft.ElevatedButton(text="Treinar Sistema", on_click=on_train_click)
    recognize_button = ft.ElevatedButton(text="Reconhecer Faces", on_click=on_recognize_click)
    message_box = ft.TextField(label="Mensagens", multiline=True, read_only=True)

    page.add(id_input, capture_button, train_button, recognize_button, message_box)

# Programa principal
if __name__ == "__main__":
    ft.app(target=main)
