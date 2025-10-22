#TP_PDI - GRUPO 7
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PROBLEMA 1: Ecualización local de histograma

def ecualizacion_local(img, M, N): 
    filas, columnas = img.shape

    #agrego bordes 
    exp_y = M // 2
    exp_x = N // 2
    img_exp = cv2.copyMakeBorder(img, exp_y, exp_y, exp_x, exp_x, cv2.BORDER_REPLICATE)
    
    img_salida = np.zeros_like(img, dtype=np.uint8)
    
    for i in range(filas):
        for j in range(columnas):
            #vecindario local
            ventana = img_exp[i:i+M, j:j+N]
            
            #histograma local
            hist, _ = np.histogram(ventana, bins=256, range=(0,256))
            hist = hist / (M*N)
            suma_acumulada = np.cumsum(hist)
            
            #nuevo valor del pixel central
            valor_nuevo = img_exp[i+exp_y, j+exp_x]
            img_salida[i, j] = np.uint8(255 * suma_acumulada[valor_nuevo])
    
    return img_salida

imagen = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

resultado_local_3x3 = ecualizacion_local(imagen, 3, 3)
plt.figure(), plt.imshow(resultado_local_3x3, cmap='gray'), plt.show(block=False)

resultado_local_25x25 = ecualizacion_local(imagen, 25, 25)
plt.figure(), plt.imshow(resultado_local_25x25, cmap='gray'), plt.show(block=False)

resultado_local_9x9=ecualizacion_local(imagen,9,9)
plt.figure(), plt.imshow(resultado_local_9x9, cmap='gray'), plt.show(block=False)

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')


plt.subplot(2,2,2)
plt.title('Ecualización local (3x3)')
plt.imshow(resultado_local_3x3, cmap='gray')


plt.subplot(2,2,3)
plt.title('Ecualización local (9x9)')
plt.imshow(resultado_local_9x9, cmap='gray')

plt.subplot(2,2,4)
plt.title('Ecualización local (20x20)')
plt.imshow(resultado_local_25x25, cmap='gray')
plt.show()