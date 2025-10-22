# Problema 2 - Trabajo practico - PDI - GRUPO 7
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, titulo="Imagen", color=False):
    plt.figure(figsize=(10,5))
    if color:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(titulo)
    plt.axis("off")
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------PARTE A-----------------
img = cv2.imread("formulario_04.png", cv2.IMREAD_GRAYSCALE)
imshow(img, "Imagen original")
#umbralado
img_binaria = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8)
imshow(img_binaria, "Imagen binaria invertida")

#detectamos lineas horizontales
#Convertimos la imagen binaria a tipo entero (por si es booleana)
img_th_ones = img_binaria > 0 
img_th_ones = img_th_ones.astype(np.uint8)

#sumamos los píxeles por filas y columnas
sum_filas = np.sum(img_th_ones, axis=1)   #suma hor
sum_columnas = np.sum(img_th_ones, axis=0)   #suma verti

#Normalizamos 
sum_filas_norm = sum_filas / np.max(sum_filas)
sum_cols_norm = sum_columnas / np.max(sum_columnas)

#definimos umbrales para decidir qué filas/columnas son líneas
umbral_fil = 0.5  #umbral horizo
umbral_col = 0.3   #umbral vert

filas = np.where(sum_filas_norm > umbral_fil)[0]
columnas = np.where(sum_cols_norm > umbral_col)[0]

print(f"filas encontradas: {len(filas)}")
print(f"colums encontradasa: {len(columnas)}")

def compactar_lineas(lineas, distancia_minima=100):
    """
    Agrupa coordenadas cercanas para que cada línea se represente con un solo valor.
    """
    if len(lineas) == 0:
        return np.array([])

    lineas_compactas = [lineas[0]]
    for l in lineas[1:]:
        if l - lineas_compactas[-1] > distancia_minima:
            lineas_compactas.append(l)
    return np.array(lineas_compactas)

filas_compactas = compactar_lineas(filas, distancia_minima=30)
columnas_compactas = compactar_lineas(columnas, distancia_minima=60)

#Filtramos por longitud (esto es para que no tenga en cuenta la col de si y no)
def filtrar_lineas_verticales_por_longitud(img_bin, columnas, fraccion_minimma_long=0.8):
    """
    columnas: lista de x detectadas
    fraccion_minima_long: fracción mínima de la longitud máxima para conservar la columna
    """
    if len(columnas) == 0:
        return np.array([])

    longitudes = []
    for x in columnas:
        #contamos píxeles que pertenecen a la línea vertical en esa columna
        col_count = np.sum(img_bin[:, x] == 255)
        longitudes.append(col_count)

    longitud_max = max(longitudes)
    columnas_filtradas = [x for x, l in zip(columnas, longitudes) if l >= fraccion_minimma_long * longitud_max]
    return np.array(columnas_filtradas)


columnas_l = filtrar_lineas_verticales_por_longitud(img_binaria, columnas_compactas,  fraccion_minimma_long=0.85)
print("Columnas tras filtrar por longitud:", columnas_l)

img_debug = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

for y in filas_compactas:
    cv2.line(img_debug, (0, y), (img_debug.shape[1], y), (255, 0, 0), 1)  
for x in columnas_l:
    cv2.line(img_debug, (x, 0), (x, img_debug.shape[0]), (0, 0, 255), 1)  
imshow(img_debug, "LINEAS DETECTADAS  rojo=vertical, azul=horizontal", color=True)

#recortamos celdas por celda
def extraer_celda_interior(img, fila_inicio, fila_fin, col_inicio, col_fin, margen=10):
    """"
    margen: pixeles a recortsr desde los bordes para evitar las líneas
    """
    celda = img[fila_inicio+margen:fila_fin -margen, col_inicio+margen:col_fin-margen]
    if celda.size == 0:
        return celda
    return celda

#para ver si la celda esta vacia
def celda_con_contenido(celda, umbral_relativo=0.002):
    """
    Devuelve True si hay escritura o marca visible en la celda.
    """
    if celda.size == 0:
        return False
    #Contamos píxeles blancos
    blancos = np.sum(celda == 255)
    rel_area = blancos / celda.size
    return rel_area > umbral_relativo

#deteccion de marcas en la parte de las preguntas
def contar_marcas_preguntas(celda, umbral_area=20):
    """
    Cuenta cuantas marcas hay en la celda s/ contornos detectados.
   ignora contornos muy pequeños (< umbral_area).
    """
    if celda.size == 0:
        return 0
    contornos, _ = cv2.findContours(celda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #filtramos contornos muy pequeños
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > umbral_area]
    return len(contornos_validos)

#verificar si tiene ninguna, un o dos marcas cada pregunta para ver si esta bien o mal
def validar_pregunta_completa(celda, numero, umbral_area=20):
    """
    Valida la celda de una pregunta según cantidad de marcas:
    """
    if celda.size == 0 or celda.size == 1:
        print(f"> Pregunta {numero}: CELDA VACIA")
        return
    
    nro_marcas = contar_marcas_preguntas(celda, umbral_area)
    
    if nro_marcas == 1:
        print(f"> Pregunta {numero}: BIEN (1 marca detectada)")
    elif nro_marcas == 0:
        print(f"> Pregunta {numero}: MAL (sin marca)")
    else:
        print(f"> Pregunta {numero}: MAL ({nro_marcas} marcas detectadas)")

#Validacion de los campos
def contar_componentes(celda_bin, area_minima=15):
    #asegurar formato bin
    _, binaria = cv2.threshold(celda_bin, 128, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    
    #filtrar pequeñas manchas
    areas_validas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > area_minima]
    
    return len(areas_validas)

def validar_celda_texto(celda, minimo=3, maximo=30):
    """
    Devuelve True si la cantidad de trazos está entre los límites esperados.
    """
    n = contar_componentes(celda)
    return minimo <= n <= maximo

def cajas_componentes(celda_bin, area_min=35):
    """
    Devuelve una lista de cajas [x, y, w, h, area] de las componentes conectadas
    con area > que area_min
    """
    _, binaria = cv2.threshold(celda_bin, 128, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    cajas = [stats[i] for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > area_min]
    return cajas

def hay_espacio(cajas, factor_espacio):
    """
    Detecta si hay un espacio horizontal significativo entre grupos de trazos.
    """
    if len(cajas) < 2:
        return False

    cajas = sorted(cajas, key=lambda c: c[0])
    espacios = []
    for i in range(1, len(cajas)):
        x_prev, _, w_prev, _, _ = cajas[i - 1]
        x_cur, _, _, _, _ = cajas[i]
        gap = x_cur - (x_prev + w_prev)
        espacios.append(gap)

    #se considera "espacio" si hay un gap mucho mayor que el ancho promedio de las letras
    anchos = [c[2] for c in cajas]
    ancho_prom = np.median(anchos)
    print(ancho_prom)
    print(anchos)
    return any(g > factor_espacio * ancho_prom for g in espacios)
 
def validar_nombre_apellido(celda):
    cajas = cajas_componentes(celda)
    n_letras = len(cajas)
    espacio = hay_espacio(cajas,0.5)
    print(espacio)
    if 5 <= n_letras <= 25 and espacio:
        return "OK"
    else:
        return "MAL"

def validar_edad(celda):
    cajas = cajas_componentes(celda)
    n_digitos = len(cajas)
    espacio = hay_espacio(cajas,0.5)
    if 2 <= n_digitos <= 3 and not espacio:
        return "OK"
    else:
        return "MAL"

def validar_mail(celda):
    cajas = cajas_componentes(celda)
    n = len(cajas)
    print(n)
    espacio = hay_espacio(cajas, 3.0)
    if espacio:
        print("hay espacio")
        return "MAL"
    if 5 <= n <= 23:
        return "OK"
    else:
        return "MAL"

def validar_legajo(celda):
    n = len(cajas_componentes(celda))
    print(n)
    if 6 <= n <= 7:
        return "OK"
    else:
        return "MAL"

def validar_comentarios(celda):
    n = len(cajas_componentes(celda))
    if 1 <= n <= 25:
        return "OK"
    else:
        return "MAL"
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------PARTE B------   
#detectar que tipo de formulario es: A, B o C.
formularios=["formulario_01.png","formulario_02.png","formulario_03.png","formulario_04.png","formulario_05.png"]
for archivo in formularios:
    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    img_binaria = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8)

    img_th_ones = (img_binaria > 0).astype(np.uint8)
    sum_filas = np.sum(img_th_ones, axis=1)
    sum_columnas = np.sum(img_th_ones, axis=0)
    sum_filas_norm = sum_filas / np.max(sum_filas)
    sum_cols_norm = sum_columnas / np.max(sum_columnas)

    filas = np.where(sum_filas_norm > 0.5)[0]
    columnas = np.where(sum_cols_norm > 0.3)[0]
    filas_compactas = compactar_lineas(filas, distancia_minima=30)
    columnas_compactas = compactar_lineas(columnas, distancia_minima=60)
    columnas_l = filtrar_lineas_verticales_por_longitud(img_binaria, columnas_compactas, fraccion_minimma_long=0.85)

    #recortamos el encabezado
    encabezado = extraer_celda_interior(img_binaria, filas_compactas[0], filas_compactas[1], columnas_l[0], columnas_l[2], margen=4)
    imshow(encabezado, f"Encabezado de {archivo}")
    alto, ancho = encabezado.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(encabezado, connectivity=8, ltype=cv2.CV_32S)

    #Filtrar por areamin
    th_area = 30
    stats_filtradas = stats[stats[:, cv2.CC_STAT_AREA] > th_area]

    #fiiltrar por ancho (descartar lineas verticales)
    min_width = max(8, ancho * 0.01)
    stats_filtradas = stats_filtradas[stats_filtradas[:, cv2.CC_STAT_WIDTH] > min_width]
  
    indice_ultima_comp = np.argmax(stats_filtradas[:, cv2.CC_STAT_LEFT])
    x, y, w, h, area = stats_filtradas[indice_ultima_comp]
    #margen
    margen = 2
    y1 = max(0, y - margen)
    y2 = min(y + h + margen, encabezado.shape[0])
    x1 = max(0, x - margen)
    x2 = min(x + w + margen, encabezado.shape[1])

    ultima_letra = encabezado[y1:y2, x1:x2]
    imshow(ultima_letra, "Letra detectada (A/B/C)")

    #calculo de la densidad de pixeles
    pixeles_blancos = np.sum(ultima_letra == 255)
    porcentaje_blancos = pixeles_blancos / ultima_letra.size
    print(f"Porcentaje de blancos: {porcentaje_blancos:.3f}")

    if 0.2 < porcentaje_blancos < 0.27:
        tipo = "A"
    elif  porcentaje_blancos > 0.35:
        tipo = "B"
    else:
        tipo = "C"
    print(f"Tipo de formulario detectado: {tipo}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------PARTE C-------
#imagen de quien completo mal y quien bien.
formularios = ["formulario_01.png","formulario_02.png","formulario_03.png","formulario_04.png","formulario_05.png"]
resultados = []
nombres_recortados = []
for archivo in formularios:
    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)

    #umbralado
    img_binaria = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8 )

    #deteccion de lineas
    img_th_ones = img_binaria > 0
    img_th_ones = img_th_ones.astype(np.uint8)
    sum_filas = np.sum(img_th_ones, axis=1)
    sum_columnas = np.sum(img_th_ones, axis=0)
    sum_filas_norm = sum_filas / np.max(sum_filas)
    sum_cols_norm = sum_columnas / np.max(sum_columnas)
    filas = np.where(sum_filas_norm > 0.5)[0]
    columnas = np.where(sum_cols_norm > 0.3)[0]
    filas_compactas = compactar_lineas(filas, distancia_minima=30)
    columnas_compactas = compactar_lineas(columnas, distancia_minima=60)
    columnas_l = filtrar_lineas_verticales_por_longitud(img_binaria, columnas_compactas, fraccion_minimma_long=0.85)
    
    celda_nombre = extraer_celda_interior(img_binaria, filas_compactas[1], filas_compactas[2], columnas_l[1], columnas_l[2])
    preg1 = extraer_celda_interior(img_binaria, filas_compactas[6], filas_compactas[7], columnas_l[1], columnas_l[2])
    preg2 = extraer_celda_interior(img_binaria, filas_compactas[7], filas_compactas[8], columnas_l[1], columnas_l[2])
    preg3 = extraer_celda_interior(img_binaria, filas_compactas[8], filas_compactas[9], columnas_l[1], columnas_l[2])
    n1 = contar_marcas_preguntas(preg1)
    n2 = contar_marcas_preguntas(preg2)
    n3 = contar_marcas_preguntas(preg3)

    correcto = (n1 == 1 and n2 == 1 and n3 == 1)

    nombres_recortados.append(celda_nombre)
    resultados.append(correcto)

alturas = [c.shape[0] for c in nombres_recortados]
anchos = [c.shape[1] for c in nombres_recortados]
alto_total = sum(alturas) + 50 * len(nombres_recortados)
ancho_max = max(anchos) + 250

resumen = np.ones((alto_total, ancho_max), dtype=np.uint8) * 255

y = 30 #indicar en q posicion se debe pegar cada recorte de nombre en la img nueva
for i, (celda, ok) in enumerate(zip(nombres_recortados, resultados)):
    h, w = celda.shape
    resumen[y:y+h, 20:20+w] = celda

    texto = "completo bien" if ok else "completo mal"
    color = (0) if ok else (128)
    cv2.putText(resumen, texto, (w + 60, y + h // 2),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y += h + 25

imshow(resumen, "Resumen de formularios completados")
cv2.imwrite("resumen_resultados.png", resumen)
#----------------------------------------------------------------------------------------------------------------------------------------------------------PARTE D------------
tipos_formularios = {"formulario_01.png": "A","formulario_02.png": "B","formulario_03.png": "A","formulario_04.png": "B","formulario_05.png": "B"}

formularios = list(tipos_formularios.keys())
resultados = []

for archivo in formularios:
    tipo = tipos_formularios[archivo]

    img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    img_binaria = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8)

    img_th_ones = (img_binaria > 0).astype(np.uint8)
    sum_filas = np.sum(img_th_ones, axis=1)
    sum_columnas = np.sum(img_th_ones, axis=0)
    sum_filas_norm = sum_filas / np.max(sum_filas)
    sum_cols_norm = sum_columnas / np.max(sum_columnas)

    filas = np.where(sum_filas_norm > 0.5)[0]
    columnas = np.where(sum_cols_norm > 0.3)[0]
    filas_compactas = compactar_lineas(filas, distancia_minima=30)
    columnas_compactas = compactar_lineas(columnas, distancia_minima=60)
    columnas_l = filtrar_lineas_verticales_por_longitud(img_binaria, columnas_compactas, fraccion_minimma_long=0.85)

    celda_nombre = extraer_celda_interior(img_binaria, filas_compactas[1], filas_compactas[2], columnas_l[1], columnas_l[2])
    celda_edad   = extraer_celda_interior(img_binaria, filas_compactas[2], filas_compactas[3], columnas_l[1], columnas_l[2])
    celda_mail   = extraer_celda_interior(img_binaria, filas_compactas[3], filas_compactas[4], columnas_l[1], columnas_l[2])
    celda_legajo = extraer_celda_interior(img_binaria, filas_compactas[4], filas_compactas[5], columnas_l[1], columnas_l[2])
    celda_coment = extraer_celda_interior(img_binaria, filas_compactas[9], filas_compactas[10], columnas_l[1], columnas_l[2])
    preg1 = extraer_celda_interior(img_binaria, filas_compactas[6], filas_compactas[7], columnas_l[1], columnas_l[2])
    preg2 = extraer_celda_interior(img_binaria, filas_compactas[7], filas_compactas[8], columnas_l[1], columnas_l[2])
    preg3 = extraer_celda_interior(img_binaria, filas_compactas[8], filas_compactas[9], columnas_l[1], columnas_l[2])

    nombre_ok = validar_nombre_apellido(celda_nombre)
    edad_ok = validar_edad(celda_edad)
    mail_ok = validar_mail(celda_mail)
    legajo_ok = validar_legajo(celda_legajo)
    coment_ok = validar_comentarios(celda_coment)

    def resultado_pregunta(celda):
        if celda.size == 0:
            return "MAL"
        n = contar_marcas_preguntas(celda)
        if n == 1:
            return "OK"
        else:
            return "MAL"

    preg1_r = resultado_pregunta(preg1)
    preg2_r = resultado_pregunta(preg2)
    preg3_r = resultado_pregunta(preg3)

    print(f"Nombre y Apellido: {nombre_ok}")
    print(f"Edad: {edad_ok}")
    print(f"Mail: {mail_ok}")
    print(f"Legajo: {legajo_ok}")
    print(f"Comentarios: {coment_ok}")
    print(f"Preguntas: {preg1_r}, {preg2_r}, {preg3_r}")

    resultados.append({ "Formulario": archivo, "Tipo": tipo, "Nombre y Apellido": nombre_ok, "Edad": edad_ok, "Mail": mail_ok, "Legajo": legajo_ok,"Pregunta 1": preg1_r,"Pregunta 2": preg2_r,"Pregunta 3": preg3_r, "Comentarios": coment_ok})

with open("resultados_formularios.csv", mode="w", newline="", encoding="utf-8") as file:
    columnas_csv = ["Formulario", "Tipo", "Nombre y Apellido", "Edad", "Mail", "Legajo", "Pregunta 1", "Pregunta 2", "Pregunta 3","Comentarios"]
    writer = csv.DictWriter(file, fieldnames=columnas_csv)
    writer.writeheader()
    writer.writerows(resultados)