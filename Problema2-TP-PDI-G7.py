#--------------------------------------
# PROBLEMA 2 : Validación de formulario
#--------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv 

# Tipo de cada formulario :
tipos_formularios = {
    "formulario_01.png": "A",
    "formulario_02.png": "A",
    "formulario_03.png": "A",
    "formulario_04.png": "B",
    "formulario_05.png": "B"
} 

# --------------------------------------
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# --------------------------------------
def imshow(img, titulo="Imagen", color=False):
    plt.figure(figsize=(10,5))
    if color:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(titulo)
    plt.axis("off")
    plt.show()

# --------------------------------------
# FUNCIONES PARA DETECCIÓN DE LÍNEAS
# --------------------------------------
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

def filtrar_lineas_verticales_por_longitud(img_bin, columnas, fraccion_minimma_long=0.8):
    """
   Filtra columnas verticales muy cortas para ignorar marcas.
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

# --------------------------------------
# FUNCIONES PARA EXTRAER CELDAS
# --------------------------------------
def extraer_celda_interior(img, fila_inicio, fila_fin, col_inicio, col_fin, margen=10):
    """
    Recorta la celda de la imagen evitando los bordes de líneas.
    margen: pixeles a recortsr desde los bordes para evitar las líneas
    """
    celda = img[fila_inicio+margen:fila_fin -margen, col_inicio+margen:col_fin-margen]
    if celda.size == 0:
        return celda
    return celda

def celda_con_contenido(celda, umbral_relativo=0.002):
    """
    Devuelve True si hay escritura o marca visible en la celda.
    umbral_relativo: fracción mínima de píxeles blancos respecto al total.
    """
    if celda.size == 0:
        return False
    #Contamos píxeles blancos
    blancos = np.sum(celda == 255)
    rel_area = blancos / celda.size
    return rel_area > umbral_relativo

# --------------------------------------
# FUNCIONES DE VALIDACIÓN DE PREGUNTAS
# --------------------------------------
def contar_marcas_preguntas(celda, umbral_area=20):
    """
    Cuenta cuántas marcas hay en la celda según contornos detectados.
    Ignora contornos muy pequeños (< umbral_area).
    """
    if celda.size == 0:
        return 0
    contornos, _ = cv2.findContours(celda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #filtramos contornos muy pequeños
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > umbral_area]
    return len(contornos_validos)

def validar_pregunta_completa(celda, numero, umbral_area=20):
    """
    Valida la celda de una pregunta según cantidad de marcas detectadas.
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

# --------------------------------------
# FUNCIONES PARA VALIDACIÓN DE TEXTO
# --------------------------------------
def contar_componentes(celda_bin, area_minima=15):
    """
    Cuenta las componentes conectadas en una celda para estimar si hay texto.
    """
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
    con área mayor que 'area_min'.
    """
    _, binaria = cv2.threshold(celda_bin, 128, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    cajas = [stats[i] for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > area_min]
    return cajas

def hay_espacio(cajas, factor_espacio=1.25):
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

    # se considera "espacio" si hay un gap mucho mayor que el ancho promedio de las letras
    anchos = [c[2] for c in cajas]
    ancho_prom = np.median(anchos)
    return any(g > factor_espacio * ancho_prom for g in espacios)

# --------------------------------------
# FUNCIONES DE VALIDACIÓN DE CAMPOS
# --------------------------------------
def validar_nombre_apellido(celda):
    cajas = cajas_componentes(celda)
    n_letras = len(cajas)
    espacio = hay_espacio(cajas,0.5)
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
    # Factor de espacio razonable, y guiones bajos ignorados en hay espacio
    espacio = hay_espacio(cajas, 3.0)
    if espacio:
        return "MAL"
    if 5 <= n <= 23:
        return "OK"
    else:
        return "MAL"

def validar_legajo(celda):
    n = len(cajas_componentes(celda))
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

# -------------------------------------------
# FUNCION PRINCIPAL PARA PROCESAR FORMULARIOS
# -------------------------------------------
def procesar_formularios(tipo_seleccionado="TODOS", mostrar_imagen=True):
    """
    Procesa los formularios del tipo indicado.
    Genera un CSV con los resultados individuales
    y una única imagen resumen general con cada formulario
    indicando si completó bien o mal.
    """

    formularios = [f for f, t in tipos_formularios.items()
                   if tipo_seleccionado == "TODOS" or t == tipo_seleccionado]
    resultados = []

    print("\n INICIANDO VALIDACIÓN DE FORMULARIOS...")
    print("=" * 60)

    for archivo in formularios:
        img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se encontró {archivo}")
            continue

        # Binarización
        img_binaria = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Detección de líneas 
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

        # Recorte de celdas
        celda_nombre = extraer_celda_interior(img_binaria, filas_compactas[1], filas_compactas[2], columnas_l[1], columnas_l[2])
        celda_edad   = extraer_celda_interior(img_binaria, filas_compactas[2], filas_compactas[3], columnas_l[1], columnas_l[2])
        celda_mail   = extraer_celda_interior(img_binaria, filas_compactas[3], filas_compactas[4], columnas_l[1], columnas_l[2])
        celda_legajo = extraer_celda_interior(img_binaria, filas_compactas[4], filas_compactas[5], columnas_l[1], columnas_l[2])
        celda_coment = extraer_celda_interior(img_binaria, filas_compactas[9], filas_compactas[10], columnas_l[1], columnas_l[2])
        preg1 = extraer_celda_interior(img_binaria, filas_compactas[6], filas_compactas[7], columnas_l[1], columnas_l[2])
        preg2 = extraer_celda_interior(img_binaria, filas_compactas[7], filas_compactas[8], columnas_l[1], columnas_l[2])
        preg3 = extraer_celda_interior(img_binaria, filas_compactas[8], filas_compactas[9], columnas_l[1], columnas_l[2])

        # --- Validaciones ---
        nombre_ok = validar_nombre_apellido(celda_nombre)
        edad_ok = validar_edad(celda_edad)
        mail_ok = validar_mail(celda_mail)
        legajo_ok = validar_legajo(celda_legajo)
        coment_ok = validar_comentarios(celda_coment)

        def resultado_pregunta(celda):
            if celda.size == 0:
                return "MAL"
            n = contar_marcas_preguntas(celda)
            return "OK" if n == 1 else "MAL"

        preg1_r = resultado_pregunta(preg1)
        preg2_r = resultado_pregunta(preg2)
        preg3_r = resultado_pregunta(preg3)

        # Guardar resultados en memoria
        resultados.append({"Formulario": archivo,"Nombre y Apellido": nombre_ok,"Edad": edad_ok,"Mail": mail_ok,"Legajo": legajo_ok,"Comentarios": coment_ok,"Pregunta 1": preg1_r,"Pregunta 2": preg2_r,"Pregunta 3": preg3_r})

    nombres_recortados = []
    estados = []

    for res in resultados:
        archivo = res["Formulario"]
        img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
        # --- Recorte de la celda de nombre ---
        img_binaria = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 8)
        img_th_ones = (img_binaria > 0).astype(np.uint8)
        sum_filas = np.sum(img_th_ones, axis=1)
        sum_columnas = np.sum(img_th_ones, axis=0)
        filas = np.where(sum_filas / np.max(sum_filas) > 0.5)[0]
        columnas = np.where(sum_columnas / np.max(sum_columnas) > 0.3)[0]
        filas_compactas = compactar_lineas(filas, 30)
        columnas_compactas = compactar_lineas(columnas, 60)
        columnas_l = filtrar_lineas_verticales_por_longitud(img_binaria, columnas_compactas, 0.85)

        celda_nombre = extraer_celda_interior(img_binaria, filas_compactas[1], filas_compactas[2],
                                              columnas_l[1], columnas_l[2])
        nombres_recortados.append(celda_nombre)

        # Determinar si completó bien o mal 
        valores = [v for k, v in res.items() if k != "Formulario"]
        completo_bien = all(v == "OK" for v in valores)
        estado = "completo bien" if completo_bien else "completo mal"
        color = (0, 150, 0) if completo_bien else (0, 0, 255)
        estados.append((estado, color))

    # Calcular tamaño de imagen final 
    alturas = [c.shape[0] for c in nombres_recortados]
    anchos = [c.shape[1] for c in nombres_recortados]
    alto_total = sum(alturas) + 80 * len(nombres_recortados) + 80
    ancho_max = max(anchos) + 300

    resumen = np.ones((alto_total, ancho_max, 3), dtype=np.uint8) * 255
    y = 60

    # Dibujar filas 
    for celda, (estado, color) in zip(nombres_recortados, estados):
        celda_bgr = cv2.cvtColor(celda, cv2.COLOR_GRAY2BGR)
        h, w = celda_bgr.shape[:2]
        resumen[y:y+h, 20:20+w] = celda_bgr

        cv2.putText(resumen, estado, (w + 60, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y += h + 60

    cv2.putText(resumen, "Resumen general de formularios",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Mostrar y guardar 
    if mostrar_imagen:
        imshow(cv2.cvtColor(resumen, cv2.COLOR_BGR2RGB), "Resumen de formularios")
        cv2.imwrite("resumen_resultados.png", resumen)
        print("\n Imagen final generada: resumen_resultados.png")

    return resultados

# --------------------------------------
# EJECUCIÓN
# --------------------------------------
if __name__ == "__main__":
    tipo = input("Ingrese tipo de formulario (A/B/C/TODOS): ").upper()
    if tipo not in ["A", "B", "C", "TODOS"]:
        tipo = "TODOS"

    resultados = procesar_formularios(tipo)

    # Mostrar por campo, en orden
    for res in resultados:
        print(f"\n--- {res['Formulario']} ---")
        for campo, valor in res.items():
            if campo != "Formulario":
                print(f"{campo}: {valor}")

    resultados_todos = procesar_formularios("TODOS", mostrar_imagen=False)
    
    #CSV de todos
    columnas_csv = ["Formulario", "Nombre y Apellido", "Edad", "Mail",
                    "Legajo", "Comentarios", "Pregunta 1", "Pregunta 2", "Pregunta 3"]

    with open("resultados_formularios.csv", mode="w", newline="", encoding="utf-8") as file:
        import csv
        writer = csv.DictWriter(file, fieldnames=columnas_csv)
        writer.writeheader()
        writer.writerows(resultados_todos)

    print("\n CSV con todos los formularios generado correctamente: resultados_formularios.csv")

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
