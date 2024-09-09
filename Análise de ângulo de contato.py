''' Descrição
Projeto de Graduação em Engenharia de Materiais - UFRJ

Título: Análise de Ângulo de Contato em Gotas utilizando Processamento de Imagem

Descrição:
Este código foi desenvolvido como parte do projeto de graduação em Engenharia de Materiais
na Universidade Federal do Rio de Janeiro (UFRJ). Ele visa analisar o ângulo de contato 
de gotas em superfícies utilizando técnicas de processamento de imagem, ajustando elipses 
e retas tangentes para determinar a evolução dos ângulos de contato ao longo do tempo.

Funcionalidades principais:
- Processamento de vídeos ou imagens para detecção de gotas.
- Cálculo do ângulo de contato por meio de ajuste de elipses e retas tangentes.
- Exibição de gráficos da evolução dos ângulos e exportação dos resultados em CSV.

Dependências:
- Python 3.x
- OpenCV
- Matplotlib
- Numpy
- Pandas
- Scipy
- Tkinter (para seleção de arquivos)

Instruções de Uso:
1. Certifique-se de que todas as dependências estejam instaladas.
2. Execute o código e selecione um arquivo de vídeo ou imagem para análise.
3. O código processará o arquivo e exibirá a evolução do ângulo de contato.
4. Ao final do processamento, os resultados serão salvos em um arquivo CSV com um timestamp.

Estrutura do Código:
- abrir_arquivo: Função para abrir um arquivo de vídeo ou imagem.
- ler_frame: Função para leitura de frames de vídeo ou imagem.
- processar_primeiro_frame: Função para processamento inicial da imagem (corte e ajuste).
- calcular_limiar: Função para calcular o limiar de binarização da imagem.
- binarizar_frame: Função para binarizar a imagem com base no limiar calculado.
- extrair_contornos: Função para extrair os contornos da gota na imagem binarizada.
- separar_gota_e_horizonte: Função para separar a gota do horizonte na imagem.
- varredura_horizontal_para_gota: Função para realizar a varredura horizontal na gota.
- ajustar_reta_horizonte: Função para ajustar uma reta aos pontos do horizonte.
- ajustar_elipse: Função para ajustar uma elipse aos pontos da gota.
- calcular_intersecoes_retas_tangentes: Função para calcular as interseções e retas tangentes.
- exibir_imagens_com_intersecoes_e_tangentes: Função para exibir a imagem com os ajustes visuais.
- calcular_angulo_contato: Função para calcular o ângulo de contato com base nas tangentes.
- plotar_grafico_evolucao_angulos: Função para plotar o gráfico da evolução dos ângulos de contato.
- exibir_tabela_angulos: Função para exibir e exportar a tabela com os ângulos calculados.
- main: Função principal para controlar o fluxo do programa.

Autoria:
- Autor: Eduardo Brum Fernandes Pimentel
- Orientador: Felipe Sampaio Alencastro
- Instituição: Universidade Federal do Rio de Janeiro (UFRJ)
- Curso: Engenharia de Materiais
- Data de conclusão: Setembro 2024

Contato:
Para dúvidas ou informações adicionais, entre em contato com o autor:
- eduardopimentel@poli.ufrj.br
- felipesa@metalmat.ufrj.b

Licença:
Este código está disponível para uso público. Você pode utilizá-lo e modificá-lo, desde que mantenha os créditos do autor.
'''

#Importar bibliotecas necessárias
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Configuração inicial
analises_por_segundo = 2  # Quantidade de análises por segundo de vídeo
passo_pixel = 5  # Passo de varredura dos pixels

def abrir_arquivo():
    """
    Abre uma janela para selecionar uma imagem ou vídeo.
    Retorna o objeto de captura de vídeo (se for vídeo), a imagem (se for imagem) e um flag indicando o tipo de arquivo.
    """
    root = Tk()
    root.withdraw()
    
    arquivo = askopenfilename(title="Selecione uma imagem ou vídeo", 
                              filetypes=[("Todos os arquivos", "*.*"), 
                                         ("Imagens", "*.png;*.jpg;*.jpeg"), 
                                         ("Vídeos", "*.mp4;*.avi")])
    
    extensao = os.path.splitext(arquivo)[1].lower()
    
    if extensao in [".png", ".jpg", ".jpeg"]:
        imagem = cv2.imread(arquivo)
        if imagem is None:
            raise ValueError("Erro ao abrir a imagem.")
        return imagem, None, "imagem"
    elif extensao in [".mp4", ".avi"]:
        video = cv2.VideoCapture(arquivo)
        if not video.isOpened():
            raise ValueError("Erro ao abrir o vídeo.")
        return None, video, "video"
    else:
        raise ValueError("Formato de arquivo não suportado. Selecione uma imagem ou um vídeo.")

def ler_frame(imagem, video, tipo_arquivo):
    """
    Função que lê o frame da imagem ou vídeo. 
    Se for uma imagem, retorna a própria imagem. Se for um vídeo, retorna o frame atual.
    """
    if tipo_arquivo == "imagem":
        return imagem
    elif tipo_arquivo == "video":
        ret, frame = video.read()
        if not ret:
            return None  # Retorna None se não houver mais frames
        return frame

def processar_primeiro_frame(frame):
    """
    Processa o primeiro frame de uma imagem ou vídeo.
    - Converte para escala de cinza
    - Remove 10% das bordas laterais
    - Corta a imagem horizontalmente no ponto mais claro, mantendo a parte inferior da imagem
    """
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    largura = frame_cinza.shape[1]
    corte_larg = int(largura * 0.10)
    frame_cortado_lateral = frame_cinza[:, corte_larg:-corte_larg]

    ponto_claro = np.argmax(np.sum(frame_cortado_lateral, axis=1))
    frame_cortado_horizontal = frame_cortado_lateral[ponto_claro:, :]
    
    return frame_cortado_horizontal

def calcular_limiar(frame):
    """
    Calcula o limiar de binarização baseado no primeiro frame.
    - Normaliza os valores de pixel para o intervalo [0, 1]
    - Gera o histograma de intensidade de pixel, suaviza e encontra o limiar mínimo.
    """
    # Normalizar o frame para o intervalo [0, 1]
    frame_norm = frame / 255.0

    # Gerar histograma de luminosidade
    hist, bin_edges = np.histogram(frame_norm, bins=256, range=(0, 1))

    # Remover os zeros iniciais e finais
    primeiro_indice_nao_zero = np.min(np.nonzero(hist))
    ultimo_indice_nao_zero = np.max(np.nonzero(hist))

    # Recortar o histograma e os bin_edges entre o primeiro e o último índice não zero
    hist = hist[primeiro_indice_nao_zero:ultimo_indice_nao_zero + 1]
    bin_edges = bin_edges[primeiro_indice_nao_zero:ultimo_indice_nao_zero + 2]

    # Suavizar o histograma
    hist_smooth = gaussian_filter(hist, sigma=3.0)

    # Encontrar o índice do mínimo global no histograma suavizado
    limiar_index = np.argmin(hist_smooth)

    # Obter o limiar a partir do bin_edges
    limiar = bin_edges[limiar_index]

    return limiar, hist_smooth, bin_edges

def binarizar_frame(frame, limiar):
    """
    Aplica a binarização ao frame com base no limiar fornecido.
    """
    # Normalizar o frame para o intervalo [0, 1]
    frame_norm = frame / 255.0
    
    # Aplicar binarização (pixels abaixo do limiar são 0, acima são 1)
    binarizado = np.where(frame_norm < limiar, 0, 1)
    
    return binarizado

def extrair_contornos(binarizado, passo_pixel):
    """
    Extrai o contorno da gota com base na imagem binarizada.
    - Varre a imagem a cada passo_pixel para encontrar os primeiros pixels pretos de cima para baixo.
    - Para cada coluna, a varredura para ao encontrar o primeiro ponto preto.
    """
    contorno = []
    altura, largura = binarizado.shape

    # Varredura vertical
    for x in range(0, largura, passo_pixel):
        for y in range(altura):
            if binarizado[y, x] == 0:  # Encontrar o primeiro ponto preto
                contorno.append((x, y))
                break  # Parar a varredura nesta coluna ao encontrar o primeiro ponto preto

    return contorno

def separar_gota_e_horizonte(contorno, passo_pixel, largura):
    """
    Separa os pontos da gota e do horizonte com base nos baselines esquerdo e direito e no zenite.
    - Define a margem superior do horizonte como 1 * passo_pixel acima da baseline.
    - A gota só pode conter pontos à direita do horizonte esquerdo e à esquerda do horizonte direito.
    - Desconsidera pontos nas extremidades ou outliers dentro do horizonte.
    - Remove os últimos 2 pontos do horizonte esquerdo e os primeiros 2 do horizonte direito para mitigar distorção.
    """
    # Encontrar o zenite (ponto mais alto da gota)
    zenite = min(contorno, key=lambda p: p[1])

    # Separar os pontos à esquerda e à direita do zenite
    pontos_esquerda = [p for p in contorno if p[0] < zenite[0]]
    pontos_direita = [p for p in contorno if p[0] > zenite[0]]

    # Encontrar os baselines esquerdo e direito (pontos mais baixos)
    baseline_esq = max(pontos_esquerda, key=lambda p: p[1]) if pontos_esquerda else zenite
    baseline_dir = max(pontos_direita, key=lambda p: p[1]) if pontos_direita else zenite

    # Definir as faixas de Y para o horizonte
    limite_horizonte_esq = baseline_esq[1] - 1 * passo_pixel
    limite_horizonte_dir = baseline_dir[1] - 1 * passo_pixel

    # Separar os pontos da gota e do horizonte
    pontos_gota = []
    pontos_horizonte_esq = []
    pontos_horizonte_dir = []

    for x, y in contorno:
        # Verificar se o ponto faz parte do horizonte esquerdo
        if (x < zenite[0] and limite_horizonte_esq <= y <= baseline_esq[1]):
            pontos_horizonte_esq.append((x, y))
        # Verificar se o ponto faz parte do horizonte direito
        elif (x > zenite[0] and limite_horizonte_dir <= y <= baseline_dir[1]):
            pontos_horizonte_dir.append((x, y))
        # Verificar se o ponto faz parte da gota, evitando extremidades e outliers
        elif x > max([p[0] for p in pontos_horizonte_esq], default=0) and \
             x < min([p[0] for p in pontos_horizonte_dir], default=largura):
            pontos_gota.append((x, y))

    # Desconsiderar os últimos 2 pontos do horizonte esquerdo e os primeiros 2 do direito
    if len(pontos_horizonte_esq) > 2:
        pontos_horizonte_esq = pontos_horizonte_esq[:-2]
    if len(pontos_horizonte_dir) > 2:
        pontos_horizonte_dir = pontos_horizonte_dir[2:]

    return pontos_gota, pontos_horizonte_esq, pontos_horizonte_dir

def varredura_horizontal_para_gota(binarizado, pontos_gota, pontos_horizonte_esq, pontos_horizonte_dir, passo_pixel):
    """
    Realiza a varredura horizontal da esquerda para a direita na faixa vertical da gota para encontrar pontos complementares.
    - A varredura é feita uma vez após a separação entre gota e horizonte.
    - Apenas pontos à direita de (maior valor do horizonte esquerdo - passo_pixel - 1) e à esquerda do primeiro ponto do horizonte direito são considerados.
    """
    altura, largura = binarizado.shape
    novos_pontos_gota = []

    # Encontrar o último ponto do horizonte esquerdo e o primeiro ponto do horizonte direito
    max_x_horizonte_esq = max([p[0] for p in pontos_horizonte_esq], default=0)  # Último ponto do horizonte esquerdo
    min_x_horizonte_dir = min([p[0] for p in pontos_horizonte_dir], default=largura)  # Primeiro ponto do horizonte direito

    # Ajuste da limitação à direita do horizonte esquerdo
    limite_x_horizonte_esq = max_x_horizonte_esq - passo_pixel - 1

    # Encontrar os limites verticais da gota (limite superior = topo da gota)
    y_min = min([p[1] for p in pontos_gota], default=0)

    # Limite inferior: maior valor de Y do horizonte esquerdo menos 2 * passo_pixel
    y_max = max([p[1] for p in pontos_horizonte_esq], default=altura) - 2 * passo_pixel

    # Realizar a varredura horizontal da esquerda para a direita na faixa vertical
    for y in range(y_min, y_max + 1, passo_pixel):
        for x in range(0, largura):  # Começando do ponto mais à esquerda da imagem (x=0)
            if binarizado[y, x] == 0:  # Encontrar o primeiro ponto preto
                # Garantir que o ponto esteja à direita de (max_x_horizonte_esq - passo_pixel - 1) e à esquerda do horizonte direito
                if limite_x_horizonte_esq < x < min_x_horizonte_dir:
                    novos_pontos_gota.append((x, y))  # Adicionar ponto atual (n)
                break  # Varredura feita apenas uma vez por linha

    return novos_pontos_gota

def ajustar_reta_horizonte(pontos_horizonte_esq, pontos_horizonte_dir):
    """
    Ajusta uma reta aos pontos do horizonte (esquerdo e direito) usando um ajuste linear.
    """
    # Combinar os pontos do horizonte esquerdo e direito
    pontos_horizonte = pontos_horizonte_esq + pontos_horizonte_dir
    x_horizonte = np.array([p[0] for p in pontos_horizonte])
    y_horizonte = np.array([p[1] for p in pontos_horizonte])

    # Ajuste linear (reta)
    coeficientes = np.polyfit(x_horizonte, y_horizonte, 1)
    a_reta, b_reta = coeficientes  # a: inclinação, b: intercepto

    return a_reta, b_reta

def elipse(coord, h, k, a, b):
    """
    Define a equação de uma elipse centrada em (h, k) com semieixos a (horizontal) e b (vertical).
    """
    x, y = coord
    return ((x - h) / a) ** 2 + ((y - k) / b) ** 2 - 1

def ajustar_elipse(pontos_gota_vertical, pontos_gota_horizontal):
    """
    Ajusta uma elipse utilizando todos os pontos da gota (varredura vertical e horizontal).
    Em caso de erro no ajuste (como quando o número de iterações excede maxfev), exibe uma mensagem de erro.
    """
    # Combinar os pontos da gota (vertical e horizontal)
    pontos_gota = pontos_gota_vertical + pontos_gota_horizontal
    x_gota = np.array([p[0] for p in pontos_gota])
    y_gota = np.array([p[1] for p in pontos_gota])

    # Parâmetros iniciais: centro (h, k) e semieixos a, b
    p0 = (np.mean(x_gota), np.mean(y_gota), np.std(x_gota), np.std(y_gota))

    try:
        # Ajustar a elipse
        params, _ = scipy.optimize.curve_fit(elipse, (x_gota, y_gota), np.zeros_like(x_gota), p0=p0, maxfev=2500)
    except RuntimeError as e:
        # Se houver um erro, exibe a mensagem de erro e retorna None
        print(f"Erro ao ajustar a elipse: {e}")
        return None
    
    return params  # Retorna h_elipse, k_elipse, a_elipse, b_elipse

def calcular_intersecoes_retas_tangentes(a_reta, b_reta, h_elipse, k_elipse, a_elipse, b_elipse):
    """
    Calcula os pontos de interseção entre a elipse e a reta do horizonte,
    e determina as equações das retas tangentes nesses pontos de interseção.
    """
    # Coeficientes da equação quadrática para encontrar interseção entre a elipse e a reta
    A = (1 / a_elipse**2) + (a_reta**2 / b_elipse**2)
    B = (-(2 * h_elipse / a_elipse**2) + (2 * a_reta * (b_reta - k_elipse) / b_elipse**2))
    C = ((h_elipse**2 / a_elipse**2) + ((b_reta - k_elipse)**2 / b_elipse**2) - 1)

    # Solução da equação quadrática (Ax^2 + Bx + C = 0) para encontrar x
    delta = B**2 - 4 * A * C

    if delta < 0:
        raise ValueError("Não há interseção entre a elipse e a reta")

    # Pontos de interseção x (duas soluções)
    x_intersecao1 = (-B + np.sqrt(delta)) / (2 * A)
    x_intersecao2 = (-B - np.sqrt(delta)) / (2 * A)

    # Calcula os valores correspondentes de y usando a equação da reta
    y_intersecao1 = a_reta * x_intersecao1 + b_reta
    y_intersecao2 = a_reta * x_intersecao2 + b_reta

    # Calcula as derivadas parciais da equação da elipse para obter a inclinação das tangentes
    def derivada_dx(x, h, a):
        return 2 * (x - h) / a**2

    def derivada_dy(y, k, b):
        return 2 * (y - k) / b**2

    # Inclinação das retas tangentes (inverso negativo da derivada implícita)
    m_tangente1 = -1 / (derivada_dy(y_intersecao1, k_elipse, b_elipse) / derivada_dx(x_intersecao1, h_elipse, a_elipse))
    m_tangente2 = -1 / (derivada_dy(y_intersecao2, k_elipse, b_elipse) / derivada_dx(x_intersecao2, h_elipse, a_elipse))

    # Equações das retas tangentes
    def equacao_tangente(x, x_intersecao, y_intersecao, m_tangente):
        return m_tangente * (x - x_intersecao) + y_intersecao

    return (x_intersecao1, y_intersecao1, m_tangente1), (x_intersecao2, y_intersecao2, m_tangente2)

def exibir_imagens_com_intersecoes_e_tangentes(
        frame_binarizado, pontos_gota_vertical, pontos_gota_horizontal, 
        pontos_horizonte_esq, pontos_horizonte_dir, a_reta, b_reta, 
        params_elipse, intersecao1, intersecao2):
    """
    Exibe a imagem binarizada com a reta do horizonte ajustada, a elipse ajustada,
    os pontos de interseção, e as retas tangentes sobrepostas.
    As tangentes são exibidas como segmentos de reta, e os pontos de interseção como 'x'.
    """
    
    plt.imshow(frame_binarizado, cmap='gray')

    # Ajuste da reta do horizonte
    x_reta = np.linspace(min([p[0] for p in pontos_horizonte_esq + pontos_horizonte_dir]), 
                         max([p[0] for p in pontos_horizonte_esq + pontos_horizonte_dir]), 50)
    y_reta = a_reta * x_reta + b_reta
    plt.plot(x_reta, y_reta, color='green', label='Reta do Horizonte', linewidth=2)

    # Ajuste da elipse
    h_elipse, k_elipse, a_elipse, b_elipse = params_elipse
    x_elipse = h_elipse + a_elipse * np.cos(np.linspace(0, 2 * np.pi, 100))
    y_elipse = k_elipse + b_elipse * np.sin(np.linspace(0, 2 * np.pi, 100))
    plt.plot(x_elipse, y_elipse, color='blue', label='Elipse da Gota', linewidth=2)

    # Exibir pontos de interseção como 'x'
    x_intersecao1, y_intersecao1, m_tangente1 = intersecao1
    x_intersecao2, y_intersecao2, m_tangente2 = intersecao2
    plt.scatter([x_intersecao1, x_intersecao2], [y_intersecao1, y_intersecao2], 
                color='orange', marker='x', s=50, zorder=5, label='Pontos de Interseção')

    # Limitar as tangentes para exibir como segmentos de reta
    # Tangente 1 (segmento de reta limitado)
    x_tangente1 = np.linspace(x_intersecao1 - 50, x_intersecao1 + 50, 100)  # Limitar a extensão horizontal
    y_tangente1 = m_tangente1 * (x_tangente1 - x_intersecao1) + y_intersecao1
    plt.plot(x_tangente1, y_tangente1, 'r--', label='Tangente', linewidth=2)

    # Tangente 2 (segmento de reta limitado)
    x_tangente2 = np.linspace(x_intersecao2 - 50, x_intersecao2 + 50, 100)  # Limitar a extensão horizontal
    y_tangente2 = m_tangente2 * (x_tangente2 - x_intersecao2) + y_intersecao2
    plt.plot(x_tangente2, y_tangente2, 'r--', linewidth=2)

    # Finalizar exibição
    plt.axis('off')
    plt.legend()
    plt.show()

def calcular_angulo_contato(m_tangente, a_reta, k_elipse, y_intersecao):
    """
    Calcula o ângulo de contato com base na inclinação da reta tangente e da linha do horizonte.
    Se o centro da elipse estiver acima da linha do horizonte, retorna o ângulo suplementar.
    """
    # Calcular o ângulo entre a tangente e o horizonte em graus
    angulo = np.degrees(np.arctan(np.abs((m_tangente - a_reta) / (1 + m_tangente * a_reta))))
    
    # Verificar se o centro da elipse está acima ou abaixo da linha do horizonte
    if k_elipse < y_intersecao:
        # Se o centro da elipse está acima da linha do horizonte, retorna o ângulo suplementar
        return 180.0 - angulo
    else:
        # Caso contrário, retorna o ângulo normal
        return angulo

def plotar_grafico_evolucao_angulos(angulos):
    """
    Plota um gráfico mostrando a evolução dos ângulos de contato ao longo do tempo para vídeos.
    Também exibe o ângulo inicial, final e médio ao lado do gráfico como uma caixa de texto separada.
    """
    tempos = [item[0] for item in angulos]
    angulos_medios = [item[3] for item in angulos]

    # Calcular ângulo inicial, final e médio geral
    angulo_inicial = angulos_medios[0]
    angulo_final = angulos_medios[-1]
    angulo_medio_geral = sum(angulos_medios) / len(angulos_medios)

    # Criar o gráfico da evolução dos ângulos
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tempos, angulos_medios, marker='o', linestyle='-', color='b', label='Ângulo médio')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Ângulo de Contato (°)')
    ax.set_title('Evolução do Ângulo de Contato ao Longo do Tempo')
    ax.grid(True)
    ax.legend()

    # Adicionar a caixa de texto à direita do gráfico
    fig.text(0.95, 0.5, 
             f"Ângulo Inicial: {angulo_inicial:.2f}°\n"
             f"Ângulo Final: {angulo_final:.2f}°\n"
             f"Ângulo Médio: {angulo_medio_geral:.2f}°",
             fontsize=12, bbox=dict(facecolor='white', edgecolor='black'),
             verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)

    # Ajustar o layout para deixar espaço para a caixa de texto
    plt.subplots_adjust(right=0.75)

    # Exibir o gráfico
    plt.show()

    # Retornar as informações dos ângulos
    return angulo_inicial, angulo_final, angulo_medio_geral

def exibir_tabela_angulos(angulos, angulo_inicial, angulo_final, angulo_medio_geral):
    """
    Exibe uma tabela com os tempos e os ângulos calculados para cada frame no caso de vídeo.
    """
    # Converter a lista de ângulos em um DataFrame
    df_angulos = pd.DataFrame(angulos, columns=['Tempo (s)', 'Ângulo Esquerdo (°)', 'Ângulo Direito (°)', 'Ângulo Médio (°)'])
    
    # Exibir a tabela (copiável e exportável)
    print(f"Ângulo inicial: {angulo_inicial:.2f}°, Ângulo médio geral: {angulo_medio_geral:.2f}°, Ângulo final: {angulo_final:.2f}°")
    print("\nTabela de ângulos de contato:")
    print(df_angulos)
    
    # Permitir a exportação se necessário
    return df_angulos

def main():
    """
    Função principal que controla o fluxo do programa.
    Abre um arquivo de vídeo ou imagem, processa o primeiro frame e lê os frames subsequentes.
    Calcula e exibe os ângulos de contato, dependendo do tipo de arquivo (imagem ou vídeo).
    """
    try:
        imagem, video, tipo_arquivo = abrir_arquivo()
    except ValueError as e:
        print(e)
        return

    limiar = None  # Para armazenar o limiar, calculado apenas no primeiro frame
    angulos = []  # Lista para armazenar os ângulos e tempos

    try:
        # Processamento do primeiro frame
        if tipo_arquivo == "imagem":
            frame = ler_frame(imagem, None, "imagem")
            frame_processado = processar_primeiro_frame(frame)
            
            # Calcular o limiar e gerar o histograma suavizado
            limiar, hist_smooth, bin_edges = calcular_limiar(frame_processado)
            
            # Binarizar o frame
            frame_binarizado = binarizar_frame(frame_processado, limiar)
            
            # Obter as dimensões da imagem (altura, largura)
            altura, largura = frame_binarizado.shape

            # Extrair contornos da gota e do horizonte
            contorno = extrair_contornos(frame_binarizado, passo_pixel)
            
            # Separar pontos da gota e do horizonte com o método ajustado, passando a largura
            pontos_gota_vertical, pontos_horizonte_esq, pontos_horizonte_dir = separar_gota_e_horizonte(contorno, passo_pixel, largura)
            
            # Executar a varredura horizontal para complementar a gota
            pontos_gota_horizontal = varredura_horizontal_para_gota(frame_binarizado, pontos_gota_vertical, pontos_horizonte_esq, pontos_horizonte_dir, passo_pixel)
            
            # Ajustar a reta do horizonte
            a_reta, b_reta = ajustar_reta_horizonte(pontos_horizonte_esq, pontos_horizonte_dir)
            
            # Ajustar a elipse da gota usando todos os pontos (vertical + horizontal)
            params_elipse = ajustar_elipse(pontos_gota_vertical, pontos_gota_horizontal)

            if params_elipse is None:
                print("Erro no ajuste da elipse. Encerrando o processamento.")
                return  # Encerra o processamento caso o ajuste da elipse falhe

            # Calcular interseções e retas tangentes
            intersecao1, intersecao2 = calcular_intersecoes_retas_tangentes(a_reta, b_reta, params_elipse[0], params_elipse[1], params_elipse[2], params_elipse[3])

            # Cálculo dos ângulos de contato (em graus), considerando o centro da elipse
            angulo_esquerdo = calcular_angulo_contato(intersecao1[2], a_reta, params_elipse[1], intersecao1[1])
            angulo_direito = calcular_angulo_contato(intersecao2[2], a_reta, params_elipse[1], intersecao2[1])
            angulo_medio = (angulo_esquerdo + angulo_direito) / 2

            # Log dos ângulos
            print(f"Ângulo esquerdo: {angulo_esquerdo:.2f}°, Ângulo direito: {angulo_direito:.2f}°, Ângulo médio: {angulo_medio:.2f}°")

            # Exibir a imagem binarizada com os ajustes da reta e da elipse, e as interseções/tangentes
            exibir_imagens_com_intersecoes_e_tangentes(frame_binarizado, pontos_gota_vertical, pontos_gota_horizontal, pontos_horizonte_esq, pontos_horizonte_dir, 
                                                       a_reta, b_reta, params_elipse, intersecao1, intersecao2)

        elif tipo_arquivo == "video":
            fps = video.get(cv2.CAP_PROP_FPS)  # Obter a taxa de quadros por segundo do vídeo
            analises_por_frame = int(fps / analises_por_segundo)  # Frames a serem analisados por segundo
            frame_count = 0

            while True: #Loop para iteração do vídeo
                frame = ler_frame(None, video, "video")
                if frame is None:
                    break

                # Processar apenas alguns frames de acordo com 'analises_por_segundo'
                if frame_count % analises_por_frame == 0:
                    frame_processado = processar_primeiro_frame(frame)
                    
                    # Calcular o limiar apenas no primeiro frame
                    if limiar is None:
                        limiar, hist_smooth, bin_edges = calcular_limiar(frame_processado)
                    
                    # Binarizar o frame
                    frame_binarizado = binarizar_frame(frame_processado, limiar)

                    # Obter as dimensões do frame
                    altura, largura = frame_binarizado.shape

                    # Extrair contornos da gota e do horizonte
                    contorno = extrair_contornos(frame_binarizado, passo_pixel)
                    
                    # Separar pontos da gota e do horizonte com o método ajustado, passando a largura
                    pontos_gota_vertical, pontos_horizonte_esq, pontos_horizonte_dir = separar_gota_e_horizonte(contorno, passo_pixel, largura)
                    
                    # Executar a varredura horizontal para complementar a gota
                    pontos_gota_horizontal = varredura_horizontal_para_gota(frame_binarizado, pontos_gota_vertical, pontos_horizonte_esq, pontos_horizonte_dir, passo_pixel)
                    
                    # Ajustar a reta do horizonte
                    a_reta, b_reta = ajustar_reta_horizonte(pontos_horizonte_esq, pontos_horizonte_dir)
                    
                    # Ajustar a elipse da gota usando todos os pontos (vertical + horizontal)
                    params_elipse = ajustar_elipse(pontos_gota_vertical, pontos_gota_horizontal)

                    if params_elipse is None:
                        print("Erro no ajuste da elipse. Encerrando o processamento.")
                        break  # Encerra o processamento caso o ajuste da elipse falhe, mas continua com a exibição dos resultados

                    # Calcular interseções e retas tangentes
                    intersecao1, intersecao2 = calcular_intersecoes_retas_tangentes(a_reta, b_reta, params_elipse[0], params_elipse[1], params_elipse[2], params_elipse[3])

                    # Cálculo dos ângulos de contato (em graus)
                    tempo_atual = frame_count / fps  # Tempo em segundos
                    angulo_esquerdo = calcular_angulo_contato(intersecao1[2], a_reta, params_elipse[1], intersecao1[1])
                    angulo_direito = calcular_angulo_contato(intersecao2[2], a_reta, params_elipse[1], intersecao2[1])
                    angulo_medio = (angulo_esquerdo + angulo_direito) / 2

                    # Armazenar os ângulos e o tempo na lista
                    angulos.append((tempo_atual, angulo_esquerdo, angulo_direito, angulo_medio))

                frame_count += 1
                plt.pause(0.001)

    except Exception as e:
        print(f"Erro no processamento: {e}")

    finally:
        # Se houver ângulos coletados, exibe o gráfico e a tabela
        if angulos:  # Finalizar processamento: Gráfico e tabela para vídeos
            angulo_inicial, angulo_final, angulo_medio_geral = plotar_grafico_evolucao_angulos(angulos)
            df_angulos = exibir_tabela_angulos(angulos, angulo_inicial, angulo_final, angulo_medio_geral)

            # Gerar timestamp para o nome do arquivo
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            nome_arquivo = f'resultados_angulos_{timestamp}.csv'

            # Exibir a tabela e permitir exportação
            df_angulos.to_csv(nome_arquivo, index=False)
            print(f"Tabela de ângulos exportada como '{nome_arquivo}'.")
        

        if tipo_arquivo == "video":
            video.release()

if __name__ == "__main__":
    main()
