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
Este código está disponível para uso público através da licensa MIT. Você pode utilizá-lo e modificá-lo, desde que mantenha os créditos do autor.
