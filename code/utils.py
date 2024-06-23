import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

def get_graph_by_date(date):
    """
    Recupera o grafo correspondente a uma determinada data de um arquivo GraphML.
    
    Parâmetros:
    - date (str): String de data no formato usado nos nomes de arquivo de grafo.
    
    Retorna:
    - nx.Graph: Objeto de grafo correspondente à data fornecida.
    
    Lança:
    - FileNotFoundError: Se o arquivo GraphML para a data fornecida não existir.
    """
    converted_date = date.replace('/', '_')
    graph_filename = f"grafos/graph_{converted_date}.graphml"
    
    if not os.path.exists(graph_filename):
        raise FileNotFoundError(f"Arquivo GraphML não encontrado para a data: {date}")
    
    G = nx.read_graphml(graph_filename)
    return G

def save_graphs_to_files(df_pox, df_edges):
    """
    Constrói grafos a partir de dados de infectados por cidade e salva cada grafo em formato GraphML.

    Args:
    - df_pox (DataFrame): DataFrame contendo dados de infectados por cidade e data.
    - df_edges (DataFrame): DataFrame contendo informações de arestas entre cidades.

    Salva:
    - Arquivos GraphML individuais para cada data, nomeados 'graph_data.graphml'.
    - Um arquivo pickle 'graph_list.pickle' contendo uma lista de todos os grafos construídos.
    """
    all_graphs = []
    
    for date in df_pox['Date'].unique():
        df_current_date = df_pox[df_pox['Date'] == date]
        G = nx.Graph()
        
        # Adicionar nós (cidades) com suas features (valor de infectados)
        for index, row in df_current_date.iterrows():
            city = index  
            for col in df_pox.columns[1:]: # Não usar a coluna date aqui
                signal_value = row[col]
                G.add_node(col, signal=signal_value)
        
        # Conexão entre cidades (arestas)
        for index, row in df_edges.iterrows():
            name_1 = row['name_1']
            name_2 = row['name_2']
            G.add_edge(name_1, name_2)
        
        # Salvar o grafo em formato GraphML
        converted_date = date.replace('/', '_')
        nx.write_graphml(G, f"grafos/graph_{converted_date}.graphml")
        all_graphs.append(G)
    
    # Salvar a lista de grafos em um arquivo pickle
    with open("grafos/graph_list.pickle", "wb") as f:
        pickle.dump(all_graphs, f)

def get_existing_graph_names(directory="grafos"):
    """
    Recupera todos os nomes de arquivos GraphML existentes no diretório especificado.
    
    Parâmetros:
    - directory (str): Caminho do diretório onde os arquivos GraphML estão armazenados.
                      O padrão é "grafos".
    
    Retorna:
    - list: Lista de nomes de arquivos GraphML existentes, ordenada pelo dia (do mais velho para o mais novo).
    """
    graphml_files = []
    
    # Lista todos os arquivos no diretório especificado
    files = os.listdir(directory)
    
    # Filtra os arquivos com base na convenção de nome de arquivo GraphML
    for file in files:
        if file.startswith("graph_") and file.endswith(".graphml"):
            # Extrai a data do nome do arquivo
            filename_without_extension = os.path.splitext(file)[0]  # remove a extensão .graphml
            date_string = filename_without_extension.split("_")[1:]  # pega a parte da data dd_mm_yyyy
            formatted_date = "_".join(date_string)  # junta de volta para formar dd_mm_yyyy
            
            # Converte a data para um objeto datetime para ordenação
            date_obj = datetime.strptime(formatted_date, "%d_%m_%Y")
            
            # Adiciona o nome do arquivo e a data como tupla
            graphml_files.append((file, date_obj))
    
    # Ordena os arquivos com base na data (do mais velho para o mais novo)
    graphml_files.sort(key=lambda x: x[1])
    
    # Retorna apenas os nomes dos arquivos, sem a data
    return [file[0] for file in graphml_files]

def plot_graph_by_date(date):
    """
    Plota o grafo correspondente a uma determinada data com configurações personalizadas.
    
    Parâmetros:
    - date (str): String de data no formato usado nos nomes de arquivo de grafo.
    
    Exibe:
    - Plot do grafo correspondente à data fornecida com cores baseadas nos valores de sinal dos nós.
    
    Lança:
    - FileNotFoundError: Se o arquivo GraphML para a data fornecida não existir.
    """
    try:
        G = get_graph_by_date(date)
    except FileNotFoundError as e:
        print(e)
        return
    
    plt.figure(figsize=(10, 8))  # Tamanho da figura
    
    pos = nx.spring_layout(G)  # Layout do grafo
    
    signal_values = [G.nodes[city]['signal'] for city in G.nodes()]  # Coletando os valores de sinal dos nós para coloração
    norm = plt.Normalize(min(signal_values), max(signal_values)) # Normalização e mapa de cores
    cmap = plt.cm.plasma
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=signal_values, cmap=cmap, alpha=0.8)  # Desenhar nós
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.0, alpha=0.5) # Desenhar arestas
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif') # Desenhar rótulos dos nós (nomes das cidades)
    
    # Configuração da barra de cores
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Signal')
    # cbar.ax.invert_yaxis()  # Opcional: inverter a barra de cores se necessário
    
    plt.title(f'Grafo para data: {date}')  # Título
    plt.axis('off')
    plt.show()  # Exibir o gráfico