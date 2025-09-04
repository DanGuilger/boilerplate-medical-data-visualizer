import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
# aqui eu calculo se a pessoa ta acima do peso ou nao
# pego o peso e divido pela altura ao quadrado (que tem que estar em metros)
# se der mais que 25, a pessoa ta com sobrepeso (vira 1), senao fica 0
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
# normalizo os dados de colesterol e glicose pra ficar mais facil de trabalhar
# se o valor for 1 (normal), vira 0. se for maior que 1 (ruim), vira 1
# assim fica tudo padronizado: 0 = bom, 1 = ruim
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    # uso o melt pra transformar as colunas em linhas
    # isso deixa os dados no formato que o seaborn precisa pro grafico
    # pego so as variaveis que interessam: colesterol, glicose, fumo, alcool, atividade e sobrepeso
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    # agora preciso contar quantas vezes cada combinacao aparece
    # agrupo por doenca cardiaca, variavel e valor, ai conto tudo
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    # crio o grafico de barras separado em dois paineis
    # um painel pra quem nao tem doenca cardiaca (cardio=0) e outro pra quem tem (cardio=1)
    catplot = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')


    # 8
    # pego a figura do grafico pra poder salvar depois
    fig = catplot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # limpo os dados tirando as linhas meio estranhas
    # pressao baixa nao pode ser maior que a alta
    # tiro tambem os valores muito extremos de altura e peso (so os 2,5% mais baixos e altos)
    # isso deixa os dados mais confiaveis
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    # calculo a correlacao entre todas as variaveis
    # isso mostra como uma coisa se relaciona com outra
    corr = df_heat.corr()

    # 13
    # crio uma mascara pra esconder a parte de cima do triangulo
    # como a correlacao e igual dos dois lados, nao precisa mostrar tudo
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    # configuro o tamanho da figura pro mapa de calor ficar bonito
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    # desenho o mapa de calor com os numeros aparecendo
    # as cores mostram se a correlacao e forte ou fraca
    # center=0 faz com que o zero fique bem no meio das cores
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, ax=ax, cbar_kws={'shrink': 0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
