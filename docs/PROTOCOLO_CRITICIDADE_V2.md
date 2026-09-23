# Protocolo de previsão da criticidade — versão 2

## Objetivo

Prever diretamente uma das quatro categorias epidemiológicas por bairro: `Baixo`, `Medio`, `Alto` ou `Critico`. Não há binarização de surto.

## Definição do alvo

- Unidade: bairro por semana epidemiológica.
- Horizonte configurável e avaliado: uma, duas, três e quatro semanas.
- Janela da incidência: soma dos casos nas quatro semanas encerradas na semana-alvo futura.
- Taxa: casos acumulados na janela / população do Censo 2022 × 100.000.
- Limites: Baixo < 100; Medio de 100 a < 300; Alto de 300 a < 500; Critico >= 500.

Para horizonte 1, a previsão feita na semana `t` estima a categoria da janela de quatro semanas encerrada em `t+1`. Portanto, três semanas da janela já são observadas e apenas a semana final é futura. Para horizontes maiores, cresce a parcela ainda não observada.

## Recorte e prevenção de vazamento

- Treinamento final: semanas-alvo de 2015 a 2020.
- Teste final: todas as semanas-alvo de 2021, nunca usadas para ajuste ou seleção.
- Validação progressiva: treino em 2015 e validação em 2016; depois 2015–2016/2017, 2015–2017/2018, 2015–2018/2019 e 2015–2019/2020.
- As previsões dos cinco cortes são consolidadas fora da amostra para selecionar o XGBoost; 2021 não participa da seleção.
- Imputação climática: mediana da mesma semana epidemiológica calculada apenas em 2015–2020.
- Todos os atributos de uma linha estão disponíveis até a semana de origem da previsão.

## Atributos acrescentados

- Oito semanas de casos, precipitação, temperatura e total de casos do Recife.
- Somas móveis de casos em 2, 4 e 8 semanas; tendência e razão de crescimento.
- Incidência recente em quatro semanas.
- Chuva acumulada e temperatura média em 4 e 8 semanas.
- Tendência municipal, população, sazonalidade da semana-alvo e bairro.

## Tratamento do desbalanceamento

- XGBoost: pesos inversos suavizados por classe; a potência do peso e os demais hiperparâmetros são escolhidos pela validação progressiva.
- RNA e LSTM: Focal Loss com pesos inversos suavizados, reduzindo a dominância da categoria Baixo.
- Seleção das redes: maior F2 macro na validação, dando mais importância à sensibilidade.

## Otimização do XGBoost

Para cada horizonte são avaliadas seis configurações, incluindo obrigatoriamente a configuração anterior. A busca cobre taxa de aprendizado, profundidade, peso mínimo de filho, gamma, amostragem de linhas e colunas, regularização L1/L2 e potência do peso das classes. Cada configuração é treinada nos cinco cortes progressivos. A escolha usa, nesta ordem, F2 macro consolidado, acurácia balanceada e log loss. O número final de árvores é a mediana dos melhores números encontrados nos cinco cortes.

## Métricas principais

- Acurácia balanceada e F1 macro: comparação global sem deixar a classe Baixo dominar.
- F2 macro: enfatiza recall, útil quando deixar de detectar uma categoria grave custa mais.
- Recall, precisão, F1 e F2 por categoria: leitura obrigatória para Alto e Critico.
- PR-AUC macro e log loss: qualidade das probabilidades produzidas.
- Erro absoluto médio de nível, acerto até um nível e kappa quadrático: respeitam a ordem das categorias.
- Taxa de subestimação grave: proporção em que a previsão fica dois ou mais níveis abaixo do real.
- Acurácia simples e F1 ponderado: métricas complementares, não critérios únicos.

Duas referências simples são incluídas: prever sempre Baixo e repetir a categoria observada na origem. Elas não são modelos concorrentes; servem para verificar se o aprendizado supera regras triviais.

## Relação com o experimento numérico

Os modelos anteriores de regressão mediam MAE, RMSE e R² para casos semanais e
permanecem recuperáveis pelo histórico do Git. Eles não fazem parte do código
operacional atual. A versão 2 é uma tarefa diferente, dedicada à categoria da
incidência acumulada em quatro semanas; por isso, as métricas numéricas antigas
não são diretamente atribuídas aos classificadores novos.
