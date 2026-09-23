# Alterações da regressão semanal para a classificação de criticidade v2

| Aspecto | Experimento anterior | Classificação de criticidade v2 |
|---|---|---|
| Saída | Quantidade contínua de casos da semana futura | Probabilidade e categoria entre Baixo, Medio, Alto e Critico |
| Janela da incidência | Uma semana | Quatro semanas encerradas na semana-alvo |
| Horizonte | 1 a 4 semanas | 1 a 4 semanas |
| Histórico de entrada | 4 semanas | 8 semanas |
| Casos | Lags 1 a 4 | Oito observações, somas 2/4/8, média, tendência e crescimento |
| Clima | Quatro lags | Oito observações, chuva acumulada e temperatura média 4/8 |
| Contexto municipal | Não usado | Casos totais do Recife e tendência municipal |
| Desbalanceamento | Sem tratamento específico | Peso de classes; Focal Loss nas redes |
| Seleção interna | Menor erro em 2020 | Maior F2 macro em 2016; XGBoost também seleciona potência dos pesos |
| Métricas centrais | MAE, RMSE e R² | Acurácia balanceada, F1/F2 macro, PR-AUC e métricas por classe |
| Natureza ordinal | Não modelada | Erro entre níveis, kappa quadrático e subestimação grave |
| Referências | Não incluídas | Sempre Baixo e persistência da categoria atual |

O código operacional atual mantém somente a classificação de criticidade v2 em
`results/criticidade_v2` e `models/criticidade_v2`. A regressão semanal anterior
permanece recuperável pelo histórico do Git, mas não integra mais esta versão da
aplicação.

## Alterações específicas do XGBoost

- O experimento anterior usava `XGBRegressor`; ele foi retirado do código operacional atual.
- Foi acrescentado um `XGBClassifier` multiclasse com objetivo `multi:softprob`.
- A potência do peso inverso das classes é escolhida entre 0,25, 0,50, 0,75 e 1,00 usando somente a validação interna.
- O treinamento final usa todos os dados de 2015–2020 e o número de árvores selecionado antes do teste.
- As probabilidades das quatro classes, a importância dos atributos e a matriz de confusão são salvas.

## Alterações específicas da RNA

- A saída passou de um neurônio de regressão para quatro logits.
- A perda quadrática foi substituída por Focal Loss ponderada.
- O critério de parada passou a ser F2 macro na validação.

## Alterações específicas da LSTM

- A sequência passou de quatro para oito semanas.
- Cada passo recebe casos do bairro, chuva, temperatura e casos totais do Recife.
- A cabeça da rede passou a produzir quatro logits e usa Focal Loss ponderada.
