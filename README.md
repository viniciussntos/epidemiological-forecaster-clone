# Epidemiological Forecaster

Aplicação de previsão da criticidade epidemiológica da dengue por bairro do Recife. O sistema usa dados semanais de casos, clima e população para classificar os horizontes `S+1` a `S+4` como `Baixo`, `Médio`, `Alto` ou `Crítico`.

O XGBoost é o modelo operacional. RNA e LSTM permanecem no projeto como experimentos comparativos do TCC.

## Arquitetura

```text
Supabase ──► GitHub Actions ──► XGBoost S+1…S+4 ──► Supabase
    ▲                                                  │
    └──────────────────── FastAPI ◄────────────────────┘
                              │
                              ▼
                     Streamlit Community Cloud
```

- **Supabase:** dados semanais e previsões.
- **GitHub Actions:** verificação diária, inferência e avaliação retroativa.
- **FastAPI:** camada de consulta e regras de apresentação.
- **Streamlit:** dashboard público.
- **Render Free:** hospedagem prevista para a FastAPI.

## Separação entre avaliação e produção

As métricas acadêmicas continuam sendo calculadas sem vazamento temporal:

- treinamento de avaliação: 2015–2020;
- teste independente: 2021.

Depois da avaliação, a versão de produção é treinada com 2015–2021. Essa versão gera previsões posteriores à última semana completa, mas não é avaliada nos dados usados em seu próprio treinamento.

Previsões operacionais passam pelos estados:

- `Pendente`: semana-alvo ainda indisponível;
- `Provisório`: resultado disponível, sujeito a notificações atrasadas;
- `Consolidado`: quatro semanas transcorridas desde a semana-alvo.

O prazo é configurado por `EVALUATION_LAG_WEEKS`, com padrão igual a `4`.

## Pré-requisitos locais

- Python 3.12;
- Git;
- conexão PostgreSQL do Supabase para tarefas de implantação;
- acesso ao repositório privado.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copie `.env.example` para `.env` somente no ambiente local. Nunca envie `.env` ou `DATABASE_URL` ao GitHub.

## Executar localmente

Backend:

```powershell
uvicorn app.main:app --reload
```

Frontend, em outro terminal:

```powershell
streamlit run app/app.py
```

Antes de iniciar, configure `DATABASE_URL` para a FastAPI e `API_URL` para o
Streamlit. A aplicação não usa CSVs nem endereços locais como fallback.

## Schema do Supabase

As tabelas operacionais já devem conter os dados semanais necessários. Para
aplicar ou atualizar somente o schema, sem ler arquivos locais:

```powershell
python -m scripts.apply_supabase_schema
```

O comando exige `DATABASE_URL`. No painel do Supabase, use **Connect → Session
pooler** (porta `5432`), compatível com ambientes IPv4 como Render e GitHub
Actions. A aplicação consulta exclusivamente:

- `ef_ibge_populacao_bairro`;
- `ef_inmet_semanal_recife`;
- `ef_dengue_semanal_bairro`;
- `ef_previsoes_xgboost_dashboard`;
- `ef_metricas_xgboost`.

O prefixo `ef_` isola as tabelas operacionais das bases antigas que já possam existir no mesmo projeto Supabase.

As tabelas têm RLS habilitado e não possuem políticas públicas. Navegadores nunca recebem a credencial PostgreSQL.

## Treinar o XGBoost de produção

O comando abaixo reutiliza os hiperparâmetros selecionados na validação temporal e treina os quatro horizontes com os dados consolidados até 2021:

```powershell
python train_xgboost_production.py
```

Os artefatos necessários à inferência ficam em `models/criticidade_v2/production/`.

O treinamento lê população, clima e casos semanais diretamente das tabelas
`ef_*`. Para validar a construção de um horizonte sem treinar:

```powershell
python prepare_criticality.py --horizon 1
```

O processamento de arquivos brutos continua disponível apenas como operação
offline explícita. Todos os caminhos devem ser informados pelo usuário:

```powershell
python prepare_data.py --dengue dengue.csv --climate inmet.csv --population ibge.csv --horizon 1
```

## Inferência automática

```powershell
python run_production_pipeline.py
```

A rotina:

1. confirma que clima e dengue estão simultaneamente completos;
2. exige 94 bairros, inclusive com zero casos;
3. detecta semanas ainda não previstas;
4. executa XGBoost para `S+1` a `S+4`;
5. atualiza previsões antigas como provisórias ou consolidadas;
6. publica métricas de produção após pelo menos quatro semanas-alvo consolidadas.

O workflow `.github/workflows/daily-inference.yml` executa essa rotina diariamente. Em branches que não sejam a branch padrão, use `workflow_dispatch` manualmente.

## Variáveis e segredos

| Ambiente | Variável | Finalidade |
|---|---|---|
| Render | `DATABASE_URL` | conexão privada da FastAPI com Supabase |
| Streamlit | `API_URL` | URL HTTPS pública da FastAPI |
| GitHub Actions | `DATABASE_URL` | leitura e gravação do pipeline diário |
| Pipeline | `EVALUATION_LAG_WEEKS=4` | prazo para consolidação |

## Deploy previsto

### FastAPI no Render

O arquivo `render.yaml` define o serviço e o endpoint de saúde `/health`.

### Streamlit Community Cloud

Configuração esperada:

- repositório: `viniciussntos/epidemiological-forecaster-clone`;
- branch: `main`;
- arquivo principal: `app/app.py`;
- segredo: `API_URL` apontando para o Render.

## Testes

```powershell
python -m unittest discover -s tests -v
```

Os testes verificam preparação de dados, ausência de vazamento temporal, contrato do dashboard, completude simultânea de clima e dengue, transição dos estados de avaliação e requisito mínimo das métricas de produção.

## Estrutura principal

```text
app/                         FastAPI e Streamlit
models/criticidade_v2/       artefatos de avaliação e produção
src/                         preparação, modelagem e acesso ao Supabase
scripts/                     aplicação do schema operacional
supabase/migrations/         schema PostgreSQL
.github/workflows/           automação diária
tests/                       testes automatizados
render.yaml                  definição da API no Render
```
