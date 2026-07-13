# Plano de Monitoramento — MovieLens MLP Recommender

## 1. Objetivo

Este documento define a estratégia de monitoramento do modelo
`movielens-mlp-recommender`, promovido no MLflow Model Registry pelos aliases:

```text
candidate
staging
production
```

O objetivo é detectar degradação operacional, problemas de qualidade de dados,
mudanças de distribuição, perda de desempenho e falhas de governança.

O projeto atual utiliza o alias:

```text
models:/movielens-mlp-recommender@production
```

como referência da versão aprovada.

No estado atual, o projeto não possui um endpoint público de inferência nem
monitoramento online ativo. Portanto, este plano descreve:

- o monitoramento já viável com os artefatos offline existentes;
- os controles obrigatórios para uma futura API de inferência;
- critérios de alerta, retreinamento e rollback;
- responsabilidades e evidências de auditoria.

## 2. Escopo

O plano cobre quatro dimensões:

1. operação;
2. dados;
3. modelo;
4. governança.

```text
monitoramento
├── operacional
├── dados
├── desempenho do modelo
└── governança e ciclo de vida
```

## 3. Modelo monitorado

| Campo | Valor |
|---|---|
| Nome registrado | `movielens-mlp-recommender` |
| Framework | PyTorch |
| Arquitetura | MLP com embeddings |
| Versão atual | `1` |
| Run ID | `dce4e464561d4af99f1752d0bbcc87bd` |
| Alias de produção | `production` |
| Status | `READY` |
| Responsável | Rafael Costa |
| Quality gate | `RMSE <= 0.26` |

## 4. Baseline de referência

As métricas offline da versão atual são:

| Métrica | Valor |
|---|---:|
| RMSE normalizado | 0.247124 |
| MAE normalizado | 0.197192 |
| MSE normalizado | 0.061070 |
| R² | 0.226679 |
| Median Absolute Error | 0.166689 |
| RMSE na escala original | 0.988496 |
| MAE na escala original | 0.788768 |

A política de promoção utiliza:

```yaml
quality_gate:
  selection_metric: rmse
  maximum_rmse: 0.26
```

O baseline vencedor foi:

```text
ridge_one_hot
```

A MLP não venceu a comparação geral, mas foi promovida porque passou no quality
gate e a política está configurada com:

```yaml
require_comparison_winner: false
```

## 5. Fontes de dados para monitoramento

### 5.1 Fontes já disponíveis

```text
reports/metrics/train_metrics.json
reports/metrics/evaluation_metrics.json
reports/metrics/baseline_metrics.json
reports/metrics/model_comparison.json
models/registry/training_run.json
models/registry/registered_model.json
models/registry/staging_promotion.json
models/registry/production_promotion.json
params.yaml
dvc.lock
```

### 5.2 Fontes futuras para serving online

```text
logs da API
métricas Prometheus
traces
eventos de inferência
dados reais de feedback
avaliações observadas após a previsão
histórico de versão utilizada
```

## 6. Monitoramento operacional

O monitoramento operacional será obrigatório quando houver um serviço de
inferência.

### 6.1 Métricas principais

| Métrica | Objetivo |
|---|---|
| disponibilidade | medir tempo em que o serviço responde |
| latência p50 | acompanhar comportamento típico |
| latência p95 | detectar degradação |
| latência p99 | detectar caudas longas |
| taxa de erro | identificar falhas de aplicação |
| throughput | medir volume de requisições |
| uso de CPU | detectar saturação |
| uso de memória | detectar pressão ou vazamento |
| reinicializações | detectar instabilidade |
| tempo de carregamento | verificar custo de inicialização |
| versão carregada | garantir rastreabilidade |

### 6.2 Limites de alerta sugeridos

| Métrica | Atenção | Crítico |
|---|---:|---:|
| disponibilidade | < 99,5% | < 99,0% |
| latência p95 | > 300 ms | > 500 ms |
| taxa de erro | > 1% | > 5% |
| uso de CPU | > 75% por 10 min | > 90% por 5 min |
| uso de memória | > 80% por 10 min | > 90% por 5 min |
| reinicializações | 2 em 1 hora | 5 em 1 hora |

Esses valores são iniciais e devem ser recalibrados após testes de carga.

## 7. Monitoramento de dados

O modelo depende de identificadores de usuários e itens conhecidos.

### 7.1 Métricas obrigatórias

| Métrica | Descrição |
|---|---|
| usuários desconhecidos | proporção de `user_id` não vistos |
| itens desconhecidos | proporção de `item_id` não vistos |
| campos ausentes | entradas incompletas |
| tipos inválidos | valores incompatíveis com o schema |
| IDs fora do intervalo | risco de erro nos embeddings |
| distribuição de ratings | mudança na variável-alvo |
| frequência por usuário | mudança no perfil de atividade |
| frequência por item | mudança de popularidade |
| tamanho do catálogo | crescimento de novos itens |

### 7.2 Limites sugeridos

| Métrica | Atenção | Crítico |
|---|---:|---:|
| usuários desconhecidos | > 5% | > 10% |
| itens desconhecidos | > 5% | > 10% |
| campos ausentes | > 0,5% | > 2% |
| tipos inválidos | > 0,1% | > 1% |
| IDs fora do intervalo | > 0% | > 0,5% |

Entradas fora do intervalo dos embeddings devem ser bloqueadas ou tratadas por
fallback.

## 8. Data drift

Data drift ocorre quando a distribuição dos dados atuais se afasta da
distribuição usada no treinamento.

### 8.1 Variáveis a acompanhar

```text
user_id
item_id
rating
frequência por usuário
frequência por item
popularidade dos itens
proporção de usuários novos
proporção de itens novos
```

### 8.2 Métodos sugeridos

- Population Stability Index;
- Kolmogorov-Smirnov para variáveis numéricas;
- Jensen-Shannon Distance;
- comparação de histogramas;
- análise de proporções;
- comparação de quantis;
- comparação por janelas temporais.

### 8.3 Faixas de PSI sugeridas

| PSI | Interpretação |
|---:|---|
| < 0,10 | distribuição estável |
| 0,10 a 0,25 | mudança moderada |
| > 0,25 | mudança relevante |

PSI acima de `0,25` deve gerar investigação.

## 9. Monitoramento do modelo

### 9.1 Métricas offline

Quando os ratings reais estiverem disponíveis, recalcular:

```text
RMSE
MAE
MSE
R²
Median Absolute Error
RMSE na escala original
MAE na escala original
```

### 9.2 Métricas de recomendação

Em uma evolução para ranking, acompanhar:

```text
Precision@K
Recall@K
NDCG@K
MAP@K
Hit Rate@K
coverage
diversity
novelty
serendipity
```

### 9.3 Métricas online futuras

```text
CTR
taxa de conversão
tempo de permanência
aceitação de recomendação
retenção
taxa de ocultação
taxa de rejeição
```

## 10. Limites de degradação

O modelo deve ser investigado quando houver:

| Condição | Ação |
|---|---|
| RMSE > 0,26 | bloquear nova promoção |
| RMSE aumentar mais de 5% | investigar |
| RMSE aumentar mais de 10% | retreinar ou rollback |
| MAE aumentar mais de 10% | investigar |
| R² cair abaixo de 0,15 | avaliar degradação |
| cobertura cair mais de 10% | investigar catálogo |
| desconhecidos > 10% | ativar fallback e retreinamento |

## 11. Monitoramento de previsões

A distribuição das previsões deve ser acompanhada.

### 11.1 Verificações

- média;
- desvio-padrão;
- mínimo;
- máximo;
- percentis;
- concentração excessiva;
- valores fora da escala válida;
- mudança por período;
- mudança por segmento.

### 11.2 Alertas

Gerar alerta quando:

- previsões ficarem fora da escala esperada;
- a variância cair de forma abrupta;
- a média mudar significativamente;
- houver concentração anormal em poucos valores;
- houver aumento de erros para grupos específicos.

## 12. Cold start

O modelo atual possui limitação de cold start.

### 12.1 Estratégias de fallback

```text
média global
média por item
itens mais populares
recomendação por conteúdo
embedding reservado para desconhecidos
```

### 12.2 Métricas

| Métrica | Objetivo |
|---|---|
| taxa de fallback | medir dependência da estratégia alternativa |
| fallback por usuário | detectar crescimento de novos usuários |
| fallback por item | detectar crescimento do catálogo |
| desempenho do fallback | comparar com o modelo principal |

## 13. Monitoramento de vieses

O modelo deve ser avaliado quanto a:

- viés de popularidade;
- sobre-exposição de itens frequentes;
- baixa diversidade;
- concentração em poucos itens;
- diferença de desempenho entre segmentos;
- feedback loops.

### 13.1 Métricas sugeridas

```text
cobertura do catálogo
índice de Gini de exposição
long-tail coverage
diversidade intra-lista
popularidade média recomendada
distribuição de exposição por item
```

## 14. Monitoramento de governança

### 14.1 Verificações obrigatórias

- alias `production` aponta para versão válida;
- versão está com status `READY`;
- Run ID existe no MLflow;
- métricas da versão estão disponíveis;
- Model Card está atualizado;
- aprovação de Staging está registrada;
- aprovação de Production está registrada;
- versão do consumidor coincide com o Registry;
- parâmetros correspondem ao `params.yaml`;
- pipeline corresponde ao `dvc.lock`.

### 14.2 Evidências de auditoria

```text
models/registry/training_run.json
models/registry/registered_model.json
models/registry/staging_promotion.json
models/registry/production_promotion.json
docs/model_card.md
docs/architecture.md
docs/monitoring.md
```

## 15. Frequência de monitoramento

### 15.1 Ambiente atual offline

| Atividade | Frequência |
|---|---|
| validação do pipeline | a cada alteração |
| testes automatizados | a cada commit |
| comparação de métricas | a cada treinamento |
| verificação do Registry | a cada promoção |
| revisão do Model Card | a cada nova versão |
| revisão do plano | a cada alteração arquitetural |

### 15.2 Ambiente online futuro

| Atividade | Frequência |
|---|---|
| disponibilidade | contínua |
| latência e erros | contínua |
| uso de recursos | contínua |
| desconhecidos | diária |
| drift | semanal |
| desempenho offline | semanal ou mensal |
| vieses | mensal |
| revisão de thresholds | trimestral |

## 16. Critérios de retreinamento

O retreinamento deve ser considerado quando ocorrer:

- RMSE acima de `0,26`;
- aumento de erro superior a 10%;
- PSI acima de `0,25`;
- usuários desconhecidos acima de 10%;
- itens desconhecidos acima de 10%;
- mudança relevante no catálogo;
- inclusão de volume significativo de dados;
- alteração de arquitetura;
- alteração no pré-processamento;
- incidente de qualidade;
- degradação consistente por múltiplas janelas.

## 17. Fluxo de retreinamento

```text
detecção de degradação
        ↓
coleta de evidências
        ↓
versionamento dos novos dados
        ↓
reexecução do pipeline DVC
        ↓
comparação com baselines
        ↓
quality gate
        ↓
registro no MLflow
        ↓
candidate
        ↓
staging
        ↓
production
```

Comando:

```bash
uv run dvc repro promote_production
```

## 18. Critérios de rollback

Executar rollback quando houver:

- erro crítico de inferência;
- incompatibilidade de schema;
- aumento abrupto da taxa de erro;
- degradação relevante de RMSE;
- incidente de segurança;
- comportamento inesperado;
- concentração anormal de previsões;
- quebra de compatibilidade no consumidor.

## 19. Procedimento de rollback

1. identificar a última versão estável;
2. validar métricas e Model Card;
3. alterar o alias `production`;
4. recarregar o consumidor;
5. executar sanity checks;
6. registrar o incidente;
7. preservar logs e evidências;
8. abrir análise de causa raiz.

Exemplo conceitual:

```python
from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    "movielens-mlp-recommender",
    "production",
    "VERSAO_ESTAVEL",
)
```

## 20. Sanity checks após promoção

Após uma promoção, validar:

- alias correto;
- versão esperada;
- Run ID correto;
- status `READY`;
- carregamento do modelo;
- inferência de exemplo;
- formato da saída;
- ausência de NaN;
- valor dentro da escala esperada;
- metadados de aprovação.

Exemplo:

```bash
uv run python - <<'PY'
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model(
    "models:/movielens-mlp-recommender@production"
)

sample = pd.DataFrame(
    [{"user_id": 10, "item_id": 42}]
)

print(model.predict(sample))
PY
```

## 21. Alertas e severidade

### INFO

- nova versão registrada;
- promoção concluída;
- pipeline reproduzido;
- relatório atualizado.

### WARNING

- aumento moderado de latência;
- PSI entre `0,10` e `0,25`;
- usuários desconhecidos acima de 5%;
- erro offline entre 5% e 10% acima da referência.

### CRITICAL

- indisponibilidade;
- taxa de erro acima de 5%;
- PSI acima de `0,25`;
- RMSE acima de `0,26`;
- usuários ou itens desconhecidos acima de 10%;
- alias `production` inválido;
- versão não carregada;
- incompatibilidade de schema.

## 22. Canais de notificação futuros

- e-mail;
- Slack;
- Microsoft Teams;
- GitHub Issues;
- PagerDuty;
- Grafana Alerting;
- Prometheus Alertmanager.

## 23. Responsabilidades

| Papel | Responsabilidade |
|---|---|
| responsável técnico | analisar métricas e aprovar ações |
| mantenedor do pipeline | corrigir falhas de execução |
| responsável pelo modelo | avaliar degradação |
| responsável pela infraestrutura | tratar incidentes operacionais |
| aprovador | autorizar promoção ou rollback |

Responsável atual:

```text
Rafael Costa
```

## 24. Registro de incidentes

Cada incidente deve registrar:

```text
data e hora
versão
alias
Run ID
sintoma
métrica afetada
impacto
causa raiz
ação corretiva
rollback executado
responsável
status
```

Modelo de registro:

```markdown
## Incidente

- Data:
- Modelo:
- Versão:
- Alias:
- Run ID:
- Sintoma:
- Métrica:
- Impacto:
- Causa:
- Ação:
- Rollback:
- Responsável:
- Status:
```

## 25. Relatório periódico

O relatório deve conter:

- versão atual;
- período analisado;
- disponibilidade;
- latência;
- taxa de erro;
- drift;
- proporção de desconhecidos;
- RMSE;
- MAE;
- cobertura;
- alertas;
- incidentes;
- recomendações;
- decisão de manter, retreinar ou reverter.

## 26. Ferramentas recomendadas

### Estado atual

```text
DVC
MLflow
Pytest
Ruff
mypy
GitHub Actions
Docker Compose
```

### Evolução futura

```text
Prometheus
Grafana
Evidently
OpenTelemetry
Loki
Alertmanager
FastAPI
```

## 27. Integração futura com CI/CD

A esteira pode incluir:

```text
lint
type check
testes
build
execução do pipeline
avaliação
quality gate
registro
staging
aprovação
production
monitoramento
```

A promoção para Production deve permanecer controlada e auditável.

## 28. Limitações atuais

- não existe API pública;
- não existe coleta contínua;
- não existe dashboard;
- não existe alerta automático;
- não existe detecção automática de drift;
- não existe feedback online;
- não existe retreinamento automático;
- não existe monitoramento por segmento.

Essas limitações devem ser removidas antes de uma utilização real em produção.

## 29. Checklist operacional

### Antes da promoção

- [ ] testes passaram;
- [ ] DVC atualizado;
- [ ] MLflow disponível;
- [ ] métricas revisadas;
- [ ] quality gate aprovado;
- [ ] comparação revisada;
- [ ] Model Card atualizado;
- [ ] aprovação registrada.

### Após a promoção

- [ ] alias validado;
- [ ] versão validada;
- [ ] Run ID validado;
- [ ] sanity check executado;
- [ ] metadados persistidos;
- [ ] monitoramento ativo;
- [ ] rollback preparado.

## 30. Referências internas

```text
docs/model_card.md
docs/architecture.md
params.yaml
dvc.yaml
dvc.lock
reports/metrics
models/registry
README.md
```

Este documento deve ser atualizado sempre que houver:

- alteração no quality gate;
- nova versão do modelo;
- novo serviço de inferência;
- novo mecanismo de monitoramento;
- alteração nos thresholds;
- novo incidente;
- mudança no processo de rollback;
- alteração no CI/CD.
