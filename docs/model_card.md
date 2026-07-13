# Model Card — MovieLens MLP Recommender

## 1. Identificação do modelo

| Campo | Valor |
|---|---|
| Nome registrado | `movielens-mlp-recommender` |
| Tipo | Rede neural MLP com embeddings |
| Framework | PyTorch |
| Versão registrada | `1` |
| Run ID de origem | `dce4e464561d4af99f1752d0bbcc87bd` |
| Alias inicial | `candidate` |
| Alias de homologação | `staging` |
| Alias de produção | `production` |
| Status no Model Registry | `READY` |
| Responsável | Rafael Costa |
| Data da documentação | 13 de julho de 2026 |

O modelo está disponível no MLflow Model Registry pelas seguintes URIs:

```text
models:/movielens-mlp-recommender@candidate
models:/movielens-mlp-recommender@staging
models:/movielens-mlp-recommender@production
```

Os aliases são utilizados como referência principal para o ciclo de vida do modelo. O estágio legado do MLflow foi mantido sincronizado para compatibilidade com a interface do MLflow 2.14.

## 2. Visão geral

O `movielens-mlp-recommender` é um modelo de recomendação baseado em uma rede neural do tipo Multilayer Perceptron (MLP).

O modelo recebe os identificadores de um usuário e de um filme e estima a avaliação que esse usuário atribuiria ao item.

O projeto foi desenvolvido como uma implementação educacional de Machine Learning Engineering e MLOps, incluindo:

- organização modular do código;
- treinamento com PyTorch;
- comparação com baselines do Scikit-Learn;
- versionamento de dados e pipeline com DVC;
- rastreamento de experimentos com MLflow;
- registro e versionamento no MLflow Model Registry;
- quality gate para promoção;
- aliases `candidate`, `staging` e `production`;
- testes automatizados;
- ambiente reproduzível com `uv`, Docker e `uv.lock`.

## 3. Uso pretendido

O modelo foi desenvolvido para:

- estimar ratings de filmes a partir de pares `user_id` e `item_id`;
- demonstrar uma arquitetura neural aplicada a sistemas de recomendação;
- comparar uma MLP com modelos baseline;
- demonstrar rastreabilidade e governança de modelos;
- servir como base para experimentos acadêmicos de recomendação;
- demonstrar um pipeline reproduzível de Machine Learning.

## 4. Usos fora do escopo

O modelo não deve ser utilizado diretamente para:

- decisões de alto impacto sobre pessoas;
- concessão de crédito, emprego, saúde ou segurança;
- recomendação comercial real sem validação adicional;
- usuários ou filmes ausentes do conjunto utilizado no treinamento;
- previsão de preferências em outros domínios;
- recomendação em tempo real sem testes de carga e latência;
- substituição de avaliações humanas ou pesquisas com usuários.

O alias `production` representa a aprovação da versão no Model Registry do projeto. Ele não significa que o modelo esteja sendo utilizado por um serviço público ou por uma aplicação comercial real.

## 5. Dataset

O projeto utiliza o dataset MovieLens 100K.

As observações representam avaliações de filmes e possuem os principais campos:

| Campo | Descrição |
|---|---|
| `user_id` | Identificador do usuário |
| `item_id` | Identificador do filme |
| `rating` | Avaliação atribuída ao filme |
| `timestamp` | Momento em que a avaliação foi registrada |

O pipeline utiliza as seguintes proporções configuradas:

```yaml
test_size: 0.2
valid_size: 0.1
seed: 42
```

Os dados são separados em conjuntos de treinamento, validação e teste. A semente fixa permite repetir a divisão e o treinamento nas mesmas condições.

### Limitações do dataset

- Os dados representam um contexto histórico específico.
- As avaliações são fornecidas voluntariamente pelos usuários.
- Usuários mais ativos possuem maior representação.
- Filmes populares tendem a receber mais avaliações.
- O conjunto não representa todos os perfis de usuários ou tipos de conteúdo.
- O dataset não contém contexto atual de navegação, intenção ou sessão.
- O projeto utiliza principalmente identificadores, sem informações semânticas detalhadas sobre os filmes.

## 6. Entradas e saídas

### Entrada esperada

| Campo | Tipo esperado | Descrição |
|---|---|---|
| `user_id` | inteiro | Usuário conhecido pelo modelo |
| `item_id` | inteiro | Filme conhecido pelo modelo |

Exemplo conceitual:

```json
{
  "user_id": 10,
  "item_id": 42
}
```

### Saída

O modelo produz uma estimativa numérica do rating esperado para o par usuário-filme.

A aplicação que consumir o modelo deve garantir que a previsão seja interpretada dentro da escala de ratings do dataset.

## 7. Arquitetura do modelo

A arquitetura principal utiliza:

| Parâmetro | Valor |
|---|---:|
| Dimensão dos embeddings | 64 |
| Dimensão da camada oculta | 128 |
| Épocas | 5 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Seed | 42 |

O modelo aprende representações vetoriais para usuários e filmes. Essas representações são utilizadas pela MLP para estimar a afinidade entre o usuário e o item.

## 8. Treinamento

O treinamento é executado por meio do pipeline DVC:

```text
download
   ↓
preprocess
   ↓
feature_eng
   ↓
train
   ↓
evaluate
```

Os parâmetros são centralizados em `params.yaml`.

As dependências do ambiente são controladas por `pyproject.toml` e `uv.lock`.

O artefato treinado é salvo em:

```text
models/checkpoints/model.pt
```

A execução correspondente foi registrada no MLflow com o Run ID:

```text
dce4e464561d4af99f1752d0bbcc87bd
```

## 9. Desempenho

### Métricas do modelo MLP

| Métrica | Valor |
|---|---:|
| RMSE normalizado | 0.247124 |
| MAE normalizado | 0.197192 |
| MSE normalizado | 0.061070 |
| R² | 0.226679 |
| Median Absolute Error | 0.166689 |
| RMSE na escala original dos ratings | 0.988496 |
| MAE na escala original dos ratings | 0.788768 |

Os valores completos estão disponíveis em:

```text
reports/metrics/evaluation_metrics.json
```

### Quality gate

A política de promoção configurada foi:

```yaml
selection_metric: rmse
maximum_rmse: 0.26
```

Resultado:

| Verificação | Valor |
|---|---:|
| RMSE observado | 0.247124 |
| Limite máximo | 0.260000 |
| Resultado | `passed` |

O modelo atendeu ao limite definido e foi considerado elegível para promoção.

## 10. Comparação com baselines

O modelo foi comparado com:

- `dummy_mean`;
- `dummy_median`;
- `ridge_one_hot`.

O vencedor da comparação pela métrica RMSE foi `ridge_one_hot`.

O MLP não foi o modelo com o menor erro no conjunto de teste.

A promoção da MLP foi permitida porque:

- o modelo passou pelo quality gate;
- a política `require_comparison_winner` está configurada como `false`;
- a rede neural é o modelo central da implementação educacional;
- o resultado inferior ao baseline está documentado de forma explícita.

Essa decisão não deve ser interpretada como evidência de que a MLP é a melhor alternativa para uma implantação comercial.

Os resultados completos estão em:

```text
reports/metrics/baseline_metrics.json
reports/metrics/model_comparison.json
```

## 11. Governança e ciclo de vida

O ciclo utilizado foi:

```text
candidate
   ↓
staging
   ↓
production
```

A promoção para Staging exigiu:

- versão registrada com status `READY`;
- correspondência entre a versão e o Run ID;
- aprovação do quality gate;
- registro do responsável e da justificativa.

A promoção para Production exigiu adicionalmente:

- passagem prévia da mesma versão pelo alias `staging`;
- validação das métricas;
- registro da aprovação de produção.

Os metadados de auditoria estão armazenados em:

```text
models/registry/registered_model.json
models/registry/staging_promotion.json
models/registry/production_promotion.json
```

Aprovador registrado: `Rafael Costa`.

## 12. Limitações conhecidas

### Cold start

O modelo depende de usuários e filmes observados no treinamento. Identificadores novos não possuem embeddings aprendidos.

Uma aplicação em produção precisaria implementar uma estratégia alternativa, como:

- média global;
- média por item;
- itens mais populares;
- recomendação baseada em conteúdo;
- embedding reservado para valores desconhecidos.

### Ausência de contexto

O modelo não utiliza:

- gênero do filme;
- descrição;
- elenco;
- horário;
- dispositivo;
- histórico recente da sessão;
- contexto temporal;
- localização;
- intenção atual do usuário.

### Generalização

O desempenho foi medido somente sobre a divisão de teste do MovieLens 100K. Não há garantia de desempenho equivalente em outros datasets ou ambientes reais.

### Métrica offline

RMSE e MAE medem erro de previsão de ratings, mas não medem diretamente:

- satisfação do usuário;
- diversidade;
- novidade;
- cobertura do catálogo;
- taxa de cliques;
- retenção;
- impacto comercial.

### Modelo não vencedor

O baseline `ridge_one_hot` obteve desempenho melhor que a MLP na métrica de seleção. Esse resultado indica que maior complexidade não garantiu melhor generalização neste experimento.

## 13. Vieses e riscos

### Viés de popularidade

Filmes com mais avaliações possuem mais dados disponíveis e podem ser favorecidos.

### Viés de seleção

As avaliações refletem apenas usuários que decidiram avaliar determinado filme.

### Viés de usuários ativos

Usuários com muitas avaliações podem influenciar mais o aprendizado das representações.

### Feedback loop

Caso recomendações futuras sejam baseadas apenas nas previsões do modelo, itens já populares podem receber ainda mais exposição.

### Homogeneização

A otimização exclusiva do erro de rating pode reduzir diversidade e novidade nas recomendações.

## 14. Cenários de falha

O modelo pode produzir resultados inadequados quando:

- recebe um `user_id` desconhecido;
- recebe um `item_id` desconhecido;
- os campos estão ausentes ou possuem tipos incorretos;
- os identificadores estão fora dos limites dos embeddings;
- a distribuição dos dados muda significativamente;
- novos padrões de preferência surgem;
- o catálogo contém muitos itens sem histórico;
- os dados de entrada não passam pelo mesmo pré-processamento;
- o modelo é carregado com parâmetros incompatíveis;
- o serviço de inferência utiliza uma versão diferente da aprovada.

A aplicação consumidora deve validar o schema antes de executar a inferência.

## 15. Considerações éticas

O sistema deve ser utilizado como mecanismo de apoio à recomendação, e não como uma avaliação definitiva dos interesses de uma pessoa.

As recomendações devem permitir:

- diversidade de itens;
- possibilidade de descoberta;
- transparência sobre personalização;
- mecanismos para corrigir preferências;
- alternativas não personalizadas.

Nenhum atributo sensível deve ser inferido a partir das recomendações.

## 16. Monitoramento recomendado

Em uma implantação real, devem ser acompanhadas:

### Métricas operacionais

- latência de inferência;
- taxa de erros;
- disponibilidade;
- consumo de memória;
- consumo de CPU ou GPU;
- volume de requisições;
- versão utilizada em cada previsão.

### Métricas de dados

- proporção de usuários desconhecidos;
- proporção de itens desconhecidos;
- valores ausentes;
- alterações nas distribuições;
- crescimento do catálogo;
- frequência das avaliações.

### Métricas do modelo

- RMSE;
- MAE;
- cobertura;
- diversidade;
- novidade;
- distribuição das previsões;
- diferença entre previsões e avaliações observadas.

O plano detalhado será mantido em `docs/monitoring.md`.

## 17. Retreinamento

O retreinamento deve ser considerado quando ocorrer:

- degradação relevante de RMSE ou MAE;
- aumento de usuários ou itens desconhecidos;
- mudança significativa na distribuição dos ratings;
- inclusão de um volume relevante de novos dados;
- alteração da arquitetura ou do pré-processamento;
- incidente relacionado à qualidade das recomendações.

Toda nova versão deverá repetir:

```text
treinamento
→ avaliação
→ comparação
→ registro
→ staging
→ production
```

## 18. Reprodutibilidade

O projeto utiliza:

- Git para código e documentação;
- DVC para dados, artefatos e pipeline;
- MLflow para experimentos e Model Registry;
- `params.yaml` para parâmetros;
- `uv.lock` para dependências;
- Docker Compose para os serviços;
- seed fixa para reduzir variações;
- testes automatizados com Pytest;
- Ruff, mypy e pre-commit para qualidade.

Com o ambiente configurado, o pipeline pode ser reproduzido com:

```bash
uv sync --locked
docker compose up -d mlflow
uv run dvc repro
```

## 19. Artefatos relacionados

| Artefato | Caminho |
|---|---|
| Modelo treinado | `models/checkpoints/model.pt` |
| Metadados do treinamento | `models/registry/training_run.json` |
| Registro do modelo | `models/registry/registered_model.json` |
| Promoção Staging | `models/registry/staging_promotion.json` |
| Promoção Production | `models/registry/production_promotion.json` |
| Métricas de treinamento | `reports/metrics/train_metrics.json` |
| Métricas de avaliação | `reports/metrics/evaluation_metrics.json` |
| Métricas dos baselines | `reports/metrics/baseline_metrics.json` |
| Comparação dos modelos | `reports/metrics/model_comparison.json` |
| Parâmetros | `params.yaml` |
| Pipeline | `dvc.yaml` |

## 20. Histórico de versões

| Versão | Data | Situação |
|---|---|---|
| 1 | 13/07/2026 | Registrada, validada e promovida para Production |

## 21. Contato e responsabilidade

Responsável técnico: `Rafael Costa`.

Este Model Card deve ser atualizado sempre que houver:

- nova versão do modelo;
- alteração do dataset;
- mudança no pré-processamento;
- mudança de arquitetura;
- alteração do quality gate;
- nova promoção para Production;
- identificação de novos riscos ou limitações.
