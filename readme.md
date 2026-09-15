# Comparação MLP vs. CNN no CIFAR-10

Projeto da disciplina **IF702** (Centro de Informática — UFPE) cujo objetivo é comparar o desempenho de uma rede neural totalmente conectada (**MLP**) e de uma rede convolucional (**CNN**) na tarefa de classificação de imagens do dataset **CIFAR-10**, avaliando o impacto de arquitetura, hiperparâmetros e *data augmentation* sobre a acurácia final.

---

## Sumário

1. [Objetivo](#objetivo)
2. [Dataset](#dataset)
3. [Estrutura do repositório](#estrutura-do-repositório)
4. [Pipeline de dados](#pipeline-de-dados)
5. [Métrica de avaliação](#métrica-de-avaliação)
6. [Modelo MLP](#modelo-mlp)
7. [Modelo CNN](#modelo-cnn)
8. [Funções de perda: o truque do MSELoss](#funções-de-perda-o-truque-do-mseloss)
9. [Treinamento e Early Stopping](#treinamento-e-early-stopping)
10. [Metodologia experimental (Optuna + WandB)](#metodologia-experimental-optuna--wandb)
11. [Resultados de validação — MLP](#resultados-de-validação--mlp)
12. [Resultados de validação — CNN](#resultados-de-validação--cnn)
13. [Análise de correlação (Optuna)](#análise-de-correlação-optuna)
14. [Resultados finais no conjunto de teste](#resultados-finais-no-conjunto-de-teste)
15. [Discussão dos resultados](#discussão-dos-resultados)
16. [Como reproduzir](#como-reproduzir)
17. [Ferramentas utilizadas](#ferramentas-utilizadas)
18. [Conclusão](#conclusão)

---

## Objetivo

Construir, treinar e comparar dois paradigmas de rede neural — um **MLP** (baseline denso) e uma **CNN** (arquitetura convolucional) — no problema de classificação de imagens do CIFAR-10, conduzindo uma avaliação experimental rigorosa (*experimental evaluation*) baseada em:

- Busca de hiperparâmetros com **Optuna**;
- Rastreamento de experimentos com **Weights & Biases (WandB)**;
- Métricas de desempenho (acurácia, precisão, recall, matriz de confusão);
- Avaliação do impacto de *data augmentation*.

## Dataset

**CIFAR-10**: 60.000 imagens coloridas (RGB, 32×32 px) distribuídas em 10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

O dataset já vem **balanceado** — 5.000 imagens por classe no conjunto de treino/validação (50.000 imagens) e 1.000 por classe no conjunto de teste (10.000 imagens). Isso é relevante metodologicamente: como não há classe majoritária, a **acurácia pura** pode ser usada como métrica principal sem risco de mascarar um modelo que apenas "chuta" a classe mais frequente.

## Estrutura do repositório

```
.
├── requirements.txt          # Dependências do projeto
├── train_mlp.py               # Otimização (Optuna) + treino do MLP
├── train_cnn.py                # Otimização (Optuna) + treino da CNN
├── test.py                    # Avaliação dos melhores modelos no conjunto de teste
├── src/
│   ├── data_loader.py         # Carregamento, split e augmentation do CIFAR-10
│   ├── models_mlp.py          # Arquitetura + loop de treino/validação do MLP
│   ├── models_cnn.py          # Arquitetura + loop de treino/validação da CNN
│   ├── metrics.py             # Cálculo de métricas e plot da matriz de confusão
│   └── utils.py                # EarlyStopping e cronômetro de inferência (TimedModel)
└── test_results/
    ├── best_models_comparison.csv
    ├── cm_best_mlp_<run_id>.png
    └── cm_best_cnn_<run_id>.png
```

## Pipeline de dados

Implementado em `src/data_loader.py`, função `get_dataloaders()`:

1. **Normalização**: todas as imagens passam por `ToTensor()` e `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`, deixando os pixels no intervalo `[-1, 1]`.
2. **Flattening condicional**: quando `is_mlp=True`, é aplicado `torch.flatten(x)`, transformando o tensor `3×32×32` em um vetor de **3.072** posições — a entrada exigida pelo MLP. A CNN mantém o formato de imagem original (necessário para preservar a estrutura espacial 2D).
3. **Split treino/validação**: as 50.000 imagens de treino oficiais do CIFAR-10 são divididas deterministicamente (`torch.Generator().manual_seed(42)`) em:
   - **40.000** imagens de treino;
   - **10.000** imagens de validação.
4. **Conjunto de teste**: as 10.000 imagens oficiais de teste do CIFAR-10 são usadas exclusivamente na avaliação final (nunca durante o tuning).
5. **Data Augmentation (opcional)**: quando `use_augmentation=True`, é criado um segundo dataset com a política `AutoAugment(CIFAR10Policy)`. Em vez de substituir todo o conjunto de treino, o pipeline **combina**:
   - Uma fração do dataset original sem augmentation;
   - O dataset inteiro após augmentation;

## Métrica de avaliação

Como o dataset é balanceado, a **acurácia total** é usada como critério principal de otimização do Optuna. Complementarmente (`src/metrics.py`), são calculados:

- **Precisão** e **Recall** (média macro, via `scikit-learn`);
- **Acurácia por classe** (diagonal da matriz de confusão dividida pelo total de cada classe real);
- **Matriz de confusão** completa, plotada com `seaborn`/`matplotlib` para inspeção qualitativa dos erros.

## Modelo MLP

Definido em `src/models_mlp.py`. Arquitetura *fully-connected* simples e parametrizável:

```python
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, num_layers, neurons_per_layer, activation_function, dropout_rate):
        ...
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, neurons_per_layer))
            layers.append(activation_function())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = neurons_per_layer
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
```

- **Entrada**: vetor de 3.072 valores (imagem 32×32×3 achatada);
- **Construção iterativa**: cada camada oculta empilha `Linear → Ativação → (Dropout opcional)`, e o número de neurônios de saída de uma camada alimenta a entrada da próxima;
- **Dropout condicional**: só é inserido se `dropout_rate > 0`;
- **Camada de saída**: `Linear(in_features, num_classes)` produz os *logits* das 10 classes;
- **Encapsulamento**: todas as camadas são unidas em um único `nn.Sequential`.

Hiperparâmetros variáveis: número de camadas, neurônios por camada, função de ativação (`ReLU`, `Tanh`, `Sigmoid`), taxa de dropout, *learning rate*, `weight_decay` e função de perda.

## Modelo CNN

Definido em `src/models_cnn.py`. Bloco convolucional dinâmico + classificador denso:

**Bloco convolucional:**
```python
for i in range(num_conv_layers):
    layers.append(nn.Conv2d(in_channels, current_filters, kernel_size, stride, padding))
    layers.append(nn.BatchNorm2d(current_filters))
    layers.append(activation_function())
    # recálculo do tamanho espacial: floor((W - K + 2P)/S) + 1
    if pool_size > 1 and current_spatial_size >= pool_size:
        layers.append(nn.MaxPool2d(pool_size, stride=pool_size))
    in_channels = current_filters
    current_filters *= 2   # dobra os filtros a cada camada
```

- Cada camada convolucional é seguida de **BatchNorm2d** e da função de ativação;
- O **tamanho espacial** do *feature map* é recalculado a cada convolução (`floor((W - K + 2P)/S) + 1`) e, quando aplicável, reduzido pela metade pelo `MaxPool2d`;
- O número de **filtros dobra** a cada camada (32 → 64 → 128, ...), seguindo o padrão clássico de CNNs;
- Uma **validação de sanidade** impede combinações inválidas de `kernel_size`/`stride`/`padding`/`pool_size` que zerariam a dimensão espacial ou estourariam um limite de 65.536 conexões na camada densa (nesses casos o Optuna descarta a *trial* via `TrialPruned`).

**Classificador final:**
```python
self.fc_block = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, hidden_1), activation_function(), nn.Dropout(dropout_rate),
    nn.Linear(hidden_1, hidden_2), activation_function(), nn.Dropout(dropout_rate),
    nn.Linear(hidden_2, num_classes)
)
```
- **Achatamento** do *feature map* convolucional em um vetor 1D;
- Duas camadas densas com **redução progressiva** (cada uma reduz à metade o número de neurônios da anterior, com piso mínimo de 64 e 32 respectivamente);
- Camada linear final mapeando para as 10 classes (logits).

Hiperparâmetros variáveis: número de camadas convolucionais, `kernel_size`, `stride`, `padding`, `pool_size`, ativação (`ReLU`, `LeakyReLU`, `GELU`), dropout, `learning rate`, `weight_decay` e função de perda.

## Funções de perda: o truque do MSELoss

Tanto o treino do MLP quanto o da CNN suportam duas funções de perda intercambiáveis via Optuna: `CrossEntropyLoss` e `MSELoss`. Como `MSELoss` não opera nativamente sobre *logits* e rótulos inteiros, o código faz um tratamento específico em `train_epoch`/`validate_epoch`:

- Para `CrossEntropyLoss`: os *logits* brutos do modelo são usados diretamente (a própria função já aplica `LogSoftmax` internamente).
- Para `MSELoss`: os rótulos são convertidos em **one-hot** (`F.one_hot(labels, num_classes=10)`) e os *logits* passam por `softmax` antes do cálculo do erro — garantindo que o MSE seja calculado sobre distribuições de probabilidade comparáveis, e não sobre valores arbitrários de logit.

## Treinamento e Early Stopping

- **Otimizadores**: `AdamW` para o MLP e `Adam` para a CNN.
- **Early Stopping** (`src/utils.py`): implementado como classe própria, monitorando a *loss* de validação:
  - `patience=5`, `min_delta=1e-3`;
  - Sempre que a *loss* de validação melhora (além do `min_delta`), o checkpoint do modelo é salvo em disco (`torch.save(model.state_dict(), path)`);
  - Se a validação não melhora por `patience` épocas consecutivas, o treino é interrompido antecipadamente.
- **Épocas por *trial***: 15, com possibilidade de parada antecipada.
- **Cronometragem de inferência** (`TimedModel`): mede o tempo médio de inferência por *batch* durante a validação (com `torch.cuda.synchronize()` quando GPU disponível), permitindo comparar o custo computacional de MLP e CNN.

## Metodologia experimental (Optuna + WandB)

O processo segue a evolução descrita no slide de apresentação:

1. Estudo do dataset (distribuição balanceada — 5.000 imagens/classe, verificada com `value_counts()`);
2. Pré-processamento (normalização e split treino/validação/teste);
3. Criação da classe e de um modelo base ("MVP");
4. Incremento gradual de hiperparâmetros a serem otimizados;
5. Treinamento sucessivo dos modelos;
6. Introdução de *Data Augmentation*;
7. Novo ciclo de treinamento com os dados aumentados;
8. Avaliação final dos resultados no conjunto de teste.

Cada rodada de *tuning* usa:
- **Optuna** (`TPESampler`, estudo persistido em SQLite `cifar10_optuna.db`) com **20 trials**;
- **15 épocas** por *trial* (sujeitas a *early stopping*);
- Critério de otimização: **maior acurácia de validação**;
- Cada *trial* é registrada como uma *run* no **WandB** (projeto `miniprojeto1-cifar10`, entidade `Proj-IF702`), agrupada em `mlp_optimization` ou `cnn_optimization`, com log de *loss*, acurácia, precisão, recall, tempo de inferência e acurácia por classe a cada época, além da matriz de confusão da melhor época.

**Espaço de busca do MLP** (`train_mlp.py`):

| Hiperparâmetro     | Espaço de busca                          |
|---------------------|-------------------------------------------|
| `lr`                 | log-uniforme entre 1e-4 e 1e-1              |
| `num_layers`         | 1 a 3                                        |
| `batch_size`         | {32, 64, 128}                               |
| `criterion`          | {CrossEntropyLoss, MSELoss}                 |
| `activation`         | {ReLU, Tanh, Sigmoid}                       |
| `dropout_rate`       | 0.0 a 0.3                                    |
| `neurons_per_layer`  | {64, 128, 256, 512}                          |
| `weight_decay`       | log-uniforme entre 1e-6 e 1e-2               |

**Espaço de busca da CNN** (`train_cnn.py`):

| Hiperparâmetro     | Espaço de busca                                  |
|---------------------|-----------------------------------------------------|
| `lr`                 | log-uniforme entre 1e-4 e 1e-1                        |
| `num_conv_layers`    | {1, 2, 3}                                            |
| `batch_size`         | {64, 128, 256}                                       |
| `criterion`          | {CrossEntropyLoss, MSELoss}                          |
| `activation`         | {ReLU, LeakyReLU, GELU}                              |
| `kernel_size`        | {3, 5}                                                |
| `stride`             | {1, 2}                                                |
| `padding`            | {0, 1, 2}                                             |
| `pool_size`          | {1 (sem pooling), 2 (pool 2×2)}                        |
| `dropout_rate`       | 0.0 a 0.3                                             |
| `weight_decay`       | log-uniforme entre 1e-6 e 1e-2                        |

## Resultados de validação — MLP

### Sem *Data Augmentation*

| Rank | Trial | Acc. Val. | LR | Camadas | Batch | Loss | Ativação | Dropout | Neurônios/camada | Tempo inf. (s/batch) |
|------|-------|-----------|-----|----------|--------|------|-----------|----------|-------------------|------------------------|
| 1 | #11 | **0.5380** | ≈2.31e-4 | 1 | 32 | MSELoss | ReLU | ≈0.054 | 256 | 0.000386 |
| 2 | #19 | 0.5309 | ≈5.50e-4 | 2 | 32 | MSELoss | ReLU | ≈0.033 | 256 | 0.000408 |
| 3 | #13 | 0.5246 | ≈7.77e-4 | 2 | 32 | MSELoss | ReLU | ≈0.006 | 256 | 0.000424 |

**Padrão observado**: os três melhores modelos usaram `MSELoss`, `batch_size=32`, ativação `ReLU`, **baixa regularização** (dropout muito pequeno) e **poucas camadas** (1 ou 2).

> **Questionamento importante**: contraintuitivamente, o modelo com a *maior* acurácia (Trial #11) foi o mais simples de todos — apenas **uma única camada oculta**. Isso sugere que, para um MLP recebendo pixels achatados do CIFAR-10, aumentar a profundidade da rede não traz ganho — e pode até prejudicar o aprendizado, dado o ruído introduzido pela ausência de estrutura espacial nos dados de entrada.

**Erros mais comuns (Trial #11)**: analisando a matriz de confusão, o modelo confunde sistematicamente pares de classes visualmente semelhantes — gato × cachorro, avião × barco, automóvel × caminhão e cervo × pássaro — evidenciando a dificuldade do MLP em capturar texturas e formas finas quando a imagem é tratada como um vetor sem relação espacial entre pixels.

### Com *Data Augmentation*

| Rank | Trial | Acc. Val. | LR | Camadas | Batch | Loss | Ativação | Dropout | Neurônios/camada | Tempo inf. (s/batch) |
|------|-------|-----------|-----|----------|--------|------|-----------|----------|-------------------|------------------------|
| 1 | #14 | 0.5196 | ≈6.01e-4 | 3 | 64 | CrossEntropyLoss | Sigmoid | ≈0.220 | 512 | 0.001560 |
| 2 | #11 | 0.5168 | ≈8.90e-4 | 3 | 64 | CrossEntropyLoss | Sigmoid | ≈0.275 | 512 | 0.001524 |
| 3 | #17 | 0.5116 | ≈1.18e-3 | 3 | 128 | CrossEntropyLoss | Sigmoid | ≈0.287 | 512 | 0.002507 |

**Padrão observado**: aqui os três melhores usaram `CrossEntropyLoss`, ativação `Sigmoid` e **3 camadas** — um padrão distinto do experimento sem augmentation.

> **Questionamento importante**: apesar de o *pipeline* de dados aumentados ter sido pensado para melhorar a generalização, as *trials* com *Data Augmentation* tiveram, em geral, **desempenho de validação pior** do que as sem augmentation (melhor acurácia caiu de 0.5380 para 0.5196). Uma hipótese razoável é que transformações de imagem (rotações, cortes, distorções de cor da política `AutoAugment`) tendem a **quebrar ainda mais** a pouca estrutura espacial que restava útil para um classificador puramente denso — o MLP não tem mecanismos (como convoluções) para se beneficiar de invariâncias geométricas, então o ruído extra da augmentation pesa mais do que ajuda.

## Resultados de validação — CNN

### Sem *Data Augmentation*

| Rank | Trial | Acc. Val. | LR | Conv. layers | Batch | Loss | Ativação | Kernel | Stride | Padding | Pool | Dropout | Tempo inf. (s/batch) |
|------|-------|-----------|-----|---------------|--------|------|-----------|---------|---------|----------|-------|----------|------------------------|
| 1 | #20 | **0.7896** | ≈8.50e-4 | 3 | 64 | CrossEntropyLoss | GELU | 3 | 1 | 1 | 2 | ≈0.225 | 0.017423 |
| 2 | #15 | 0.7876 | ≈8.32e-4 | 3 | 64 | CrossEntropyLoss | GELU | 3 | 1 | 1 | 2 | ≈0.238 | 0.017593 |
| 3 | #9  | 0.7819 | ≈1.53e-3 | 3 | 64 | CrossEntropyLoss | ReLU | 3 | 1 | 1 | 2 | ≈0.220 | 0.019555 |

**Padrão observado**: os três melhores modelos compartilham **exatamente** o mesmo conjunto de decisões arquiteturais: `num_conv_layers=3`, `batch_size=64`, `criterion=CrossEntropyLoss`, `kernel_size=3`, `stride=1`, `padding=1`, `pool_size=2` — variando apenas `lr`, ativação e dropout. Isso indica uma região bastante estável e favorável do espaço de busca para a CNN.

### Com *Data Augmentation*

| Rank | Trial | Acc. Val. | LR | Conv. layers | Batch | Loss | Ativação | Kernel | Stride | Padding | Pool | Dropout | Weight decay | Tempo inf. (s/batch) |
|------|-------|-----------|-----|---------------|--------|------|-----------|---------|---------|----------|-------|----------|----------------|------------------------|
| 1 | #24 | **0.8094** | ≈7.43e-4 | 3 | 64 | CrossEntropyLoss | ReLU | 5 | 1 | 2 | 2 | ≈0.172 | ≈1.23e-5 | 0.018173 |
| 2 | #22 | 0.8029 | ≈4.43e-4 | 3 | 64 | CrossEntropyLoss | ReLU | 5 | 1 | 2 | 2 | ≈0.200 | ≈6.53e-5 | 0.018432 |
| 3 | #25 | 0.7980 | ≈4.25e-4 | 3 | 64 | CrossEntropyLoss | ReLU | 5 | 1 | 2 | 2 | ≈0.172 | ≈1.22e-5 | 0.018296 |

> **Diferente do MLP, a CNN se beneficiou do *Data Augmentation***: a melhor acurácia de validação subiu de 0.7896 (sem augmentation) para **0.8094** (com augmentation). Isso é esperado — as convoluções são naturalmente mais robustas a pequenas variações geométricas e de cor introduzidas pela augmentation, aproveitando a diversidade extra sem perder a noção de vizinhança espacial dos pixels.

## Análise de correlação (Optuna)

A análise de importância de hiperparâmetros do Optuna sobre a acurácia de validação da CNN mostra que:

- **`lr` (learning rate)** é o hiperparâmetro de **maior importância** — pequenas variações na taxa de aprendizado impactam fortemente o resultado final;
- Em seguida aparecem **`num_conv_layers`**, **`padding`** e **`stride`**, reforçando que a geometria da convolução (como o *feature map* é reduzido a cada camada) é um fator estrutural relevante;
- O uso de **`CrossEntropyLoss`** aparece com correlação positiva consistente em relação à acurácia, enquanto **`MSELoss`** aparece com correlação negativa — corroborando o padrão observado nas melhores *trials*, que usaram exclusivamente `CrossEntropyLoss`;
- `dropout_rate`, `batch_size` e a escolha da função de ativação têm impacto mais discreto, mas ainda perceptível, especialmente combinados a taxas de aprendizado maiores.

O gráfico de evolução das *trials* do Optuna (acurácia de validação × ordem das *trials*) mostra uma tendência de subida ao longo dos 20 experimentos (R² ≈ 0.34 na reta de tendência), evidenciando que o otimizador bayesiano (TPE) foi progressivamente direcionando a busca para regiões mais promissoras do espaço de hiperparâmetros.

## Resultados finais no conjunto de teste

Ao final da otimização, `test.py` busca automaticamente — via API do WandB — a *run* com maior `val_acc` em cada grupo (`mlp_optimization` e `cnn_optimization`), baixa o checkpoint `.pth` correspondente, reconstrói a arquitetura a partir da configuração salva e avalia no conjunto de **teste** (nunca visto durante o tuning):

| Modelo | WandB Run ID | Val Acc | **Test Acc** | Test Precision | Test Recall | Test Loss | Inf. Time (s/batch) | Batch Size | LR | Ativação | Loss Fn |
|--------|----------------|----------|----------------|-------------------|----------------|-------------|------------------------|--------------|-----|-----------|-----------|
| **MLP** | `c45l7zpm` | 0.5309 | **51,56 %** | 0.5139 | 0.5156 | 0.0629 | 0.000949 | 32 | ≈2.31e-4 | ReLU | MSELoss |
| **CNN** | `vtm1gwzb` | 0.8024 | **79,33 %** | 0.7957 | 0.7933 | 0.6285 | 0.138940 | 64 | ≈4.43e-4 | ReLU | CrossEntropyLoss |

A CNN vencedora usa: `num_conv_layers=3`, `kernel_size=5`, `stride=1`, `padding=2`, `pool_size=2`, `dropout_rate≈0.1995`, `weight_decay≈6.53e-5`.
O MLP vencedor usa: `num_layers=1`, `neurons_per_layer=256`, `dropout_rate≈0.0544`, `weight_decay≈6.37e-6`.

A diferença é expressiva: a **CNN supera o MLP em quase 28 pontos percentuais de acurácia** no teste, ao custo de um tempo de inferência por *batch* cerca de **146× maior** (0.1389s vs. 0.00095s), reflexo do custo computacional adicional das convoluções e do classificador denso maior.

## Discussão dos resultados

**Por que o MLP teve desempenho tão inferior?**

1. **Dados planificados** — o vetor de 3.072 valores perde qualquer noção de "linha" e "coluna" da imagem original; pixels vizinhos deixam de estar próximos no vetor de entrada;
2. **Ausência de pesos compartilhados** — diferentemente dos filtros convolucionais, cada neurônio do MLP aprende pesos totalmente independentes para cada posição de pixel, exigindo muito mais parâmetros e dados para aprender os mesmos padrões visuais;
3. **Nenhuma relação de proximidade espacial** — o MLP não tem qualquer viés indutivo (*inductive bias*) que favoreça padrões locais (bordas, texturas, formas), que são justamente o que compõe a maior parte da informação relevante em uma imagem natural.

Como consequência, o MLP tem dificuldade sistemática em separar classes que compartilham silhueta/cor/textura geral (gato × cachorro, avião × barco, cervo × pássaro, automóvel × caminhão), enquanto a CNN, ao preservar a estrutura 2D e aplicar filtros compartilhados, consegue extrair características visuais muito mais discriminativas — refletido diretamente nos quase 28 p.p. de vantagem em acurácia de teste.

## Como reproduzir

```bash
# 1. Crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instale as dependências
python -m pip install -r requirements.txt

# 3. Configure sua conta do Weights & Biases
wandb login

# 4. Rode a otimização de hiperparâmetros (20 trials, 15 épocas cada)
python train_mlp.py
python train_cnn.py

# 5. Avalie os melhores modelos de cada grupo no conjunto de teste
python test.py

# 6. Desative o ambiente virtual ao final
deactivate
```

Os resultados de teste (métricas + matrizes de confusão) são salvos automaticamente em `test_results/`.

## Ferramentas utilizadas

- **PyTorch / TorchVision** — definição dos modelos, *dataloaders* e transformações do CIFAR-10;
- **Optuna** (`TPESampler`) — busca bayesiana de hiperparâmetros, com persistência em SQLite;
- **Weights & Biases (WandB)** — rastreamento de experimentos, versionamento de checkpoints e visualização de métricas/gráficos de importância e correlação;
- **AdamW / Adam** — otimizadores usados no MLP e na CNN, respectivamente;
- **scikit-learn** — cálculo de acurácia, precisão, recall e matriz de confusão;
- **Matplotlib / Seaborn** — visualização das matrizes de confusão.

## Conclusão

O experimento confirma, de forma quantitativa, uma intuição fundamental de visão computacional: **arquiteturas que exploram a estrutura espacial das imagens (CNNs) superam largamente redes densas genéricas (MLPs)** na tarefa de classificação de imagens, mesmo quando ambas passam por um processo comparável e rigoroso de otimização de hiperparâmetros (mesmo orçamento de *trials*, épocas e critério de seleção). Os resultados também mostram que o benefício do *Data Augmentation* **depende do viés indutivo do modelo**: ele ajudou a CNN a generalizar melhor, mas prejudicou o MLP, que não tem mecanismos para tirar proveito das invariâncias geométricas introduzidas pelas transformações.

**Resumo comparativo final:**

| | MLP | CNN |
|---|-----|-----|
| Melhor arquitetura | 1 camada, 256 neurônios | 3 camadas conv., kernel 5, 64 batch |
| Acurácia de teste | 51,56 % | **79,33 %** |
| Custo de inferência | Muito baixo | ~146× maior que o MLP |
| Efeito do Data Augmentation | Negativo | Positivo |
