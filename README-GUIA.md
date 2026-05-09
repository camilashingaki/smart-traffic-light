# Guia de Contribuição 🚦

Este documento contém instruções para configuração do ambiente, organização do projeto, commits e fluxo de colaboração no GitHub.

---

# Estrutura do Projeto

```text id="b3j94t"
smart-traffic-light/
│
├── computer-vision/
├── machine-learning/
├── esp32-controller/
├── integration/
└── docs/
```

# Configuração Inicial

## 1. Instalar Git

Verifique se o Git está instalado:

```bash id="i9lq0k"
git --version
```

Caso não esteja instalado:

[Git SCM](https://git-scm.com/)

---

# Configuração SSH (Obrigatória)

O GitHub não aceita mais autenticação por senha no terminal.
Por isso, todos os integrantes devem configurar uma chave SSH.

---

## 1. Verificar se já existe chave SSH

```bash id="ff0j5r"
ls ~/.ssh
```

Se aparecer algo como:

```text id="apg0qv"
id_ed25519
id_ed25519.pub
```

a chave já existe.

---

## 2. Criar chave SSH

```bash id="s4w9zi"
ssh-keygen -t ed25519 -C "seu-email@gmail.com"
```

Pressione `Enter` quando ele pedir o caminho, e também quando ele pedir senha (é o padrão).

---

## 3. Iniciar agente SSH

```bash id="qv5d1m"
eval "$(ssh-agent -s)"
```

---

## 4. Adicionar chave SSH

```bash id="45mg92"
ssh-add ~/.ssh/id_ed25519
```

---

## 5. Copiar chave pública

```bash id="vjlwm0"
cat ~/.ssh/id_ed25519.pub
```

Copie todo o conteúdo exibido.

---

## 6. Adicionar chave no GitHub

Abrir:

[GitHub SSH Keys](https://github.com/settings/keys)

Depois:

* clicar em `New SSH Key`
* colar a chave
* salvar

---

## 7. Testar conexão

```bash id="dl3lmy"
ssh -T git@github.com
```

Se estiver funcionando, aparecerá algo parecido com:

```text id="79kz6u"
Hi username! You've successfully authenticated.
```

---

# Exemplo Prático de Contribuição 🚦

Este exemplo mostra exatamente como um integrante deve adicionar um código novo na parte do ESP32.

---

# Cenário

Imagine que você fez um código novo para controlar os LEDs do semáforo e quer enviar para o GitHub.

O arquivo ficou em:

```text id="dsvd5t"
esp32-controller/src/semaforo.cpp
```

---

# Passo 1 — Entrar no projeto

Abra o terminal e vá até a pasta do projeto:

```bash id="6m8i7n"
cd smart-traffic-light
```

---

# Passo 2 — Atualizar o projeto

Antes de começar:

```bash id="muqzkl"
git pull
```

Isso baixa as alterações mais recentes do grupo.

---

# Passo 3 — Criar sua branch

Exemplo para alguém trabalhando no ESP32:

```bash id="h8r6lf"
git checkout -b feature/esp32-semaforo
```

Agora tudo que você fizer ficará separado da branch principal.

---

# Passo 4 — Adicionar seu código

Coloque seus arquivos dentro da pasta correta.

Exemplo:

```text id="q1q8sk"
esp32-controller/
└── src/
    └── semaforo.cpp
```

---

# Passo 5 — Verificar alterações

```bash id="q8v0v2"
git status
```

O Git mostrará os arquivos modificados.

Exemplo:

```text id="5wygcc"
modified: esp32-controller/src/semaforo.cpp
```

---

# Passo 6 — Adicionar arquivos

```bash id="n99q5d"
git add .
```

---

# Passo 7 — Criar commit

Exemplo:

```bash id="hx8gb5"
git commit -m "feat: adiciona controle inicial do semáforo"
```

---

# Passo 8 — Enviar para o GitHub

```bash id="jqlj8v"
git push
```

---

# Resultado

Seu código agora estará salvo no GitHub na sua branch.

Depois disso:

* abrir Pull Request
  ou
* pedir para integrar na `main`

---

# Exemplo Completo

Fluxo inteiro:

```bash id="o5c0wh"
cd smart-traffic-light

git pull

git checkout -b feature/esp32-semaforo

git status

git add .

git commit -m "feat: adiciona controle inicial do semáforo"

git push
```

---

# Importante ⚠️

Cada integrante deve trabalhar principalmente na sua área:

| Área                | Pasta               |
| ------------------- | ------------------- |
| Visão Computacional | `computer-vision/`  |
| Machine Learning    | `machine-learning/` |
| ESP32               | `esp32-controller/` |
| Integração          | `integration/`      |
| Documentos          | `docs/`             |

---

# Exemplos de Branches

```text id="lnk2yb"
feature/opencv-detection
feature/ml-model
feature/esp32-semaforo
feature/serial-communication
```
---

# Tipos de Commit

| Tipo       | Uso                     |
| ---------- | ----------------------- |
| `feat`     | nova funcionalidade     |
| `fix`      | correção de bug         |
| `docs`     | documentação            |
| `refactor` | reorganização de código |
| `test`     | testes                  |
| `style`    | formatação              |
| `chore`    | manutenção geral        |

---

# Boas Práticas

✅ Fazer commits pequenos e organizados

✅ Trabalhar apenas na sua área

✅ Testar antes de enviar

✅ Atualizar documentação quando necessário

✅ Manter o código organizado

---

# Evitar

❌ Subir vídeos pesados

❌ Subir ambientes virtuais (`venv/`)

❌ Alterar código de outras áreas sem avisar

❌ Fazer commits genéricos como:

```text id="ljgxph"
"arrumei coisas"
```

---

# Commits Recomendados

✅ Bons exemplos:

```bash id="l8a4y6"
git commit -m "feat: adiciona contagem de carros"
```

```bash id="9phd2m"
git commit -m "fix: corrige leitura da câmera"
```

```bash id="qg2ckr"
git commit -m "docs: adiciona arquitetura do sistema"
```

---

# Arquivos Ignorados

Os seguintes arquivos NÃO devem ser enviados ao GitHub:

```text id="p4j4tz"
*.mp4
*.avi
venv/
__pycache__/
```

Esses arquivos devem estar no `.gitignore`.
