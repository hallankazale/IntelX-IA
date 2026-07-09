# IntelX Verify — MVP com APIs reais permitidas

Sistema web estático para GitHub Pages com:

- Consulta real de CNPJ via BrasilAPI
- Consulta real de domínio via RDAP / Registro.br
- Validação local de CPF
- Validação local de telefone
- Validação local de e-mail
- Histórico local no navegador
- Relatório em PDF via impressão/salvar PDF

## Como rodar localmente

Abra o arquivo `index.html` no navegador.

## Como subir no GitHub Pages

1. Crie um repositório no GitHub.
2. Envie `index.html`, `styles.css`, `app.js` e `README.md` para a raiz.
3. Vá em Settings > Pages.
4. Em Source, escolha Deploy from branch.
5. Escolha branch `main` e pasta `/root`.
6. Salve e aguarde o link público.

## Observação importante sobre CPF e telefone

Este MVP NÃO puxa dados privados por CPF ou telefone.
Ele apenas valida o formato/dígitos. Para consulta cadastral real, use somente APIs contratadas e autorizadas, com base legal, consentimento, logs de auditoria e backend seguro.

## Próxima evolução profissional

Criar backend Node/NestJS para:

- Guardar histórico em Supabase/PostgreSQL
- Proteger chaves de APIs
- Fazer DNS/MX real de e-mails
- Conectar APIs KYC autorizadas
- Criar autenticação de usuários
- Salvar auditoria de cada consulta

