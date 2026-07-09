const pages=[...document.querySelectorAll('.page')];
const navs=[...document.querySelectorAll('.nav')];
const $=id=>document.getElementById(id);
const historyKey='intelx_history_v2';
const configKey='intelx_kyc_config';

function go(id){pages.forEach(p=>p.classList.toggle('show',p.id===id));navs.forEach(n=>n.classList.toggle('active',n.dataset.page===id)); if(id==='historico') renderHistory(); updateStats();}
navs.forEach(n=>n.onclick=()=>go(n.dataset.page));

function onlyDigits(v){return (v||'').replace(/\D/g,'')}
function validCPF(cpf){cpf=onlyDigits(cpf); if(cpf.length!==11||/^(\d)\1+$/.test(cpf)) return false; let s=0; for(let i=0;i<9;i++)s+=parseInt(cpf[i])*(10-i); let d=11-(s%11); if(d>=10)d=0; if(d!==parseInt(cpf[9]))return false; s=0; for(let i=0;i<10;i++)s+=parseInt(cpf[i])*(11-i); d=11-(s%11); if(d>=10)d=0; return d===parseInt(cpf[10]);}
function validCNPJ(cnpj){cnpj=onlyDigits(cnpj); if(cnpj.length!==14||/^(\d)\1+$/.test(cnpj)) return false; let calc=(base)=>{let pos=base.length-7,total=0; for(let i=base.length;i>=1;i--){total+=parseInt(base[base.length-i])*pos--; if(pos<2)pos=9} let r=total%11; return r<2?0:11-r}; return calc(cnpj.slice(0,12))==cnpj[12] && calc(cnpj.slice(0,13))==cnpj[13];}
function validEmail(e){return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test((e||'').trim())}
function validPhone(p){const d=onlyDigits(p); return d.length>=10 && d.length<=13}
function formatDate(){return new Date().toLocaleString('pt-BR')}
function getHist(){return JSON.parse(localStorage.getItem(historyKey)||'[]')}
function setHist(h){localStorage.setItem(historyKey,JSON.stringify(h)); updateStats()}
function saveRecord(r){const h=getHist(); h.unshift({...r,id:crypto.randomUUID?.()||Date.now(),createdAt:formatDate()}); setHist(h.slice(0,50));}

async function investigar(){
  const tipo=$('tipo').value; const valor=$('valor').value.trim(); const consent=$('consent').checked;
  if(!valor) return alert('Digite um valor para investigar.');
  if(!consent) return alert('Confirme que há autorização/base legal.');
  $('result').classList.add('hidden'); $('progress').classList.remove('hidden'); $('steps').innerHTML=''; $('bar').style.width='0%';
  const steps=['Validando entrada','Consultando fonte permitida','Organizando evidências','Calculando confiança','Gerando relatório'];
  for(let i=0;i<steps.length;i++){await wait(350); $('steps').innerHTML+=`<li>${steps[i]}</li>`; $('bar').style.width=((i+1)/steps.length*100)+'%';}
  try{
    let data;
    if(tipo==='cnpj') data=await searchCNPJ(valor);
    if(tipo==='dominio') data=await searchDomain(valor);
    if(tipo==='cpf') data=validateCPFReport(valor);
    if(tipo==='telefone') data=validatePhoneReport(valor);
    if(tipo==='email') data=validateEmailReport(valor);
    showResult(data); saveRecord(data);
  }catch(e){showResult({tipo,valor,status:'Erro',risco:'Atenção',fonte:'Sistema',resumo:e.message,campos:{erro:e.message}})}
}
function wait(ms){return new Promise(r=>setTimeout(r,ms))}

async function searchCNPJ(v){
  const cnpj=onlyDigits(v); if(!validCNPJ(cnpj)) throw new Error('CNPJ inválido pelos dígitos verificadores.');
  const res=await fetch(`https://brasilapi.com.br/api/cnpj/v1/${cnpj}`);
  if(!res.ok) throw new Error('CNPJ não encontrado ou API indisponível.');
  const j=await res.json();
  return {tipo:'CNPJ',valor:cnpj,status:'Consulta real concluída',risco:j.descricao_situacao_cadastral==='ATIVA'?'Baixo':'Atenção',fonte:'BrasilAPI',resumo:`Empresa ${j.razao_social||'não informada'} com situação ${j.descricao_situacao_cadastral||'não informada'}.`,campos:{'Razão social':j.razao_social,'Nome fantasia':j.nome_fantasia,'Situação':j.descricao_situacao_cadastral,'Abertura':j.data_inicio_atividade,'Município':j.municipio,'UF':j.uf,'CNAE':j.cnae_fiscal_descricao,'Porte':j.porte,'Natureza jurídica':j.natureza_juridica}}
}
async function searchDomain(v){
  let domain=v.replace(/^https?:\/\//,'').split('/')[0].trim().toLowerCase();
  if(!/^[a-z0-9.-]+\.[a-z]{2,}$/i.test(domain)) throw new Error('Domínio inválido. Use exemplo.com.br');
  const endpoint=domain.endsWith('.br')?`https://rdap.registro.br/domain/${domain}`:`https://rdap.org/domain/${domain}`;
  const res=await fetch(endpoint);
  if(!res.ok) throw new Error('Domínio não encontrado ou RDAP indisponível.');
  const j=await res.json();
  const events=(j.events||[]).reduce((a,e)=>{a[e.eventAction]=e.eventDate;return a},{});
  const ns=(j.nameservers||[]).map(n=>n.ldhName).filter(Boolean).join(', ');
  return {tipo:'Domínio',valor:domain,status:'Consulta real concluída',risco:(j.status||[]).some(s=>String(s).includes('inactive'))?'Atenção':'Baixo',fonte:domain.endsWith('.br')?'Registro.br RDAP':'RDAP.org',resumo:`Domínio ${domain} localizado via RDAP.`,campos:{'Handle':j.handle,'Status':(j.status||[]).join(', '),'Criado':events.registration||events['registration'],'Atualizado':events.last_changed||events['last changed']||events.last_changed,'Expira':events.expiration,'Nameservers':ns,'RDAP':endpoint}}
}
function validateCPFReport(v){const cpf=onlyDigits(v); const ok=validCPF(cpf); return {tipo:'CPF',valor:cpf,status:'Validação local concluída',risco:ok?'Baixo':'Alto',fonte:'Algoritmo de dígitos verificadores',resumo:ok?'CPF possui formato matematicamente válido. Não confirma identidade.':'CPF inválido pelos dígitos verificadores.',campos:{'CPF válido':ok?'Sim':'Não','Observação':'Este MVP não consulta nome, endereço, mãe/pai ou dados privados.'}}}
function validatePhoneReport(v){const d=onlyDigits(v); const ok=validPhone(d); return {tipo:'Telefone',valor:d,status:'Validação local concluída',risco:ok?'Baixo':'Atenção',fonte:'Validador local',resumo:ok?'Telefone tem tamanho compatível. Não confirma titularidade.':'Telefone fora do padrão esperado.',campos:{'Telefone possível':ok?'Sim':'Não','DDD':d.length>=10?d.slice(0,2):'Não identificado','Observação':'Consulta de titularidade exige API autorizada e consentimento.'}}}
function validateEmailReport(v){const e=v.trim().toLowerCase(); const ok=validEmail(e); const domain=e.split('@')[1]||''; return {tipo:'E-mail',valor:e,status:'Validação local concluída',risco:ok?'Baixo':'Atenção',fonte:'Validador local',resumo:ok?'E-mail tem formato válido. Para MX real use backend DNS.':'E-mail inválido.',campos:{'Formato válido':ok?'Sim':'Não','Domínio':domain,'Próximo passo':'Adicionar backend para DNS/MX e reputação.'}}}

function showResult(r){
  $('progress').classList.add('hidden'); const el=$('result'); el.classList.remove('hidden');
  const campos=Object.entries(r.campos||{}).map(([k,v])=>`<div class="kv"><small>${k}</small><b>${v||'Não informado'}</b></div>`).join('');
  el.innerHTML=`<h2>Resultado</h2><p><span class="badge">${r.status}</span></p><h3>${r.tipo}: ${r.valor}</h3><p>${r.resumo}</p><div class="result-grid">${campos}</div><p class="muted">Fonte: ${r.fonte} • Risco: ${r.risco}</p><button class="primary" onclick='printReport(${JSON.stringify(r).replaceAll("'","&apos;")})'>Gerar/Salvar PDF</button>`;
}
function printReport(r){
  const w=window.open('','_blank'); const campos=Object.entries(r.campos||{}).map(([k,v])=>`<tr><td>${k}</td><td>${v||''}</td></tr>`).join('');
  w.document.write(`<html><head><title>Relatório IntelX</title><style>body{font-family:Arial;padding:35px}h1{color:#0b5563}table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:10px}.muted{color:#555}</style></head><body><h1>Relatório IntelX Verify</h1><p class="muted">Gerado em ${formatDate()}</p><h2>${r.tipo}: ${r.valor}</h2><p><b>Status:</b> ${r.status}</p><p><b>Fonte:</b> ${r.fonte}</p><p><b>Risco:</b> ${r.risco}</p><p>${r.resumo}</p><table>${campos}</table><p class="muted">Relatório baseado em fontes públicas/validação local. Não substitui verificação jurídica.</p></body></html>`); w.document.close(); w.print();
}
function renderHistory(){const h=getHist(); $('history').innerHTML=h.length?h.map(x=>`<div class="history-item"><b>${x.tipo}: ${x.valor}</b><p>${x.resumo}</p><small>${x.createdAt} • ${x.fonte}</small></div>`).join(''):'<p class="muted">Sem histórico ainda.</p>'}
function clearHistory(){localStorage.removeItem(historyKey); renderHistory(); updateStats()}
function saveConfig(){localStorage.setItem(configKey,$('kycConfig').value); alert('Configuração salva localmente. Em produção, coloque chaves secretas apenas no backend.');}
function updateStats(){const h=getHist(); $('statCnpj').textContent=h.filter(x=>x.tipo==='CNPJ').length; $('statDomain').textContent=h.filter(x=>x.tipo==='Domínio').length; $('statValid').textContent=h.filter(x=>['CPF','Telefone','E-mail'].includes(x.tipo)).length; $('statTotal').textContent=h.length;}
$('kycConfig').value=localStorage.getItem(configKey)||''; updateStats();
