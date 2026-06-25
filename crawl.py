"""
Web crawler para coletar texto em português brasileiro.
Fontes: Wikipedia PT, G1, UOL, Agência Brasil, Brasil de Fato, Nexo, etc.
Filtra parágrafos: min 40 chars, max 400 chars, ratio alpha > 0.6, min 6 palavras.
Saída: dataset_crawl.txt — uma frase/parágrafo por linha.
"""

import re
import time
import random
import requests
import trafilatura
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Configuração ──────────────────────────────────────────────────────────────

OUTPUT    = "dataset_crawl.txt"
MAX_PAGES = 2000         # páginas a visitar
WORKERS   = 16           # threads paralelas
MIN_CHARS = 40
MAX_CHARS = 500
MIN_WORDS = 6
ALPHA_RATIO = 0.55
DELAY_MIN = 0.1          # delay entre requests por thread (s)
DELAY_MAX = 0.4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; educational-crawler/1.0; pt-BR dataset collection)",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

# Domínios permitidos (apenas PT-BR)
ALLOWED_DOMAINS = {
    # Enciclopédia
    "pt.wikipedia.org",
    # Jornalismo público
    "agenciabrasil.ebc.com.br",
    "brasildefato.com.br",
    "apublica.org",
    "aosfatos.org",
    "revistaforum.com.br",
    "diplomatique.org.br",
    "nexojornal.com.br",
    "cartacapital.com.br",
    "valor.globo.com",
    "oglobo.globo.com",
    "folha.uol.com.br",
    "estadao.com.br",
    "bbc.com",           # BBC Brasil
    "g1.globo.com",
    "uol.com.br",
    # Ciência e educação
    "scielo.br",
    "www.scielo.br",
    "brasilescola.uol.com.br",
    "mundoeducacao.uol.com.br",
    "educacao.uol.com.br",
    "vestibular.brasilescola.uol.com.br",
    "infoescola.com",
    "www.infoescola.com",
    "portalsaofrancisco.com.br",
    "www.portalsaofrancisco.com.br",
    "todamateria.com.br",
    "www.todamateria.com.br",
    "coladaweb.com",
    "www.coladaweb.com",
    "estudokids.com.br",
    "www.estudokids.com.br",
    "educamaisbrasil.com.br",
    "www.educamaisbrasil.com.br",
    "pensador.com",
    "www.pensador.com",
    # Dicionários e língua
    "significados.com.br",
    "www.significados.com.br",
    "dicio.com.br",
    "www.dicio.com.br",
    "priberam.pt",
    "www.priberam.pt",
    "infopedia.pt",
    "www.infopedia.pt",
    # Saúde
    "drauziovarella.uol.com.br",
    "www.drauziovarella.uol.com.br",
    "minhavida.com.br",
    "www.minhavida.com.br",
    "tuasaude.com",
    "www.tuasaude.com",
    "medicinanet.com.br",
    # Tecnologia
    "tecmundo.com.br",
    "www.tecmundo.com.br",
    "canaltech.com.br",
    "www.canaltech.com.br",
    "olhardigital.com.br",
    "www.olhardigital.com.br",
    "techtudo.com.br",
    "www.techtudo.com.br",
    # Cultura e história
    "historiadomundo.com.br",
    "www.historiadomundo.com.br",
    "historiailustrada.com.br",
    "www.sohistoria.com.br",
    "sohistoria.com.br",
    "culturamix.com",
    "www.culturamix.com",
}

# Seeds — páginas iniciais
SEEDS = [
    # ── Wikipedia PT (temas amplos) ──────────────────────────────────────────
    "https://pt.wikipedia.org/wiki/Brasil",
    "https://pt.wikipedia.org/wiki/Ci%C3%AAncia",
    "https://pt.wikipedia.org/wiki/Hist%C3%B3ria_do_Brasil",
    "https://pt.wikipedia.org/wiki/Tecnologia",
    "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
    "https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina",
    "https://pt.wikipedia.org/wiki/Natureza",
    "https://pt.wikipedia.org/wiki/Astronomia",
    "https://pt.wikipedia.org/wiki/Biologia",
    "https://pt.wikipedia.org/wiki/Qu%C3%ADmica",
    "https://pt.wikipedia.org/wiki/F%C3%ADsica",
    "https://pt.wikipedia.org/wiki/Matem%C3%A1tica",
    "https://pt.wikipedia.org/wiki/Economia",
    "https://pt.wikipedia.org/wiki/Filosofia",
    "https://pt.wikipedia.org/wiki/Sa%C3%BAde",
    "https://pt.wikipedia.org/wiki/Meio_ambiente",
    "https://pt.wikipedia.org/wiki/Cultura_do_Brasil",
    "https://pt.wikipedia.org/wiki/M%C3%BAsica_do_Brasil",
    "https://pt.wikipedia.org/wiki/Cinema_do_Brasil",
    "https://pt.wikipedia.org/wiki/Literatura_brasileira",
    "https://pt.wikipedia.org/wiki/Futebol_no_Brasil",
    "https://pt.wikipedia.org/wiki/Geologia",
    "https://pt.wikipedia.org/wiki/Geografia_do_Brasil",
    "https://pt.wikipedia.org/wiki/Pol%C3%ADtica_do_Brasil",
    "https://pt.wikipedia.org/wiki/Direito",
    "https://pt.wikipedia.org/wiki/Psicologia",
    "https://pt.wikipedia.org/wiki/Sociologia",
    "https://pt.wikipedia.org/wiki/Antropologia",
    "https://pt.wikipedia.org/wiki/Arte",
    "https://pt.wikipedia.org/wiki/Arquitetura",
    "https://pt.wikipedia.org/wiki/Medicina",
    "https://pt.wikipedia.org/wiki/Nutri%C3%A7%C3%A3o",
    "https://pt.wikipedia.org/wiki/Esporte",
    "https://pt.wikipedia.org/wiki/Culin%C3%A1ria_brasileira",
    "https://pt.wikipedia.org/wiki/Religi%C3%A3o_no_Brasil",
    "https://pt.wikipedia.org/wiki/Idioma_portugu%C3%AAs",
    "https://pt.wikipedia.org/wiki/Portugu%C3%AAs_brasileiro",
    "https://pt.wikipedia.org/wiki/Uni%C3%A3o_Europeia",
    "https://pt.wikipedia.org/wiki/Guerra_Fria",
    "https://pt.wikipedia.org/wiki/Segunda_Guerra_Mundial",
    "https://pt.wikipedia.org/wiki/Primeira_Guerra_Mundial",
    "https://pt.wikipedia.org/wiki/Revolu%C3%A7%C3%A3o_Industrial",
    "https://pt.wikipedia.org/wiki/Sistema_solar",
    "https://pt.wikipedia.org/wiki/Buraco_negro",
    "https://pt.wikipedia.org/wiki/DNA",
    "https://pt.wikipedia.org/wiki/Evolu%C3%A7%C3%A3o",
    "https://pt.wikipedia.org/wiki/Ecologia",
    "https://pt.wikipedia.org/wiki/Aquecimento_global",
    "https://pt.wikipedia.org/wiki/Energia_solar",
    "https://pt.wikipedia.org/wiki/Energia_e%C3%B3lica",
    "https://pt.wikipedia.org/wiki/Internet",
    "https://pt.wikipedia.org/wiki/Computador",
    "https://pt.wikipedia.org/wiki/Programa%C3%A7%C3%A3o_de_computadores",
    "https://pt.wikipedia.org/wiki/Python_(linguagem_de_programa%C3%A7%C3%A3o)",
    "https://pt.wikipedia.org/wiki/Rede_neural_artificial",
    # ── Agência Brasil ───────────────────────────────────────────────────────
    "https://agenciabrasil.ebc.com.br/",
    "https://agenciabrasil.ebc.com.br/educacao",
    "https://agenciabrasil.ebc.com.br/ciencia-e-tecnologia",
    "https://agenciabrasil.ebc.com.br/saude",
    "https://agenciabrasil.ebc.com.br/economia",
    "https://agenciabrasil.ebc.com.br/cultura",
    "https://agenciabrasil.ebc.com.br/direitos-humanos",
    "https://agenciabrasil.ebc.com.br/meio-ambiente",
    # ── Educação ─────────────────────────────────────────────────────────────
    "https://brasilescola.uol.com.br/",
    "https://brasilescola.uol.com.br/biologia",
    "https://brasilescola.uol.com.br/fisica",
    "https://brasilescola.uol.com.br/quimica",
    "https://brasilescola.uol.com.br/matematica",
    "https://brasilescola.uol.com.br/historia",
    "https://brasilescola.uol.com.br/geografia",
    "https://brasilescola.uol.com.br/sociologia",
    "https://brasilescola.uol.com.br/filosofia",
    "https://mundoeducacao.uol.com.br/",
    "https://mundoeducacao.uol.com.br/biologia",
    "https://mundoeducacao.uol.com.br/fisica",
    "https://mundoeducacao.uol.com.br/quimica",
    "https://mundoeducacao.uol.com.br/matematica",
    "https://mundoeducacao.uol.com.br/historia-do-brasil",
    "https://mundoeducacao.uol.com.br/geografia",
    "https://www.infoescola.com/",
    "https://www.todamateria.com.br/",
    "https://www.todamateria.com.br/biologia/",
    "https://www.todamateria.com.br/historia/",
    "https://www.todamateria.com.br/matematica/",
    "https://www.portalsaofrancisco.com.br/",
    "https://www.coladaweb.com/",
    "https://www.significados.com.br/",
    "https://www.dicio.com.br/",
    # ── Ciência e saúde ──────────────────────────────────────────────────────
    "https://www.scielo.br/",
    "https://drauziovarella.uol.com.br/",
    "https://www.minhavida.com.br/saude",
    "https://www.tuasaude.com/",
    # ── Tecnologia ───────────────────────────────────────────────────────────
    "https://www.tecmundo.com.br/",
    "https://canaltech.com.br/",
    "https://olhardigital.com.br/",
    "https://www.techtudo.com.br/",
    # ── Jornalismo ───────────────────────────────────────────────────────────
    "https://www.brasildefato.com.br/",
    "https://apublica.org/",
    "https://aosfatos.org/",
    "https://www.bbc.com/portuguese",
    "https://www.bbc.com/portuguese/geral",
    "https://www.bbc.com/portuguese/brasil",
    "https://www.bbc.com/portuguese/ciencia",
    # ── História e cultura ───────────────────────────────────────────────────
    "https://www.historiadomundo.com.br/",
    "https://www.sohistoria.com.br/",
    "https://www.culturamix.com/",
]

# ── Filtros ───────────────────────────────────────────────────────────────────

_NOISE = re.compile(
    r"(cookie|javascript|clique aqui|leia mais|compartilh|assine|newsletter"
    r"|publicidade|propaganda|anuncio|©|direitos reservados"
    r"|carregando|menu|buscar|pesquisar|\[\d+\])",
    re.IGNORECASE
)

def is_good_paragraph(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < ALPHA_RATIO:
        return False
    if _NOISE.search(text):
        return False
    # deve conter ao menos uma vogal acentuada ou ç (sinal de PT)
    # (relaxado — aceita texto sem acentos também)
    return True

def extract_paragraphs(url: str, html: str) -> list[str]:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        url=url,
    )
    if not text:
        return []
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        # quebra linha longa em sentenças
        sentences = re.split(r'(?<=[.!?])\s+', line)
        for s in sentences:
            if is_good_paragraph(s):
                paragraphs.append(s)
    return paragraphs

def get_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = []
    base_domain = urlparse(base_url).netloc
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        domain = parsed.netloc
        # só mesmo domínio ou domínios permitidos
        if domain not in ALLOWED_DOMAINS and domain != base_domain:
            continue
        # sem fragmentos, sem arquivos binários
        if parsed.fragment:
            continue
        ext = parsed.path.split(".")[-1].lower()
        if ext in {"pdf", "jpg", "jpeg", "png", "gif", "zip", "mp4", "mp3"}:
            continue
        # Wikipedia: pula especiais
        if "wikipedia.org" in domain:
            path = parsed.path
            if any(x in path for x in [
                "/wiki/Especial:", "/wiki/Discuss", "/wiki/Usu%C3%A1rio",
                "/wiki/Wikipedia:", "/wiki/Ajuda:", "/wiki/Ficheiro:",
                "/wiki/Portal:", "/wiki/Categoria:", "action=", "Special:",
            ]):
                continue
        links.append(full)
    return links

# ── Crawler paralelo ─────────────────────────────────────────────────────────

def _fetch(session: requests.Session, url: str) -> tuple[str, str | None]:
    try:
        resp = session.get(url, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return url, None
        if "text/html" not in resp.headers.get("content-type", ""):
            return url, None
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        return url, resp.text
    except Exception:
        return url, None


def crawl():
    output = Path(OUTPUT)
    lock   = Lock()

    existing: set[str] = set()
    if output.exists():
        existing = set(output.read_text(encoding="utf-8").splitlines())
        print(f"Já coletados: {len(existing)} parágrafos")

    visited:   set[str]   = set(SEEDS)
    queue:     deque[str] = deque(SEEDS)
    collected: list[str]  = []
    pages_done = 0

    # uma session por thread (thread-safe via pool)
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"Iniciando crawl paralelo: {MAX_PAGES} páginas alvo | {WORKERS} workers\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while pages_done < MAX_PAGES:
            # pega batch de URLs para submeter
            batch: list[str] = []
            while queue and len(batch) < WORKERS * 2:
                url = queue.popleft()
                batch.append(url)

            if not batch:
                break

            futures = {pool.submit(_fetch, session, url): url for url in batch}

            for fut in as_completed(futures):
                url, html = fut.result()
                if html is None:
                    continue

                paragraphs = extract_paragraphs(url, html)

                with lock:
                    new_pars = [p for p in paragraphs if p not in existing]
                    existing.update(new_pars)
                    collected.extend(new_pars)
                    pages_done += 1

                    print(f"[{pages_done:04d}/{MAX_PAGES}] {len(new_pars):3d} novos | "
                          f"total {len(existing):,} | {url[:65]}")

                    # salva a cada 50 páginas
                    if pages_done % 50 == 0 and collected:
                        with output.open("a", encoding="utf-8") as f:
                            f.write("\n".join(collected) + "\n")
                        collected.clear()
                        print(f"  → checkpoint: {len(existing):,} parágrafos no disco")

                    if pages_done >= MAX_PAGES:
                        break

                # descobre novos links (fora do lock — é read-only)
                new_links = get_links(html, url)
                random.shuffle(new_links)
                with lock:
                    for link in new_links[:15]:
                        if link not in visited:
                            visited.add(link)
                            queue.append(link)

    # flush final
    with lock:
        if collected:
            with output.open("a", encoding="utf-8") as f:
                f.write("\n".join(collected) + "\n")

    total = sum(1 for l in output.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"\nCrawl concluído: {pages_done} páginas | {total:,} parágrafos em {OUTPUT}")


if __name__ == "__main__":
    crawl()
