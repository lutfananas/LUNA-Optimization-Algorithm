#!/usr/bin/env python3
"""Search for very recent AMC papers (2024-2026) on optimization."""
import requests, json, time

HEADERS = {'User-Agent': 'LUNA-Research/1.0 (mailto:lutfananas@example.com)'}
AMC_ISSN = '0096-3003'

QUERIES = [
    'evolutionary algorithm 2025',
    'metaheuristic 2025',
    'swarm optimization 2025',
    'adaptive optimization 2025',
    'population-based optimization 2024',
    'stochastic optimization 2024',
    'global search 2024 2025',
    'numerical optimization 2025',
    'convergence analysis optimization 2025',
    'constrained optimization 2025',
    'parameter adaptation 2025',
    'mutation operator 2025',
    'benchmark 2025 optimization',
    'operator selection 2024 2025',
    'reinforcement learning optimization 2025',
    'adaptive step size 2025',
    'multi-strategy 2025 optimization',
    'hybrid metaheuristic 2025',
    'chaos optimization 2024 2025',
    'differential evolution 2025',
    'neural network optimization 2024 2025',
    'gradient descent 2025',
    'stochastic gradient 2025',
    'iterative method optimization 2025',
    'nonlinear optimization 2025',
]


def search(query, from_year=2024):
    params = {
        'query': query, 'rows': 30,
        'select': 'DOI,title,container-title,issued,author,ISSN,is-referenced-by-count,abstract',
        'filter': f'issn:{AMC_ISSN},from-pub-date:{from_year}-01-01'
    }
    try:
        r = requests.get('https://api.crossref.org/works',
                        params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json().get('message', {}).get('items', [])
    except Exception as e:
        return []


def main():
    all_records = {}
    for i, q in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] {q}", flush=True)
        items = search(q, from_year=2024)
        for it in items:
            doi = it.get('DOI', '')
            if not doi: continue
            title = it.get('title', [''])[0] if it.get('title') else ''
            year = it.get('issued', {}).get('date-parts', [[None]])[0][0]
            authors = it.get('author', [])[:3]
            author_str = '; '.join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors])
            cited = it.get('is-referenced-by-count', 0)
            abstract = it.get('abstract', '')[:300] if it.get('abstract') else ''
            all_records[doi] = {
                'doi': doi, 'title': title, 'year': year,
                'authors': author_str, 'cited_by': cited, 'abstract': abstract
            }
        time.sleep(0.3)

    # Filter for optimization-relevant titles
    keywords = [
        'optimization', 'metaheuristic', 'swarm', 'evolutionary',
        'algorithm', 'search', 'mutation', 'crossover', 'adaptive',
        'particle swarm', 'genetic', 'differential evolution',
        'convergence', 'Markov', 'stochastic', 'random',
        'gradient', 'Newton', 'iterative',
    ]
    filtered = []
    seen = set()
    for r in all_records.values():
        title_lower = r['title'].lower()
        if not any(k.lower() in title_lower for k in keywords):
            continue
        norm = ''.join(c.lower() for c in r['title'] if c.isalnum())
        if norm in seen: continue
        seen.add(norm)
        filtered.append(r)

    # Sort: newest first, then most cited
    filtered.sort(key=lambda x: (-(x['year'] or 0), -(x['cited_by'] or 0)))

    print(f"\nTotal unique: {len(all_records)}")
    print(f"Optimization-filtered: {len(filtered)}")
    print(f"\nTop 20 most relevant AMC papers (2024-2026):")
    for i, r in enumerate(filtered[:20]):
        print(f"\n{i+1}. [{r['cited_by']} cit, {r['year']}] {r['title']}")
        print(f"   Authors: {r['authors']}")
        print(f"   DOI: 10.1016/j.amc.{r['doi'].split('.')[-1]}")
        if r.get('abstract'):
            print(f"   Abstract: {r['abstract'][:250]}")

    with open('/home/z/my-project/download/AMC_2024_2026.json', 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"\nSaved: AMC_2024_2026.json ({len(filtered)} records)")


if __name__ == "__main__":
    main()
