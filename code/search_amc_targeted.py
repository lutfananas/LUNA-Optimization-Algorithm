#!/usr/bin/env python3
"""Targeted AMC search for recent (2020-2025) papers directly relevant to LUNA.
Focus on: metaheuristic algorithms, optimization methods, evolutionary computation,
swarm intelligence, adaptive operators, parameter control, numerical optimization.
"""
import os, sys, json, time
import requests
import pandas as pd

OUT = '/home/z/my-project/download'
HEADERS = {'User-Agent': 'LUNA-Research/1.0 (mailto:lutfananas@example.com)'}

AMC_ISSN = '0096-3003'

# More targeted queries for actual metaheuristic/optimization papers
QUERIES = [
    'metaheuristic algorithm numerical optimization',
    'swarm optimization algorithm',
    'evolutionary algorithm continuous optimization',
    'adaptive operator selection',
    'parameter control evolutionary',
    'improved particle swarm',
    'improved differential evolution',
    'hybrid optimization algorithm',
    'chaotic optimization algorithm',
    'opposition based learning optimization',
    'Levy flight optimization',
    'mutation strategy evolutionary',
    'global optimization method',
    'stochastic global optimization',
    'population based optimization',
    'nature inspired algorithm',
    'whale optimization improved',
    'grey wolf improved',
    'gravitational search improved',
    'firefly algorithm improved',
    'bat algorithm',
    'harmony search improved',
    'ant colony continuous',
    'teaching learning optimization',
    'flower pollination',
    'salp swarm',
    'sine cosine algorithm',
    'sparrow search',
    'seagull optimization',
    'moth flame optimization',
    'multiverse optimizer',
    'dragonfly algorithm',
    'grasshopper optimization',
    'wildebeest optimization',
    'tunicate swarm',
    'honey badger algorithm',
    'chimp optimization',
    'aquila optimizer',
    'dwarf mongoose',
    'zebra optimization',
]


def search_crossref(query, rows=30, filter_issn=None, from_year=2020):
    params = {
        'query': query, 'rows': rows,
        'select': 'DOI,title,container-title,issued,author,ISSN,is-referenced-by-count,abstract'
    }
    filters = []
    if filter_issn:
        filters.append(f'issn:{filter_issn}')
    if from_year:
        filters.append(f'from-pub-date:{from_year}-01-01')
    if filters:
        params['filter'] = ','.join(filters)
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
        items = search_crossref(q, rows=20, filter_issn=AMC_ISSN, from_year=2020)
        for it in items:
            doi = it.get('DOI', '')
            if not doi:
                continue
            title = it.get('title', [''])[0] if it.get('title') else ''
            container = it.get('container-title', [''])[0] if it.get('container-title') else ''
            year = it.get('issued', {}).get('date-parts', [[None]])[0][0]
            authors = it.get('author', [])[:4]
            author_str = '; '.join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors])
            cited = it.get('is-referenced-by-count', 0)
            issns = it.get('ISSN', [])
            abstract = it.get('abstract', '')[:400] if it.get('abstract') else ''

            all_records[doi] = {
                'doi': doi,
                'title': title,
                'container': container,
                'year': year,
                'authors': author_str,
                'cited_by': cited,
                'issns': issns,
                'abstract': abstract,
                'query_matched': q,
            }
        time.sleep(0.3)

    print(f"\nTotal unique records: {len(all_records)}")

    # Strict filter: only AMC, only true optimization/metaheuristic papers
    strict_keywords = [
        'metaheuristic', 'optimization algorithm', 'swarm',
        'evolutionary algorithm', 'particle swarm', 'differential evolution',
        'genetic algorithm', 'whale optimization', 'grey wolf',
        'ant colony', 'firefly', 'harmony search', 'gravitational search',
        'salp swarm', 'bat algorithm', 'flower pollination',
        'sine cosine', 'sparrow search', 'seagull', 'moth flame',
        'multiverse', 'dragonfly', 'grasshopper', 'wildebeest',
        'tunicate', 'honey badger', 'chimp', 'aquila', 'dwarf mongoose',
        'zebra', 'slime mould', 'harris hawks', 'jellyfish',
        'teaching learning', 'affine scaling', 'global optimization',
        'numerical optimization', 'constrained optimization',
        'opposition-based', 'opposition based', 'chaotic',
        'Levy flight', 'Lévy flight', 'adaptive operator',
        'benchmark function', 'CEC 2022', 'CEC 2017',
        'mutation', 'crossover', 'search strategy',
        'exploration exploitation',
    ]

    filtered = []
    seen_titles = set()
    for r in all_records.values():
        title_lower = r['title'].lower()
        if not any(k.lower() in title_lower for k in strict_keywords):
            continue
        # Dedupe by normalized title
        norm_title = ''.join(c.lower() for c in r['title'] if c.isalnum())
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        filtered.append(r)

    # Sort by citations desc, year desc
    filtered.sort(key=lambda x: (-(x['cited_by'] or 0), -(x['year'] or 0)))

    print(f"Strictly filtered: {len(filtered)}")

    # Print top 25
    print(f"\nTop 25 recent AMC metaheuristic papers (2020-2025):")
    for i, r in enumerate(filtered[:25]):
        print(f"\n{i+1}. [{r['cited_by']:4d} cit, {r['year']}] {r['title']}")
        print(f"   Authors: {r['authors']}")
        print(f"   DOI: {r['doi']}")
        if r.get('abstract'):
            print(f"   Abstract: {r['abstract'][:250]}")

    # Save
    df = pd.DataFrame(filtered)
    df.to_csv(f'{OUT}/AMC_targeted_citations.csv', index=False)
    with open(f'{OUT}/AMC_targeted_citations.json', 'w') as f:
        json.dump({
            'issn': AMC_ISSN,
            'total_unique': len(all_records),
            'strictly_filtered': len(filtered),
            'all_records': filtered,
        }, f, indent=2)
    print(f"\nSaved: AMC_targeted_citations.csv")


if __name__ == "__main__":
    main()
