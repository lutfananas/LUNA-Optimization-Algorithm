#!/usr/bin/env python3
"""Search Applied Mathematics and Computation (AMC) for recent papers (2023-2026)
on convergence analysis, benchmark methodology, and adaptive metaheuristic design.
"""
import os, sys, json, time
import requests
import pandas as pd

OUT = '/home/z/my-project/download'
HEADERS = {'User-Agent': 'LUNA-Research/1.0 (mailto:lutfananas@example.com)'}

AMC_ISSN = '0096-3003'

QUERIES = [
    'convergence analysis metaheuristic 2024',
    'convergence proof evolutionary algorithm 2023',
    'benchmark methodology optimization 2024',
    'CEC 2022 benchmark evaluation',
    'adaptive parameter metaheuristic 2024',
    'adaptive operator selection 2023',
    'parameter adaptation evolutionary 2025',
    'Markov chain convergence optimization 2024',
    'stochastic convergence global optimization 2023',
    'no free lunch theorem optimization',
    'algorithm selection benchmark 2024',
    'adaptive step size optimization 2025',
    'chaos optimization algorithm 2024',
    'hybrid metaheuristic 2024 2025',
    'reinforcement learning metaheuristic 2024',
    'surrogate-assisted optimization 2025',
    'multi-swarm optimization 2024',
    'opposition-based learning 2024',
    'Levy flight optimization 2024',
    'quantum-behaved particle swarm 2024',
]


def search_crossref(query, rows=50, filter_issn=None, from_year=2023):
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
        print(f"  ERROR: {e}")
        return []


def main():
    print(f"Searching AMC (ISSN {AMC_ISSN}) for 2023-2026 papers...")
    all_records = {}

    for i, q in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] Query: {q}")
        items = search_crossref(q, rows=50, filter_issn=AMC_ISSN, from_year=2023)
        print(f"  Found: {len(items)} items")
        for it in items:
            doi = it.get('DOI', '')
            if not doi:
                continue
            title = it.get('title', [''])[0] if it.get('title') else ''
            container = it.get('container-title', [''])[0] if it.get('container-title') else ''
            year = it.get('issued', {}).get('date-parts', [[None]])[0][0]
            authors = it.get('author', [])[:3]
            author_str = '; '.join([f"{a.get('family', '')}, {a.get('given', '')}" for a in authors])
            cited = it.get('is-referenced-by-count', 0)
            issns = it.get('ISSN', [])
            abstract = it.get('abstract', '')[:300] if it.get('abstract') else ''

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
        time.sleep(0.4)

    print(f"\n{'='*60}")
    print(f"Total unique records: {len(all_records)}")

    amc_records = {d: r for d, r in all_records.items()
                   if AMC_ISSN in r['issns'] or 'Applied Mathematics and Computation' in r['container']}
    print(f"AMC-verified records: {len(amc_records)}")

    # Sort by year desc, then citations desc
    sorted_records = sorted(amc_records.values(),
                           key=lambda x: (-(x['year'] or 0), -(x['cited_by'] or 0)))

    print(f"\nTop 30 recent AMC papers (2023-2026):")
    for i, r in enumerate(sorted_records[:30]):
        print(f"  {i+1}. [{r['cited_by']:4d} cit, {r['year']}] {r['title'][:90]}")
        print(f"     DOI: {r['doi']}")

    df = pd.DataFrame(sorted_records)
    df.to_csv(f'{OUT}/AMC_recent_citations.csv', index=False)

    with open(f'{OUT}/AMC_recent_citations.json', 'w') as f:
        json.dump({
            'issn': AMC_ISSN,
            'total_queries': len(QUERIES),
            'amc_verified_records': len(amc_records),
            'all_records': sorted_records,
        }, f, indent=2)

    print(f"\nSaved: {OUT}/AMC_recent_citations.csv")
    print(f"Saved: {OUT}/AMC_recent_citations.json")


if __name__ == "__main__":
    main()
