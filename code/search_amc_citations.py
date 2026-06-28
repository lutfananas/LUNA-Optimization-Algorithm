#!/usr/bin/env python3
"""
Search Applied Mathematics and Computation (AMC) journal for papers
relevant to LUNA: metaheuristic optimization, swarm intelligence,
CEC benchmarks, theoretical analysis, parameter tuning, etc.

Uses Crossref API. Filters to AMC ISSN: 0096-3003
"""
import os, sys, json, time
import requests
import pandas as pd

OUT = '/home/z/my-project/download'
HEADERS = {'User-Agent': 'LUNA-Research/1.0 (mailto:lutfananas@example.com)'}

AMC_ISSN = '0096-3003'

# Search queries targeting AMC papers relevant to LUNA
QUERIES = [
    'metaheuristic optimization algorithm',
    'swarm intelligence',
    'nature-inspired algorithm',
    'particle swarm optimization',
    'differential evolution',
    'whale optimization',
    'grey wolf optimizer',
    'genetic algorithm optimization',
    'benchmark CEC optimization',
    'parameter adaptation metaheuristic',
    'opposition-based learning',
    'chaotic map metaheuristic',
    'Levy flight optimization',
    'global optimization convergence',
    'convergence proof metaheuristic',
    'Markov chain optimization',
    'hitting time stochastic search',
    'ergodicity stochastic optimization',
    'multimodal optimization',
    'hybrid metaheuristic',
    'astronomy inspired optimization',
    'lunar gravitational',
    'orbital mechanics optimization',
    'vis-viva equation',
    'physics-inspired metaheuristic',
    'tidal force algorithm',
    'gravitational search',
]


def search_crossref(query, rows=50, filter_issn=None):
    """Search Crossref, optionally filter by ISSN."""
    params = {'query': query, 'rows': rows, 'select': 'DOI,title,container-title,issued,author,ISSN,is-referenced-by-count'}
    if filter_issn:
        params['filter'] = f'issn:{filter_issn}'
    try:
        r = requests.get('https://api.crossref.org/works',
                        params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json().get('message', {}).get('items', [])
    except Exception as e:
        print(f"  ERROR for query '{query}': {e}")
        return []


def main():
    print(f"Searching AMC (ISSN {AMC_ISSN}) for LUNA-relevant papers...")
    print(f"Queries: {len(QUERIES)}")

    all_records = {}  # by DOI

    for i, q in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] Query: {q}")
        items = search_crossref(q, rows=50, filter_issn=AMC_ISSN)
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

            all_records[doi] = {
                'doi': doi,
                'title': title,
                'container': container,
                'year': year,
                'authors': author_str,
                'cited_by': cited,
                'issns': issns,
                'query_matched': q,
            }
        time.sleep(0.5)  # be polite

    print(f"\n{'='*60}")
    print(f"Total unique records: {len(all_records)}")
    print(f"{'='*60}")

    # Filter: keep only AMC papers (sanity check)
    amc_records = {d: r for d, r in all_records.items()
                   if AMC_ISSN in r['issns'] or 'Applied Mathematics and Computation' in r['container']}
    print(f"AMC-verified records: {len(amc_records)}")

    # Sort by citations
    sorted_records = sorted(amc_records.values(),
                           key=lambda x: (-x['cited_by'] if x['cited_by'] else 0, x['year'] or 0))

    # Top 50 most-cited
    print("\nTop 30 most-cited AMC papers (relevant to LUNA):")
    for i, r in enumerate(sorted_records[:30]):
        print(f"  {i+1}. [{r['cited_by']:5d} cit, {r['year']}] {r['title'][:90]}")
        print(f"     DOI: {r['doi']}")

    # Save
    df = pd.DataFrame(sorted_records)
    df.to_csv(f'{OUT}/AMC_citations.csv', index=False)

    # Also save as JSON for easy parsing
    with open(f'{OUT}/AMC_citations.json', 'w') as f:
        json.dump({
            'issn': AMC_ISSN,
            'total_queries': len(QUERIES),
            'total_unique_records': len(all_records),
            'amc_verified_records': len(amc_records),
            'top_50_most_cited': sorted_records[:50],
            'all_records': sorted_records,
        }, f, indent=2)

    print(f"\nSaved: {OUT}/AMC_citations.csv")
    print(f"Saved: {OUT}/AMC_citations.json")

    # Generate BibTeX entries for the top 30
    print("\nGenerating BibTeX for top 30...")
    bib_entries = []
    for i, r in enumerate(sorted_records[:30]):
        # Create BibTeX key: first authorlastname + year + firstword
        first_author = r['authors'].split(';')[0].strip() if r['authors'] else 'Unknown'
        lastname = first_author.split(',')[0].strip().replace(' ', '') if first_author != 'Unknown' else 'Unknown'
        first_word = r['title'].split()[0].strip('.,;:!?') if r['title'] else 'X'
        first_word = ''.join(c for c in first_word if c.isalnum())
        key = f"{lastname}{r['year'] or 'XXXX'}{first_word}"

        # Build BibTeX
        title = r['title'].replace('{', '').replace('}', '')
        bib = f"""@article{{{key},
  title   = {{{title}}},
  author  = {{{r['authors']}}},
  journal = {{Applied Mathematics and Computation}},
  year    = {{{r['year'] or 'XXXX'}}},
  doi     = {{{r['doi']}}},
}}
"""
        bib_entries.append(bib)

    with open(f'{OUT}/AMC_references.bib', 'w') as f:
        f.write('\n'.join(bib_entries))
    print(f"Saved: {OUT}/AMC_references.bib ({len(bib_entries)} entries)")


if __name__ == "__main__":
    main()
