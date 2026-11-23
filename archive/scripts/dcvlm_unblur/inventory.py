"""Pin the three upstream pools and enumerate only DataComp TAR files."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from prepare import RUN, save

PINS = {
    'dcvlm_pool_small': 'fb66c73cc925874aaf520978d31d8ed1aeab307e',
    'dcvlm_pool_medium': 'a6dc0aed421b1b9af178e8728c0f31681107b251',
    'dcvlm_pool_large': 'c76328c4afd15d3a6333b68a48dabf4f8f0819cc',
}


def listing(task):
    repo, folder = task
    cache = RUN / 'inventory' / (repo + '_' + folder.replace('/', '_') + '.json')
    if cache.exists():
        return json.loads(cache.read_text())
    url = f'https://huggingface.co/api/datasets/mlfoundations/{repo}/tree/{PINS[repo]}/{folder}?recursive=false&limit=1000'
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        while url:
            for attempt in range(8):
                try:
                    response = session.get(url, timeout=(10, 60))
                    response.raise_for_status()
                    rows.extend(response.json())
                    url = response.links.get('next', {}).get('url')
                    break
                except requests.RequestException:
                    if attempt == 7:
                        raise
                    time.sleep(2 + attempt)
    save(cache, rows)
    print(repo, folder, len(rows), flush=True)
    return rows


def main():
    (RUN / 'inventory').mkdir(exist_ok=True)
    tasks = [(repo, 'datacomp_1b') for repo in PINS]
    found = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        while tasks:
            next_tasks = []
            for (repo, folder), rows in zip(tasks, pool.map(listing, tasks)):
                for row in rows:
                    if row['type'] == 'directory':
                        next_tasks.append((repo, row['path']))
                    elif row['path'].endswith('.tar'):
                        found.append(dict(repo=repo, revision=PINS[repo], path=row['path'],
                                          size=row['size'], sha256=row['lfs']['oid']))
            tasks = next_tasks
    by_name = {}
    for row in found:
        name = row['path'].rsplit('/', 1)[-1]
        assert name not in by_name, name
        by_name[name] = row
    save(RUN / 'upstream_inventory.json', by_name)
    print('TOTAL', len(found), sum(r['size'] for r in found), flush=True)


if __name__ == '__main__':
    main()
