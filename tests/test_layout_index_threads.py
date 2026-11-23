"""The layout index is queried from the image thread pool; connections must not be shared."""

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

from corpus_pipeline.production import portable_refs


def _index(tmp_path, rows=2000):
    path = tmp_path / "layout.sqlite"
    db = sqlite3.connect(path)
    db.execute("create table entries(id integer primary key, offset integer, payload text)")
    db.executemany("insert into entries values(?,?,?)", ((i, i * 512, "x" * 64) for i in range(rows)))
    db.commit()
    db.close()
    return str(path)


def test_each_thread_gets_its_own_connection_and_reuses_it(tmp_path):
    index = _index(tmp_path)
    barrier = threading.Barrier(4)

    def connection(_):
        barrier.wait()
        first = portable_refs._layout_db(index, os.getpid())
        return id(first), first is portable_refs._layout_db(index, os.getpid())

    with ThreadPoolExecutor(max_workers=4) as pool:
        seen = list(pool.map(connection, range(4)))
    assert all(reused for _, reused in seen)
    assert len({ident for ident, _ in seen}) == 4


def test_concurrent_lookups_return_the_right_rows(tmp_path):
    index = _index(tmp_path)

    def lookup(i):
        row = (
            portable_refs._layout_db(index, os.getpid())
            .execute("select id from entries where offset=?", (i * 512,))
            .fetchone()
        )
        return row[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lookup, list(range(2000)) * 10)) == list(range(2000)) * 10
