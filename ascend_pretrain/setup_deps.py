"""Prepare the pinned CANN 9.0 / Core 0.17 dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-cann', '--with-nnal', dest='with_cann', action='store_true',
                        help='Download all three matching CANN packages, verifying their SHA256')
    parser.add_argument('--install-runtime', action='store_true', help='Install CANN into the project prefix')
    parser.add_argument('--install-python', action='store_true', help='Create the pinned project Python environment')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    project = package.parent
    lock = json.loads((package/'deps.core17.lock.json').read_text())
    deps = package/'.deps'; deps.mkdir(exist_ok=True)
    for name, entry in lock['sources'].items():
        path = deps/'core17'/name
        if not path.exists():
            path.mkdir(parents=True)
            run('git', '-C', path, 'init', '-q')
            run('git', '-C', path, 'remote', 'add', 'origin', entry['url'])
            run('git', '-C', path, 'fetch', '--depth', '1', 'origin', entry['commit'])
            run('git', '-C', path, 'checkout', '--detach', entry['commit'])
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != entry['commit']:
            raise RuntimeError(f'{name}: commit differs from lock; refusing to overwrite this checkout')
        run('git', '-C', path, 'diff', '--quiet', 'HEAD', '--')
        print(f'{name}: verified {actual}', flush=True)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    if args.install_python:
        python_version = lock['runtime']['python']
        run('uv', 'python', 'install', python_version, '--install-dir', project/'.python')
        interpreter = project/f'.python/cpython-{python_version}-linux-x86_64-gnu/bin/python3.12'
        python = project/'.venv-core17/bin/python'
        if not python.exists():
            run('uv', 'venv', '--relocatable', '--python', interpreter, project/'.venv-core17')
        lines = (project/'environment.core17.txt').read_text().splitlines()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as requirements:
            requirements.write('\n'.join(line for line in lines if not line.startswith('-e '))+'\n')
            requirements.flush()
            run('uv','pip','sync','--python',python,'--index-strategy','unsafe-best-match',
                '--extra-index-url','https://download.pytorch.org/whl/cpu',requirements.name,env=env)
        for name in ('MegatronAdaptor','TransformerEngineNPU','MindSpeed'):
            run('uv','pip','install','--python',python,'--no-deps','--no-build-isolation',
                '-e',deps/'core17'/name,env=env)
    if args.with_cann or args.install_runtime:
        downloads = deps/'cann90-packages'; downloads.mkdir(exist_ok=True)
        for entry in lock['cann_packages']:
            path = downloads/entry['file']
            if not path.exists():
                temporary = path.with_suffix('.partial')
                run('curl','-fL','--retry','3','--connect-timeout','15',
                    '-H','Referer: https://www.hiascend.com/',entry['url'],'-o',temporary)
                temporary.rename(path)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            if digest != entry['sha256']:
                raise RuntimeError(f'{path}: SHA256 mismatch')
            print(f'{entry["kind"]}: verified {digest}',flush=True)
        if args.install_runtime:
            prefix = deps/'Ascend90'
            env['PATH'] = str(project/'.venv-core17/bin')+':'+env['PATH']
            for kind in ('toolkit','910b-ops','nnal'):
                installer = downloads/next(e['file'] for e in lock['cann_packages'] if e['kind']==kind)
                command = 'if [ -f "$1/cann/set_env.sh" ]; then source "$1/cann/set_env.sh"; fi\nexec bash "$2" --quiet --install --install-for-all --install-path="$1"'
                run('bash','-c',command,'cann-install',prefix,installer,env=env)


if __name__ == '__main__':
    main()
