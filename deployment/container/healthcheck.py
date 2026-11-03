"""Non-generating readiness check for the three supervised services."""
import json
import urllib.request


def main():
    for url in ('http://127.0.0.1:8000/readyz',
                'http://127.0.0.1:8501/_stcore/health'):
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError('Service not ready')
            if url.endswith('/readyz') and not json.load(response).get('ready'):
                raise RuntimeError('Dependencies not ready')


if __name__ == '__main__':
    main()
