"""Readiness probes never generate an answer or load a second model copy."""
import asyncio
from pathlib import Path
import httpx
from sqlalchemy import text
from fastapi import Depends
from database import get_db
from rag.domain_config import list_domain_configs
from rag.service_config import llm_url
from rag.operations import snapshot


async def model_status(path='/v1/rag/status'):
    try:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            response = await client.get(llm_url()+path)
            response.raise_for_status()
            value=response.json()
            return value if isinstance(value,dict) else None
    except (httpx.HTTPError, ValueError):
        return None


async def readiness(db):
    checks = {'database': False, 'datasets': False, 'model': False, 'retrieval_assets': False}
    try:
        await asyncio.wait_for(db.execute(text('SELECT 1')), timeout=3)
        checks['database'] = True
    except Exception:
        pass
    try:
        configs = list_domain_configs()
        checks['datasets'] = any(c.db_path.is_file() and c.vector_dir.is_dir() for c in configs)
    except Exception:
        pass
    status = await model_status()
    checks['model'] = bool(status and status.get('qwen_ready'))
    checks['retrieval_assets'] = bool(status and status.get('retrieval_assets_ready'))
    return {'ready': all(checks.values()), 'checks': checks}


def recent_alerts():
    result = []
    root = Path(__file__).resolve().parents[2]
    import json
    for path in sorted((root/'logs/operations').glob('alerts*.log'), key=lambda p:p.stat().st_mtime, reverse=True)[:10]:
        try:
            for line in path.read_text(encoding='utf-8')[-12000:].splitlines():
                try: result.append(json.loads(line))
                except ValueError: pass
        except OSError:
            pass
    return sorted(result,key=lambda e:e.get('time',0),reverse=True)[:20]


async def operations_status(db):
    return {'backend': snapshot(), 'model': await model_status('/v1/operations'), 'readiness': await readiness(db), 'alerts': recent_alerts()}
