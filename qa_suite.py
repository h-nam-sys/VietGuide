"""Live chat API QA with frozen cases, preserved grading and cache diagnostics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import difflib
import hashlib
import time
import unicodedata
import http.cookiejar
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / 'testing' / 'fixtures' / 'qa_cases.json'


def normalize(value):
    text = unicodedata.normalize('NFD', str(value).lower().replace('đ', 'd'))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^\w\s.,]', ' ', text).split())


# Language-level unit equivalence for QA only; never rewrites a production answer.
_UNIT = r'(?:san pham|don vi|ban|the|may|gio|luot|nguoi|ngay|thang|nam|kg|lit|ve|phong)'


def unit_equivalent_text(value):
    """Recognize explicit distributive units without borrowing units from the question.

    Keep the original matcher too. Reject implicit-rate inference across negation,
    sentence boundaries or a conflicting explicit rate (e.g. per month).
    """
    raw = unicodedata.normalize('NFD', str(value).lower().replace('đ','d'))
    raw = ''.join(c for c in raw if not unicodedata.combining(c))
    raw = re.sub(r'(?<![a-z])(dong|vnd|d)\s*/\s*(' + _UNIT + r')\b', r'\1 moi \2', raw)
    text = normalize(raw)
    text = re.sub(r'\btung\s+(' + _UNIT + r')\b', r'moi \1', text)
    hints = []
    pattern = (r'\b(?:mot|1)\s+(?P<unit>' + _UNIT + r')\b'
               r'(?P<tail>[^.!?\n]{0,80}?)'
               r'(?P<amount>\d[\d., ]*\s*(?:dong|vnd|d)\b)')
    bounded = normalize(raw.replace('\n', '.'))
    bounded = re.sub(r'\btung\s+(' + _UNIT + r')\b', r'moi \1', bounded)
    for match in re.finditer(pattern, bounded):
        tail = match.group('tail')
        if re.search(r'\b(?:khong|chua|mien|tru|giam)\b', tail):
            continue
        if tail.strip() and not re.search(r'\b(?:la|gia|phi|tinh|ton|het)\b',tail):
            continue
        following = bounded[match.end():match.end()+35]
        rate = re.match(r'\s*(?:moi|tung)\s+(' + _UNIT + r')\b', following)
        if rate and rate.group(1) != match.group('unit'):
            continue
        hints.append('moi ' + match.group('unit'))
    return text + (' ' + ' '.join(hints) if hints else '')


def parse_sse(lines, started=None):
    parts, metadata, errors, done, buffer = [], {}, [], False, []
    first_text_ms = None
    started = time.perf_counter() if started is None else started
    def consume():
        nonlocal done, first_text_ms
        if not buffer:
            return
        data = '\n'.join(buffer)
        buffer.clear()
        if data == '[DONE]':
            done = True
            return
        try:
            value = json.loads(data)
            if value.get('type') == 'chunk':
                if first_text_ms is None and value.get('text'):
                    first_text_ms = round((time.perf_counter()-started)*1000, 3)
                parts.append(value.get('text', ''))
            elif value.get('type') == 'metadata':
                metadata.update(value)
            elif 'error' in value:
                errors.append(str(value['error']))
        except (ValueError, AttributeError, TypeError) as exc:
            errors.append('Invalid SSE event: ' + str(exc))
    for raw in lines:
        line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if not line:
            consume()
        elif line.startswith('data:'):
            buffer.append(line[5:].lstrip(' '))
    consume()
    return {'answer': ''.join(parts), 'metadata': metadata, 'done': done, 'stream_errors': errors, 'first_text_ms': first_text_ms}


class Client:
    def __init__(self, base, user_id, interval=6.5, timeout=300):
        self.base, self.user_id, self.interval, self.timeout = base.rstrip('/'), user_id, interval, timeout
        self.last_chat = None
        self.last_trace = None
        self.traces = set()
        self.cookies = http.cookiejar.CookieJar()
        self.csrf_token = ''
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(self.cookies))

    def request(self, path, payload=None, stream=False):
        if payload is not None and not self.csrf_token:
            session, _, _ = self.request('/session')
            self.csrf_token = session['csrf_token']
        is_chat = bool(re.fullmatch(r'/chat/\d+', path))
        if is_chat and self.last_chat is not None:
            remaining = self.interval - (time.monotonic() - self.last_chat)
            if remaining > 0:
                time.sleep(remaining)
        trace = uuid.uuid4().hex[:16]
        self.traces.add(trace)
        self.last_trace = trace
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data,
            headers={'Content-Type': 'application/json', 'X-Session-CSRF': self.csrf_token,
                     'X-Legal-Timing-Trace': trace})
        if is_chat:
            self.last_chat = time.monotonic()
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = parse_sse(response, started=started) if stream else json.load(response)
            if path == '/session':
                self.csrf_token = result['csrf_token']
                token = next((c.value for c in self.cookies if c.name == 'legal_session'), '')
                if token:
                    self.user_id = 'anon:' + hashlib.sha256(token.split('.')[1].encode()).hexdigest()
            return result, trace, round((time.perf_counter() - started) * 1000, 3)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:1000]
            raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc


def collect_logs(root, traces):
    """Include only this run's trace IDs; no need to pick a prior ZIP."""
    events, files = [], {}
    for folder, pattern in [('timing', '*.log'), ('conversations', '*.jsonl')]:
        for path in sorted((root / 'logs' / folder).glob(pattern)):
            lines = []
            with path.open(encoding='utf-8', errors='replace') as source:
                for line in source:
                    try:
                        value = json.loads(line.split('TIMING ', 1)[1] if folder == 'timing' else line)
                    except (ValueError, IndexError):
                        continue
                    if value.get('request_id') in traces:
                        events.append(value)
                        lines.append(line)
            if lines:
                files[path.relative_to(root).as_posix()] = ''.join(lines)
    return events, files


def parse_sources(message):
    values = message.get('sources') or []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except ValueError:
            return []
    return values if isinstance(values, list) else []


def check_turn(turn, events, source_audit=None, *, unit_equivalence=True):
    checks = []
    def check(name, valid, detail='', severity='FAIL'):
        checks.append({'name': name, 'status': 'PASS' if valid else severity, 'detail': detail})
    if turn.get('error'):
        return 'ERROR', [{'name':'request', 'status':'ERROR', 'detail':turn['error']}]
    expected, actual = turn['expected'], turn['actual']
    relevant = [e for e in events if e.get('request_id') == turn['request_id']]
    cache_events = [e for e in relevant if e.get('task') == 'answer.cache']
    if turn.get('payload', {}).get('bypass_answer_cache') and cache_events:
        check('answer_cache_bypassed', all(e.get('cache_enabled') is False and not e.get('cache_hit') for e in cache_events),
              'Cacheless QA requires backend acknowledgement and no cache hits.')
    if type(turn.get('payload', {}).get('answer_guidance')) is bool and cache_events:
        wanted_guidance = turn['payload']['answer_guidance']
        check('answer_guidance_acknowledged', all(
            e.get('runtime_options', {}).get('RAG_ANSWER_GUIDANCE') is wanted_guidance for e in cache_events),
            'The answering backend must acknowledge the requested guidance setting.')
    decisions = [e['decision'] for e in relevant if e.get('task') == 'backend.services.domain_router.classifier' and e.get('event') == 'decision']
    decision = decisions[-1] if decisions else None
    turn['classifier_decision'] = decision
    selected = [e for e in relevant if e.get('task') == 'rag.evidence.diagnostic' and e.get('event') == 'field_selection']
    fields = selected[-1].get('selected', []) if selected else []
    completed = [e for e in relevant if e.get('task') == 'rag.evidence.diagnostic' and e.get('event') == 'complete']
    cached = [e for e in relevant if e.get('task') == 'answer.cache' and e.get('cache_hit')
              and isinstance(e.get('evidence_fields'), list)]
    if completed:
        fields = completed[-1].get('fields', [])
    elif cached:
        fields = cached[-1]['evidence_fields']
    validated = [e for e in relevant if e.get('task') == 'rag.answer.diagnostic'
                 and e.get('event') == 'validated']
    if validated:
        fields = validated[-1].get('evidence_fields', [])
    field_evidence_available = bool(selected or completed or cached or validated)
    turn['selected_fields'] = fields
    stream = actual['stream']
    saved = actual.get('saved_message') or {}
    sources = parse_sources(saved)
    titles = [s for s in sources if isinstance(s, str) and not s.startswith('RAG_')]
    domain = stream['metadata'].get('domain')
    source_domain = next((s[len('RAG_DOMAIN:'):] for s in sources if isinstance(s, str) and s.startswith('RAG_DOMAIN:')), None)
    state_domain = (actual.get('history', {}).get('checklist', {}).get('_domain_state') or {}).get('name')
    turn['observed'] = {'domain':domain or source_domain, 'state_domain':state_domain, 'titles':titles}
    check('stream_completed', stream['done'])
    check('no_stream_errors', not stream['stream_errors'], str(stream['stream_errors']))
    check('nonempty_answer', bool(stream['answer'].strip()))
    check('answer_saved', bool(saved) and saved.get('content') == stream['answer'])
    if 'domain' in expected:
        check('domain', (domain or source_domain) == expected['domain'], str(domain or source_domain))
    if 'state_domain' in expected:
        check('domain_state', state_domain == expected['state_domain'], str(state_domain))
    if 'routes' in expected:
        route = decision.get('route') if isinstance(decision, dict) else None
        check('route_action', route in expected['routes'], str(route), severity='FAIL' if decision else 'REVIEW')
    if expected.get('no_retrieval'):
        retrieval = [e for e in relevant if e.get('task') == 'rag.retrieval.diagnostic' and e.get('event') == 'request']
        check('no_dataset_retrieval', not retrieval and not titles, str(titles), severity='FAIL' if relevant else 'REVIEW')
    if 'records' in expected:
        wanted = {normalize(t) for t in expected['records']}
        found = {normalize(t) for t in titles}
        check('required_records', wanted <= found, 'Found: ' + str(titles))
    if expected.get('exact_records'):
        check('no_extra_records', {normalize(t) for t in titles} == {normalize(t) for t in expected.get('records', [])}, str(titles))
    for forbidden in expected.get('forbidden_records', []):
        check('no_stale_record:' + forbidden, normalize(forbidden) not in {normalize(t) for t in titles})
    for field in expected.get('fields', []):
        check('field:' + field, field in fields, str(fields), severity='FAIL' if field_evidence_available else 'REVIEW')
    text = normalize(stream['answer'])
    unit_text = unit_equivalent_text(stream['answer']) if unit_equivalence else text
    if expected.get('source_absence'):
        audit = (source_audit or {}).get('domains', {}).get(expected['source_absence'], {})
        check('source_absence_oracle', audit.get('status') == 'no_title_match'
              and audit.get('rows_scanned', 0) > 0, str(audit.get('status')), severity='REVIEW')
        check('no_substitute_records', not titles, str(titles))
        check('explicit_no_match', bool(re.search(r'(?:khong|chua).*(?:tim thay|thong tin|du lieu)', text)))
    for pattern in expected.get('answer_patterns', []):
        check('answer_pattern:' + pattern, bool(re.search(pattern, text) or re.search(pattern, unit_text)), 'Required answer fact absent.' if expected.get('require_answer_patterns') else 'Wording-dependent check; inspect answer if absent.', severity='FAIL' if expected.get('require_answer_patterns') else 'REVIEW')
    for pattern in expected.get('forbidden_patterns', []):
        check('no_injected_value:' + pattern, not re.search(pattern, text))
    for pattern in expected.get('contradiction_patterns', []):
        check('no_contradictory_caveat:' + pattern, not re.search(pattern, text),
              'Answer contains a declared contradictory or unsupported caveat.')
    patterns = expected.get('answer_any_patterns', [])
    if patterns:
        check('accepted_answer_form', any(re.search(pattern, text) or re.search(pattern, unit_text) for pattern in patterns),
              'Answer did not match any declared response form.')
    if expected.get('any_of'):
        alternatives = []
        for option in expected['any_of']:
            branch = {**turn, 'expected':option}
            alternatives.append(check_turn(branch, events, source_audit, unit_equivalence=unit_equivalence))
        rank = {'PASS':0, 'REVIEW':1, 'FAIL':2, 'ERROR':3}
        best_status, best_checks = min(alternatives, key=lambda item:rank[item[0]])
        check('accepted_response_policy', best_status=='PASS',
              'At least one declared refusal or grounded-answer policy must pass.', severity=best_status)
        if best_status != 'PASS':
            checks.extend({**c, 'name':'policy:' + c['name']} for c in best_checks if c['status']!='PASS')
    if expected.get('review_answer'):
        checks.append({'name':'answer_correctness', 'status':'REVIEW', 'detail':'Routing is checked; this answer has no supplied factual oracle.'})
    status = 'FAIL' if any(c['status'] == 'FAIL' for c in checks) else 'REVIEW' if any(c['status'] == 'REVIEW' for c in checks) else 'PASS'
    return status, checks


def execute_suite(client, scenarios, checkpoint, repeats=1, *, cacheless=False):
    results = []
    total = sum(len(c['turns']) for c in scenarios) * repeats
    for repeat in range(1, repeats + 1):
        for scenario in scenarios:
            cid = None
            start_error = None
            try:
                response, _, _ = client.request('/start-conversation', {
                    'title': 'QA ' + scenario['id'], 'domain': scenario.get('domain','auto'),
                    'allow_switching':scenario.get('allow_switching',True)})
                cid = response['conversation_id']
            except Exception as exc:
                start_error = str(exc)
            broken_context = False
            for index, step in enumerate(scenario['turns'], 1):
                print(f"[{len(results) + 1}/{total}] {scenario['id']} / turn {index} / repeat {repeat}: {step['prompt']}", flush=True)
                result = {'scenario':scenario['id'], 'repeat':repeat, 'turn':index,
                          'conversation_id':cid, 'prompt':step['prompt'], 'expected':step.get('expected', {})}
                if start_error or broken_context:
                    result['error'] = start_error or 'Earlier request failed; dependent context is not reliable.'
                else:
                    try:
                        payload = {'user_prompt':step['prompt'], 'domain':scenario.get('domain','auto'), 'mode':scenario.get('mode','strict')}
                        payload.update(step.get('options', {}))
                        if cacheless:
                            payload['bypass_answer_cache'] = True
                        stream, trace, elapsed = client.request('/chat/' + str(cid), payload, stream=True)
                        result.update(request_id=trace, elapsed_ms=elapsed, payload=payload)
                        history, _, _ = client.request('/history/' + str(cid))
                        messages = history.get('messages', [])
                        saved = next((m for m in reversed(messages) if m.get('role') == 'assistant'), None)
                        result['actual'] = {'stream':stream, 'saved_message':saved,
                            'history': {'checklist':history.get('checklist', {}), 'current_procedure':history.get('current_procedure')}}
                        print('  ' + stream['answer'].replace('\n',' ')[:150], flush=True)
                    except Exception as exc:
                        result['error'] = str(exc)
                        result.setdefault('request_id', getattr(client, 'last_trace', None))
                        broken_context = True
                results.append(result)
                with checkpoint.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    return results


def performance_summary(results, events):
    """Measured end-to-end latency; cold first turn is reported separately."""
    cache = {}
    for event in events:
        if event.get('task') == 'answer.cache':
            cache[event.get('request_id')] = event
    timed = [r for r in results if isinstance(r.get('elapsed_ms'), (int, float))]
    def stats(turns):
        values = sorted(r['elapsed_ms'] / 1000 for r in turns)
        if not values:
            return {'count': 0, 'mean_seconds': None, 'median_seconds': None, 'p95_seconds': None}
        # Linear interpolation; include the definition so reports are reproducible.
        position = (len(values) - 1) * .95
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return {'count': len(values), 'mean_seconds': statistics.mean(values),
                'median_seconds': statistics.median(values),
                'p95_seconds': values[low] + (values[high] - values[low]) * (position - low)}
    warm = timed[1:]
    tasks = ('backend.services.domain_router.choose_domain',
             'rag.query_preparation.prepare_search_queries', 'rag.retrieval_service.rerank_records',
             'rag.query_pipeline.select_requested_records', 'backend.services.llm_service.stream_generator',
             'llm.nonstream_inference')
    stages = {task: stats([e for e in events if e.get('task') == task and e.get('event') == 'end'
                          and isinstance(e.get('elapsed_ms'), (int, float))]) for task in tasks}
    return {'stage_timings': stages, 'all': stats(timed), 'excluding_first_request': stats(warm),
            'uncached_excluding_first_request': stats([r for r in warm
                if not cache.get(r.get('request_id'), {}).get('cache_hit')]),
            'first_request_seconds': timed[0]['elapsed_ms'] / 1000 if timed else None,
            'cache_hits': sum(bool(cache.get(r.get('request_id'), {}).get('cache_hit')) for r in results),
            'cache_bypass_acknowledged': sum(cache.get(r.get('request_id'), {}).get('cache_enabled') is False for r in results),
            'answer_cache_events': sum(r.get('request_id') in cache for r in results),
            'note': 'Uncached excludes recorded answer-cache hits, not model/index caches. P95 uses linear interpolation.'}


def report(results):
    counts = Counter(r['status'] for r in results)
    lines = ['SYSTEM TEST RESULTS', json.dumps(dict(counts)),
             'PASS means the stated checks passed, not that every factual claim is verified.',
             'REVIEW includes wording-dependent checks or answers without factual ground truth.',
             'UI layout, feedback clicks, concurrency, and approved online actions are not tested.', '']
    if any('previous_checker' in r for r in results):
        strict_counts = Counter(r.get('previous_checker', {}).get('status', r['status']) for r in results)
        lines.insert(2, 'Strict-unit wording counts on the same answers: ' + json.dumps(dict(strict_counts)))
        lines.insert(3, 'Current counts accept equivalent explicit unit wording; recorded answers are unchanged.')
    for turn in results:
        lines.extend([f"[{turn['status']}] {turn['scenario']} / turn {turn['turn']} / repeat {turn['repeat']}",
                      'USER: ' + turn['prompt'],
                      'ASSISTANT: ' + turn.get('actual', {}).get('stream', {}).get('answer', '[request failed]'),
                      'Route: ' + json.dumps(turn.get('classifier_decision'), ensure_ascii=False),
                      'Records: ' + json.dumps(turn.get('observed', {}).get('titles', []), ensure_ascii=False),
                      'Fields: ' + json.dumps(turn.get('selected_fields', []), ensure_ascii=False)])
        for check in turn['checks']:
            if check['status'] != 'PASS':
                lines.append(check['status'] + ': ' + check['name'] + ' - ' + check.get('detail',''))
        lines.append('')
    return '\n'.join(lines)


def audit_sources(root, domains):
    """Read current SQLite titles; report availability without inventing an oracle.

    Matching is deliberately mechanical. No match is not proof of absence under
    aliases. Nearby titles are diagnostics, never acceptable substitute answers.
    """
    targets = {'entertainment_content':'Interstella', 'amazon_products':'Sony WH-1000XM5'}
    report = {'note':'Read-only source inspection. Title matches/nearby titles do not prove semantic completeness.', 'domains':{}}
    sys.path.insert(0, str(root))
    for name, target in targets.items():
        if name not in domains:
            continue
        try:
            from rag.domain_config import load_domain_config, identifier
            config = load_domain_config(name, root=root)
            table = identifier(config.table)
            primary = identifier(config.primary_entity)
            record_id = identifier(config.id_column)
            compact = lambda value: ''.join(c for c in normalize(value) if c.isalnum())
            query = compact(target)
            entries, nearby, total = [], [], 0
            with sqlite3.connect(config.db_path.resolve().as_uri()+'?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                for row in db.execute(f'SELECT "{record_id}", "{primary}" FROM "{table}"'):
                    total += 1
                    title = str(row[primary] or '')
                    key = compact(title)
                    # Movies often contain years/suffixes; Sony titles often
                    # insert other words between the brand and model number.
                    match = any(alias in key for alias in ('interstella', 'interstellar')) if name=='entertainment_content' else 'sony' in key and 'wh1000xm5' in key
                    if match:
                        entries.append({'id':str(row[record_id]), 'title':title})
                    nearby.append((difflib.SequenceMatcher(None, query, key).ratio(), title))
                    nearby = sorted(nearby, reverse=True)[:3]
                public = [identifier(key) for key in config.evidence_fields]
                columns = ', '.join('"'+key+'"' for key in public)
                for entry in entries[:5]:
                    row = db.execute(f'SELECT {columns} FROM "{table}" WHERE "{record_id}"=?', (entry['id'],)).fetchone()
                    entry['public_values'] = {key:str(row[key])[:600] for key in public if row[key] is not None} if row else {}
            report['domains'][name] = {'target':target, 'generation':config.generation,
                'revision':config.revision, 'rows_scanned':total, 'title_match_count':len(entries),
                'title_matches':entries[:5], 'nearby_titles':[title for _,title in nearby],
                'status':'title_matches_found' if entries else 'no_title_match'}
        except (OSError, ValueError, RuntimeError, sqlite3.Error, ImportError) as exc:
            report['domains'][name] = {'target':target, 'status':'unavailable', 'error':str(exc)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--suite', choices=['full','smoke','review'], default='full')
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--repeats', type=int, choices=[1,2,3], default=1)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--require-all-domains', action='store_true')
    parser.add_argument('--cacheless', action='store_true', help='Bypass generated-answer cache reads AND writes on every turn')
    args = parser.parse_args()
    manifest = json.loads(args.cases.read_text(encoding='utf-8'))
    review_cases = {'manual_domain_lock', 'auto_detect_once_lock', 'movie_topic',
                    'product_topic', 'unsupported_detail', 'injection_resistance'}
    scenarios = [c for c in manifest['scenarios'] if args.suite == 'full'
                 or (args.suite == 'smoke' and c.get('smoke'))
                 or (args.suite == 'review' and c['id'] in review_cases)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = ROOT/'qa_exports'/stamp
    directory.mkdir(parents=True)
    client = Client(args.base_url, str(uuid.uuid4()), timeout=args.timeout)
    health, _, _ = client.request('/health')
    available, _, _ = client.request('/domains')
    names = {d['id'] for d in available}
    if args.require_all_domains:
        missing = set().union(*(set(c.get('requires', [])) for c in scenarios)) - names
        if missing:
            raise RuntimeError('Required test domains unavailable: ' + ', '.join(sorted(missing)))
    selected, skipped = [], []
    for case in scenarios:
        missing = set(case.get('requires', [])) - names
        if missing:
            skipped.append({'scenario':case['id'], 'reason':'Unavailable domains: ' + ', '.join(sorted(missing))})
        else:
            selected.append(case)
    source_audit = audit_sources(ROOT, names)
    (directory/'source_audit.json').write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding='utf-8')
    total = sum(len(c['turns']) for c in selected) * args.repeats
    print(f'{total} turns in {len(selected)} scenarios. Suite: {args.suite}', flush=True)
    print('Separate test run: ' + client.user_id, flush=True)
    print('Uses the real backend/model. Keep services running; avoid other chats during the run.', flush=True)
    checkpoint = directory/'checkpoint.jsonl'
    results = execute_suite(client, selected, checkpoint, args.repeats, cacheless=args.cacheless)
    # Responses are persisted before DONE; short flush allowance is for diagnostic files only.
    time.sleep(.25)
    events, files = collect_logs(ROOT, client.traces)
    for turn in results:
        old_status, old_checks = check_turn(turn, events, source_audit, unit_equivalence=False)
        turn['previous_checker'] = {'policy':'same checks with strict unit wording', 'status':old_status, 'checks':old_checks}
        turn['status'], turn['checks'] = check_turn(turn, events, source_audit)
    summary = report(results)
    summary += '\nPERFORMANCE (seconds):\n' + json.dumps(performance_summary(results, events), indent=2)
    (directory/'summary.txt').write_text(summary, encoding='utf-8')
    (directory/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    run = {'suite':args.suite, 'test_user_id':client.user_id, 'base_url':args.base_url,
           'repeats':args.repeats, 'turn_count':len(results), 'available_domains':sorted(names),
           'skipped_scenarios':skipped, 'counts':dict(Counter(r['status'] for r in results)),
           'diagnostic_events':len(events),
           'answer_cache_bypassed':args.cacheless,
           'checker_version':2,
           'previous_checker_counts':dict(Counter(r['previous_checker']['status'] for r in results)),
           'performance':performance_summary(results, events),
           'runtime_options_observed':[json.loads(value) for value in sorted({
               json.dumps(e['runtime_options'],sort_keys=True) for e in events
               if e.get('task')=='answer.cache' and isinstance(e.get('runtime_options'),dict)})],
           'runtime_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                            for folder in ('rag','backend/services') for path in sorted((ROOT/folder).glob('*.py'))},
           'qa_harness_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ('backend/schemas.py','qa_suite.py','testing/run_cacheless_tests.py','testing/targeted_analysis.py') if (ROOT/name).is_file()},
           'note':'Tests create isolated chat records; no existing chats are changed. Restart services before running so on-disk hashes describe the loaded code.'}
    (directory/'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')
    destination = directory.parent/('system_tests_' + stamp + '.zip')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ('summary.txt','results.json','run.json','checkpoint.jsonl','source_audit.json'):
            if (directory/name).exists():bundle.write(directory/name,name)
        bundle.writestr('qa_cases.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, content in files.items():bundle.writestr(name,content)
    print('\nFinished: ' + json.dumps(run['counts']), flush=True)
    print('Send this ZIP:\n' + str(destination.resolve()), flush=True)
    if skipped:print('Skipped unavailable-domain scenarios: ' + str(len(skipped)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, urllib.error.URLError, RuntimeError) as exc:
        raise SystemExit('System test stopped: ' + str(exc))
