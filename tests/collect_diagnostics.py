"""Check recent local conversation/timing logs and collect one diagnostic ZIP."""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def recent_files(root, *, all_sessions=False):
    logs = root / 'logs'
    sessions = sorted((p for p in (logs / 'services').glob('*') if p.is_dir()), key=lambda p: p.name)
    session = sessions[-1] if sessions else None
    if all_sessions:
        files = [p for folder in sessions for p in folder.glob('*.log')]
        for directory, pattern in [('conversations', '*.jsonl'), ('timing', '*.log')]:
            files.extend((logs / directory).glob(pattern))
        return session, sorted(p for p in set(files) if p.resolve().is_relative_to(logs.resolve()))
    if session:
        try:
            cutoff = datetime.strptime(session.name, '%Y%m%dT%H%M%S_%fZ').replace(tzinfo=timezone.utc).timestamp() - 5
        except ValueError:
            cutoff = session.stat().st_mtime - 5
    else:
        cutoff = datetime.now(timezone.utc).timestamp() - 86400
    files = list(session.glob('*.log')) if session else []
    for directory, pattern in [('conversations', '*.jsonl'), ('timing', '*.log')]:
        files.extend(p for p in (logs / directory).glob(pattern) if p.stat().st_mtime >= cutoff)
    return session, sorted(p for p in set(files) if p.resolve().is_relative_to(logs.resolve()))


def inspect(root, files):
    messages = Counter()
    missing_ids = 0
    malformed = []
    invalid_messages = []
    requests = defaultdict(Counter)
    context_records = 0
    turns = []
    timing_requests = set()
    chat_completions = {}
    timing_errors = []
    resets = []
    for path in files:
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8', errors='replace') as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        value = json.loads(line)
                        if value.get('event') == 'message':
                            role = value.get('role', 'unknown')
                            if role not in {'user', 'assistant'} or not isinstance(value.get('content'), str) or value.get('conversation_id') is None:
                                invalid_messages.append(f'{path.relative_to(root)}:{number}')
                            messages[role] += 1
                            request = value.get('request_id')
                            if not request or request == '-':
                                missing_ids += 1
                            else:
                                requests[request][role] += 1
                        elif value.get('event') == 'turn':
                            turns.append(value)
                        elif value.get('event') == 'interaction':
                            context_records += 1
                    except (ValueError, TypeError, AttributeError):
                        malformed.append(f'{path.relative_to(root)}:{number}')
        elif path.parent.name == 'timing':
            with path.open(encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    if 'TIMING ' not in line:
                        continue
                    try:
                        value = json.loads(line.split('TIMING ', 1)[1])
                    except ValueError:
                        continue
                    identity = value.get('request_id')
                    if identity and identity != '-':
                        timing_requests.add(identity)
                    if value.get('status') == 'error' and value.get('error') not in {'RerunException', 'StopException'}:
                        timing_errors.append({key: value.get(key) for key in ('request_id', 'task', 'error', 'elapsed_ms')})
                    if value.get('task') == 'backend.http_request' and value.get('event') == 'end' and value.get('method') == 'POST' and '/chat/' in value.get('path', ''):
                        chat_completions[identity] = {key: value.get(key) for key in ('status', 'status_code', 'elapsed_ms')}
        else:
            context = deque(maxlen=60)
            with path.open(encoding='utf-8', errors='replace') as stream:
                for number, line in enumerate(stream, 1):
                    context.append(line.rstrip())
                    if 'ConnectionResetError' in line or 'WinError 10054' in line:
                        resets.append({'file': str(path.relative_to(root)), 'line': number, 'preceding_context': list(context)})
    paired = sum(bool(roles['user'] and roles['assistant']) for roles in requests.values())
    report = {
        'message_counts': dict(messages),
        'turn_count': len(turns),
        'turn_status_counts': dict(Counter(t.get('status') for t in turns)),
        'requests_with_user_and_assistant': paired,
        'messages_missing_request_ids': missing_ids,
        'conversation_requests_missing_timing': sorted(set(requests) - timing_requests),
        'requests_with_user_but_no_saved_assistant': sorted(key for key, roles in requests.items() if roles['user'] and not roles['assistant']),
        'interaction_records': context_records,
        'malformed_conversation_lines': malformed,
        'invalid_message_records': invalid_messages,
        'chat_completions': chat_completions,
        'timing_errors': timing_errors[-50:],
        'connection_resets': resets[-10:],
    }
    missing_completed_answers = [key for key, roles in requests.items() if roles['user'] and not roles['assistant']
                                 and chat_completions.get(key, {}).get('status') == 'ok'
                                 and chat_completions.get(key, {}).get('status_code') == 200]
    report['completed_chats_missing_saved_answer'] = missing_completed_answers
    report['conversation_capture_check'] = 'pass' if paired and not missing_ids and not malformed and not invalid_messages and not missing_completed_answers and not report['conversation_requests_missing_timing'] else 'needs_review'
    versions = {}
    for name in ('uvicorn', 'fastapi', 'starlette', 'httpx', 'h11', 'streamlit', 'onnxruntime-directml', 'llama-cpp-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    report['runtime'] = {'python': platform.python_version(), 'system': platform.system(), 'versions': versions}
    report['note'] = ('This checks saved diagnostics, not answer correctness. An in-progress or cancelled chat may lack an assistant record. '
                      'Connection-reset context is evidence for investigation, not a confirmed cause.')
    return report


def transcript(files):
    turns, messages = [], []
    for path in files:
        if path.suffix != '.jsonl':
            continue
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                item = json.loads(line)
                if item.get('event') == 'turn':
                    turns.append(item)
                elif item.get('event') == 'message':
                    messages.append(item)
            except (ValueError, AttributeError):
                continue
    entries = []
    summarized = {(t.get('request_id'), t.get('pid'), t.get('conversation_id')) for t in turns}
    for turn in turns:
        lines = [
            'Time: ' + str(turn.get('started_at')),
            'Chat: ' + str(turn.get('conversation_id')) + ' | Request: ' + str(turn.get('request_id')),
            'Domain: ' + str(turn.get('selected_domain')) + ' | Status: ' + str(turn.get('status')) + ' | Duration (ms): ' + str(turn.get('elapsed_ms')),
            'Classifier: ' + json.dumps(turn.get('classifier_decision'), ensure_ascii=False),
            'USER:', turn.get('user_prompt') or '',
            'ASSISTANT:', turn.get('assistant_response') or '[No response captured]',
            'Error: ' + str(turn.get('error') or ''), '', '=' * 60, '',
        ]
        entries.append((turn.get('started_at', turn.get('timestamp', '')), '\n'.join(lines)))
    # Include form/MCP action replies and older logs which have no combined turn.
    for message in messages:
        key = (message.get('request_id'), message.get('pid'), message.get('conversation_id'))
        if key in summarized:
            continue
        text = '\n'.join([
            'Time: ' + str(message.get('timestamp')),
            'Chat: ' + str(message.get('conversation_id')) + ' | Request: ' + str(message.get('request_id')),
            'Saved message (no combined turn summary):',
            str(message.get('role', '')).upper() + ':', message.get('content') or '',
            '', '=' * 60, '',
        ])
        entries.append((message.get('timestamp', ''), text))
    return '\n'.join(text for _, text in sorted(entries, key=lambda item: item[0])) or 'No chat records found.\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Print log checks without making a ZIP')
    parser.add_argument('--all-sessions', action='store_true', help='Include all existing log sessions, including chats before a service restart')
    args = parser.parse_args()
    session, files = recent_files(ROOT, all_sessions=args.all_sessions)
    if not files:
        parser.exit(1, 'No recent service/timing/conversation logs found. Start services and complete a chat first.\n')
    report = inspect(ROOT, files)
    print('Saved messages: ' + json.dumps(report['message_counts']))
    print('Requests with prompt + answer: ' + str(report['requests_with_user_and_assistant']))
    print('Conversation capture: ' + report['conversation_capture_check'])
    print('Connection resets found: ' + str(len(report['connection_resets'])))
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    directory = ROOT / 'diagnostic_exports'
    directory.mkdir(exist_ok=True)
    name = 'diagnostics_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.zip'
    destination = directory / name
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('chat_transcript.txt', transcript(files))
        archive.writestr('diagnostic_report.json', json.dumps(report, ensure_ascii=False, indent=2))
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print('Created: ' + str(destination))
    print('Includes full prompt/answer text and local diagnostic logs. Nothing was uploaded.')


if __name__ == '__main__':
    main()
