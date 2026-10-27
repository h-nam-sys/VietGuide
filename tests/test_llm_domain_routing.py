"""Dependency-free routing regression tests; classifier responses are mocked.
Run: python -m unittest discover -s tests -p test_llm_domain_routing.py
"""
import asyncio
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import types
import unicodedata
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        httpx = types.ModuleType('httpx')
        httpx.HTTPError = type('HTTPError', (Exception,), {})
        fastapi = types.ModuleType('fastapi')
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                super().__init__(detail)
        fastapi.HTTPException = HTTPException
        qp = types.ModuleType('rag.query_pipeline')
        def plain(text):
            text = unicodedata.normalize('NFD', text.casefold().replace('đ', 'd'))
            return ' '.join(''.join(c for c in text if not unicodedata.combining(c)).split())
        qp._plain = plain
        qp.explicit_fields = lambda *args: []
        with patch.dict(sys.modules, {'httpx': httpx, 'fastapi': fastapi, 'rag.query_pipeline': qp}):
            spec = importlib.util.spec_from_file_location('router_under_test', ROOT / 'backend/services/domain_router.py')
            self.router = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.router)
        self.configs = [types.SimpleNamespace(name=name, primary_entity='topic',
            schema={'topic': 'Chủ đề', 'fee': 'Lệ phí'}, evidence_fields=['topic', 'fee'],
            profile={'description': description, 'routing_examples': [example]})
            for name, description, example in [
                ('administrative_procedures', 'Thủ tục hành chính', 'Đăng ký khai sinh'),
                ('university_admission', 'Tuyển sinh HCMUS', 'Học phí Sinh học')]]
        self.router.ready_domains = lambda: self.configs
        self.router.default_name = lambda: 'administrative_procedures'
        self.router.domain_hints = lambda c: ('Nộp lệ phí xét tuyển nguyện vọng',) if c.name == 'university_admission' else ('Thủ tục đăng ký khai sinh',)
        self.router.span = lambda *a, **kw: nullcontext()
        self.router.local_domain_matches = lambda *a: self.fail('Title shortcut must not run')
        self.events = []
        self.router.emit = lambda task, event, **fields: self.events.append((event, fields))
        self.requests = []
        self.decision = {'route': 'administrative_procedures'}
        self.decisions = []
        owner = self
        class Client:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, **kwargs):
                owner.requests.append(kwargs['json'])
                decision = owner.decisions.pop(0) if owner.decisions else owner.decision
                content = decision if isinstance(decision, str) else json.dumps(decision)
                return types.SimpleNamespace(raise_for_status=lambda: None,
                    json=lambda: {'choices': [{'message': {'content': content}}]})
        self.router.httpx.AsyncClient = Client

    def choose(self, question='Đăng ký khai sinh mất bao lâu và lệ phí bao nhiêu?', state=None, requested='auto', titles=None):
        return asyncio.run(self.router.choose_domain(question, state or {}, titles or [], requested))

    def test_birth_registration_reaches_classifier_with_descriptions(self):
        config, _, _ = self.choose()
        self.assertEqual(config.name, 'administrative_procedures')
        self.assertEqual(len(self.requests), 1)
        request = json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(request['question'], 'Đăng ký khai sinh mất bao lâu và lệ phí bao nhiêu?')
        for actual, expected in zip(request['domains'], self.configs):
            self.assertEqual(actual['description'], expected.profile['description'])
            self.assertEqual(actual['examples'], expected.profile['routing_examples'])
            self.assertEqual(set(actual), {'id', 'description', 'examples'})
        self.assertFalse(self.requests[0]['stream'])
        self.assertEqual(self.requests[0]['max_tokens'], 80)
        self.assertEqual(dict(self.events)['request']['payload'], self.requests[0])

    def test_manual_choice_skips_classifier(self):
        config, state, _ = self.choose(requested='university_admission')
        self.assertEqual(config.name, 'university_admission')
        self.assertEqual(state['selection'], 'manual')
        self.assertFalse(self.requests)

    def test_detect_once_lock_unchanged(self):
        config, _, changed = self.choose(state={'name': 'university_admission', 'locked': True})
        self.assertEqual(config.name, 'university_admission')
        self.assertFalse(changed)
        self.assertFalse(self.requests)

    def test_switching_calls_classifier_with_context(self):
        config, _, changed = self.choose(state={'name': 'university_admission', 'allow_switching': True}, titles=['Học phí Sinh học'])
        self.assertEqual(config.name, 'administrative_procedures')
        self.assertTrue(changed)
        request = json.loads(self.requests[0]['messages'][1]['content'])
        self.assertTrue(request['followup_context_available'])
        self.assertEqual(request['active_context'], {'domain':'university_admission','subjects':['Học phí Sinh học']})
        self.assertNotIn('active_entities', request)

    def test_clear_property_followup_unchanged(self):
        self.router.property_followup = lambda *a: True
        config, _, changed = self.choose(question='Lệ phí bao nhiêu?',
            state={'name': 'administrative_procedures', 'allow_switching': True}, titles=['Đăng ký khai sinh'])
        self.assertEqual(config.name, 'administrative_procedures')
        self.assertFalse(changed)
        self.assertFalse(self.requests)

    def test_classifier_can_still_request_clarification(self):
        self.decision = {'route': 'clarify'}
        with self.assertRaises(self.router.DomainClarification): self.choose()
        self.assertEqual(len(self.requests), 1)

    def test_invalid_or_unknown_classifier_output_still_clarifies(self):
        for decision in ('not JSON', {'route': 'invented'}):
            with self.subTest(decision=decision):
                self.decision = decision
                with self.assertRaises(self.router.DomainClarification): self.choose()

    def test_logs_raw_output_and_success(self):
        self.choose()
        events = dict(self.events)
        self.assertEqual(json.loads(events['raw_decision']['content']), self.decision)
        self.assertEqual(events['selected']['domain'], 'administrative_procedures')

    def test_logs_model_ambiguity_reason(self):
        self.decision = {'route': 'clarify'}
        with self.assertRaises(self.router.DomainClarification): self.choose()
        self.assertEqual(dict(self.events)['clarification']['reason'], 'model_ambiguous')

    def test_logs_invalid_domain_and_preserves_raw_output(self):
        self.decision = {'route': 'invented'}
        with self.assertRaises(self.router.DomainClarification): self.choose()
        events = dict(self.events)
        self.assertIn('invented', events['raw_decision']['content'])
        self.assertEqual(events['clarification']['reason'], 'decision_validation_failed')
        self.assertEqual(events['clarification']['error'], 'Unknown route action or dataset ID')

    def test_logs_both_followup_clarification_reasons(self):
        self.decision = {'route': 'followup'}
        with self.assertRaises(self.router.DomainClarification): self.choose()
        self.assertEqual(dict(self.events)['clarification']['reason'], 'followup_without_current_domain')
        self.events.clear()
        with self.assertRaises(self.router.DomainClarification):
            self.choose(state={'name': 'administrative_procedures', 'allow_switching': True})
        self.assertEqual(dict(self.events)['clarification']['reason'], 'followup_without_active_entities')

    def test_only_ready_domain_does_not_need_classifier(self):
        self.configs = self.configs[:1]
        config, _, _ = self.choose()
        self.assertEqual(config.name, 'administrative_procedures')
        self.assertFalse(self.requests)

    def test_library_route_switches_directly_in_one_call(self):
        self.configs.append(types.SimpleNamespace(name='university_test', primary_entity='topic',
            schema={'topic': 'Chủ đề', 'fee': 'Phí'}, evidence_fields=['topic', 'fee'],
            profile={'description': 'Dịch vụ sinh viên Demo', 'routing_examples': ['Mượn sách thư viện']}))
        self.decision = {'route': 'university_test'}
        config, state, switched = self.choose(
            question='Ở thư viện Demo, mỗi lần được mượn mấy cuốn và trong bao lâu?',
            state={'name': 'administrative_procedures', 'allow_switching': True}, titles=['Đăng ký khai sinh'])
        self.assertEqual(config.name, 'university_test')
        self.assertTrue(switched)
        self.assertEqual(state['name'], 'university_test')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(dict(self.events)['decision']['kind'], 'domain')

    def test_legacy_conflicting_output_is_rejected_not_reinterpreted(self):
        self.decision = {'domain': 'university_admission', 'kind': 'followup'}
        with self.assertRaises(self.router.DomainClarification):
            self.choose(state={'name': 'administrative_procedures', 'allow_switching': True}, titles=['Đăng ký khai sinh'])
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn('selected', dict(self.events))

    def test_valid_followup_keeps_domain_without_retry(self):
        self.decision = {'route': 'followup'}
        config, state, switched = self.choose(question='Còn trường hợp đó?',
            state={'name': 'administrative_procedures', 'allow_switching': True}, titles=['Đăng ký khai sinh'])
        self.assertEqual(config.name, 'administrative_procedures')
        self.assertFalse(switched)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(dict(self.events)['decision']['kind'], 'followup')

    def test_none_never_selects_previous_or_default_domain(self):
        self.decision = {'route': 'none'}
        for state in ({}, {'name': 'administrative_procedures', 'allow_switching': True}):
            with self.subTest(state=state):
                self.events.clear()
                with self.assertRaises(self.router.DomainNoMatch) as caught:
                    self.choose(question='Hướng dẫn nấu trứng', state=state, titles=['Đăng ký khai sinh'])
                self.assertEqual(caught.exception.domains, [])
                self.assertNotIn('selected', dict(self.events))

    def test_invalid_model_output_gets_at_most_one_repair(self):
        self.decision = 'not JSON'
        with self.assertRaises(self.router.DomainClarification): self.choose()
        self.assertEqual(len(self.requests), 2)

    def test_bad_output_can_be_repaired_without_mutating_original_input(self):
        self.decisions = [{'domain': 'university_admission', 'kind': 'conversation'},
                          {'route': 'university_admission'}]
        config, _, switched = self.choose(state={'name': 'administrative_procedures', 'allow_switching': True})
        self.assertEqual(config.name, 'university_admission')
        self.assertTrue(switched)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.requests[0]['messages']), 2)
        self.assertEqual(len(self.requests[1]['messages']), 3)
        self.assertIn('exactly one field', json.loads(self.requests[1]['messages'][-1]['content'])['validation_error'])

    def test_http_failure_is_not_retried(self):
        async def fail(*args, **kwargs):
            self.requests.append(kwargs['json'])
            raise self.router.httpx.HTTPError('unavailable')
        self.router.httpx.AsyncClient.post = fail
        with self.assertRaises(self.router.DomainClarification): self.choose()
        self.assertEqual(len(self.requests), 1)

    def test_conversation_does_not_select_a_dataset(self):
        self.decision = {'route': 'conversation'}
        for state in ({}, {'name': 'administrative_procedures', 'allow_switching': True}):
            with self.assertRaises(self.router.DomainConversation):
                self.choose(question='Cảm ơn', state=state, titles=['Đăng ký khai sinh'])
            self.assertIsNone(dict(self.events)['selected']['domain'])

    def test_route_schema_rejects_extra_fields_nulls_and_unknown_actions(self):
        for value in ({'route': 'none', 'domain': 'administrative_procedures'},
                      {'route': None}, {'route': []}, {'route': 'ambiguous'}, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.router.parse_domain_decision(json.dumps(value), {'administrative_procedures'})
        self.assertEqual(self.router.parse_domain_decision('```json\n{"route":"none"}\n```', {}), {'route': 'none'})

    def test_same_domain_explicit_subject_is_not_a_switch(self):
        self.decision = {'route': 'administrative_procedures'}
        config, _, switched = self.choose(state={'name': 'administrative_procedures', 'allow_switching': True}, titles=['Đăng ký khai sinh'])
        self.assertFalse(switched)
        self.assertEqual(config.name, 'administrative_procedures')

    def test_compact_input_limits_examples_and_keeps_question_last(self):
        self.configs[0].profile['routing_examples'] = ['example' + str(i) for i in range(8)]
        self.router.domain_hints = lambda *args: self.fail('Routing must not load record titles')
        self.choose()
        request = json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(list(request)[-1], 'question')
        self.assertEqual(request['domains'][0]['examples'], ['example0', 'example1', 'example2'])
        self.assertEqual(len(self.configs[0].profile['routing_examples']), 8)
        self.assertEqual(set(request), {'domains', 'question'})
        self.assertIn('without guessing an old topic', self.requests[0]['messages'][0]['content'])

    def test_only_compact_source_context_is_sent(self):
        self.decision = {'route': 'administrative_procedures'}
        self.choose(question='A new subject?', state={'name':'university_admission', 'allow_switching':True}, titles=['Prior public subject'])
        request = json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(request['active_context'], {'domain':'university_admission','subjects':['Prior public subject']})
        self.assertNotIn('history', request)
        self.assertEqual(list(request)[-1], 'question')

    def test_current_question_is_preserved_when_context_is_added(self):
        self.choose(question='A standalone library question')
        first = json.loads(self.requests[-1]['messages'][1]['content'])
        self.choose(question='A standalone library question',
            state={'name':'administrative_procedures', 'allow_switching':True},
            titles=['Old birth registration'])
        second = json.loads(self.requests[-1]['messages'][1]['content'])
        self.assertEqual(first['question'],second['question'])
        self.assertEqual(first['domains'],second['domains'])
        self.assertEqual(second['active_context']['subjects'],['Old birth registration'])

    def test_followup_action_is_exposed_to_query_resolution(self):
        self.decision = {'route':'followup'}
        _, state, _ = self.choose(question='And renewal?',
            state={'name':'administrative_procedures', 'allow_switching':True}, titles=['Old subject'])
        self.assertEqual(state['route_action'], 'followup')

    def test_route_control_outcomes_are_not_runtime_errors(self):
        from rag.timing import exception_outcome
        for exc in (self.router.DomainConversation(), self.router.DomainNoMatch(), self.router.DomainClarification([])):
            self.assertEqual(exception_outcome(exc), ('control_flow', None))
        self.assertEqual(exception_outcome(ValueError('bad')), ('error', 'ValueError'))

    def test_subjectless_renewal_keeps_current_domain_without_classifier(self):
        for question in ('Có được gia hạn không và làm ở đâu?', 'co duoc gia han khong va lam o dau?',
                         'Gia hạn được mấy lần, thêm bao nhiêu ngày?', 'Can I renew it and where?'):
            with self.subTest(question=question):
                config, state, switched = self.choose(question=question,
                    state={'name':'university_admission','allow_switching':True}, titles=['Previous service'])
                self.assertEqual(config.name,'university_admission')
                self.assertEqual(state['route_action'],'followup')
                self.assertFalse(switched)
                self.assertFalse(self.requests)

    def test_named_new_renewal_subject_still_uses_classifier(self):
        for question in ('Gia hạn máy khoan ở Trạm Mây được không?', 'Can I renew my passport?',
                         'Tôi không hỏi gia hạn, đăng ký khai sinh mất bao lâu?', 'Có gia hạn sách thư viện không?'):
            with self.subTest(question=question):
                before=len(self.requests)
                config, _, switched=self.choose(question=question,
                    state={'name':'university_admission','allow_switching':True},titles=['Old service'])
                self.assertEqual(len(self.requests),before+1)
                self.assertEqual(config.name,'administrative_procedures')
                self.assertTrue(switched)

    def test_subjectless_renewal_without_active_subject_is_not_guessed(self):
        for state,titles in [({},[]),({'name':'university_admission','allow_switching':True},[])]:
            before=len(self.requests)
            self.choose(question='Có được gia hạn không và làm ở đâu?',state=state,titles=titles)
            self.assertEqual(len(self.requests),before+1)

    def test_ascii_question_gets_equivalent_normalized_coverage(self):
        question='thu vien Demo moi lan muon may cuon sach va trong bao lau?'
        self.configs[1].profile['description']='Mượn sách thư viện và dịch vụ sinh viên Demo.'
        self.configs[1].profile['routing_examples']=['Mỗi lần mượn tối đa mấy cuốn sách?']
        original=json.loads(json.dumps(self.configs[1].profile))
        self.choose(question=question)
        request=json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(request['question'],question)
        self.assertEqual(list(request)[-1],'question')
        domain=request['domains'][1]
        self.assertEqual(domain['normalized_description'],'muon sach thu vien va dich vu sinh vien demo.')
        self.assertEqual(domain['normalized_examples'],['moi lan muon toi da may cuon sach?'])
        self.assertEqual(self.configs[1].profile,original)
        self.assertNotIn('current_domain',request)
        self.assertEqual(len(self.requests),1)

    def test_accented_classifier_input_keeps_previous_shape(self):
        self.choose(question='Ở thư viện Demo, mỗi lần mượn mấy cuốn?')
        request=json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(set(request),{'domains','question'})
        for domain in request['domains']:
            self.assertEqual(set(domain),{'id','description','examples'})

    def test_renewal_shortcut_does_not_override_manual_selection_or_detect_once_lock(self):
        for state in ({'name':'university_admission','selection':'manual','allow_switching':True},
                      {'name':'university_admission','locked':True,'allow_switching':False}):
            config, next_state, changed=self.choose(question='Có được gia hạn không?',state=state,titles=['Old service'],requested=None)
            self.assertEqual(config.name,'university_admission')
            self.assertEqual(next_state['route_action'],'domain')
            self.assertFalse(changed)
            self.assertFalse(self.requests)


if __name__ == '__main__':
    unittest.main()
