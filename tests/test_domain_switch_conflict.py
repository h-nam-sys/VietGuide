"""Replay captured routing disagreement with real router/orchestration, mocked model I/O."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch
import test_routing_contract_integration as fixture

ROOT = Path(__file__).resolve().parents[1]


class RoutingConflictTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixture.ContractIntegrationTests.setUp
    run_turn = fixture.ContractIntegrationTests.run_turn

    def captured(self):
        data = json.loads((ROOT/'tests/fixtures/domain_switch_20261027.json').read_text(encoding='utf-8'))
        self.fixture.configs = [types.SimpleNamespace(name=c['id'], profile={
            'description':c['description'],'routing_examples':c['examples'],'allow_forms':False},
            primary_entity='topic',schema={'topic':'Chủ đề','fee':'Phí'},evidence_fields=['topic','fee'])
            for c in data['domains']]
        self.payload.user_prompt = data['question']
        self.references.side_effect = lambda question,*a: question  # Actual [] result preserves question verbatim.
        self.module.resolve_search_query = lambda question,*a,**kw: (question,[])
        return data

    async def test_captured_followup_disagreement_reclassifies_before_retrieval(self):
        data = self.captured()
        self.fixture.decisions = [data['captured_classifier_decision'], {'route':data['expected_domain']}]
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests),2)
        retry = self.fixture.requests[1]
        self.assertFalse(json.loads(retry['messages'][1]['content'])['followup_context_available'])
        self.assertIn('followup is NOT an allowed action',retry['messages'][0]['content'])
        self.llm.get_legal_response_stream.assert_awaited_once()
        args = self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(args['domain'],'university_test')
        self.assertEqual(args['prompt'],data['question'])
        self.assertEqual(args['search_query'],data['question'])
        self.assertEqual(args['active_procedures'],[])
        self.assertEqual(self.references.await_count,1)
        self.assertEqual(self.conversation.checklist['_domain_state']['name'],'university_test')

    async def test_same_recovery_works_for_another_domain_without_named_rules(self):
        self.captured()
        self.payload.user_prompt = 'Now find USB cables in the Amazon dataset.'
        self.fixture.decisions = [{'route':'followup'},{'route':'amazon_products'}]
        await self.run_turn()
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['domain'],'amazon_products')
        self.assertEqual(len(self.fixture.requests),2)

    async def test_valid_action_followup_keeps_subject_without_an_extra_route_call(self):
        self.captured()
        self.payload.user_prompt = 'Còn điều kiện của thủ tục đó?'
        self.fixture.decision = {'route':'followup'}
        self.references.side_effect = lambda question,*a: 'CURRENT QUESTION: '+question+'\nREFERENCED SUBJECTS: khai sinh'
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests),1)
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['domain'],'administrative_procedures')

    async def test_standalone_same_domain_question_does_not_trigger_conflict_retry(self):
        self.captured()
        self.payload.user_prompt = 'Đăng ký kết hôn cần giấy tờ gì?'
        self.fixture.decision = {'route':'administrative_procedures'}
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests),1)
        self.llm.get_legal_response_stream.assert_awaited_once()

    async def test_manual_domain_lock_is_preserved(self):
        self.captured()
        self.payload.domain = 'administrative_procedures'
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests),0)
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['domain'],'administrative_procedures')

    async def test_detect_once_domain_lock_is_preserved(self):
        self.captured()
        self.payload.allow_switching = False
        await self.run_turn()
        self.assertEqual(len(self.fixture.requests),0)
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['domain'],'administrative_procedures')

    async def test_repeated_followup_output_clarifies_without_wrong_domain_retrieval(self):
        self.captured()
        self.fixture.decision = {'route':'followup'}
        frames = await self.run_turn()
        self.assertEqual(len(self.fixture.requests),3)  # Initial route + bounded existing repair budget.
        self.llm.get_legal_response_stream.assert_not_awaited()
        self.assertEqual(self.dbmod.log_interaction.call_args.args[-1],'DOMAIN_CLARIFICATION')
        self.assertEqual(frames[-1],'data: [DONE]\n\n')

    async def test_reroute_none_uses_existing_no_match_handler(self):
        self.captured()
        self.fixture.decisions = [{'route':'followup'},{'route':'none'}]
        await self.run_turn()
        self.llm.get_legal_response_stream.assert_not_awaited()
        self.assertEqual(self.saved[-1].sources,['RAG_ROUTE:none'])


class CollectorSessionTests(unittest.TestCase):
    def test_explicit_all_sessions_includes_chat_before_restart(self):
        spec = importlib.util.spec_from_file_location('collector_sessions',ROOT/'tests/collect_diagnostics.py')
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for session in ('20261027T073700_000000Z','20261027T074300_000000Z'):
                p = root/'logs/services'/session/'backend.log'
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text('service log')
            old = root/'logs/conversations/old.jsonl'
            old.parent.mkdir(parents=True);old.write_text('{}\n')
            os.utime(old,(1,1))
            _,recent = collector.recent_files(root)
            self.assertNotIn(old,recent)
            _,all_files = collector.recent_files(root,all_sessions=True)
            self.assertIn(old,all_files)
            self.assertEqual(len(all_files),3)


if __name__ == '__main__':
    unittest.main()
