"""Focused checks for existing conversation context, without live model inference."""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock
import test_routing_contract_integration as fixture
import test_turn_context as references_fixture

ROOT=Path(__file__).resolve().parents[1]


class FollowupContextTests(unittest.IsolatedAsyncioTestCase):
    run_turn=fixture.ContractIntegrationTests.run_turn

    def setUp(self):
        fixture.ContractIntegrationTests.setUp(self)
        profile=json.loads((ROOT/'testing/fixtures/xuong_nho/xuong_nho_profile.json').read_text(encoding='utf-8'))
        self.config=types.SimpleNamespace(name='maker_lab_test',primary_entity=profile['primary_entity'],
            schema=profile['schema_mapping'],profile={**profile,'allow_forms':False},
            evidence_fields=[k for k in profile['schema_mapping'] if k not in profile['excluded_fields']])
        self.fixture.configs.append(self.config)
        self.conversation.checklist={'_domain_state':{'name':'maker_lab_test','selection':'auto','allow_switching':True,'locked':True}}
        self.title='Mượn máy chiếu — Trạm Nắng'
        self.conversation.current_procedure=self.title
        self.messages[:]=[types.SimpleNamespace(role='assistant',content='40 phút; 3 ngày',
            sources=[self.title,'RAG_CONTENT:{}','RAG_DOMAIN:maker_lab_test'])]
        # Load the actual pure schema matcher instead of the routing fixture's stub.
        source=ast.parse((ROOT/'rag/query_pipeline.py').read_text(encoding='utf-8'))
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='explicit_fields')
        ns={'_plain':self.fixture.router._plain,'re':self.fixture.router.re}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'schema_matcher','exec'),ns)
        self.fixture.router.explicit_fields=ns['explicit_fields']

    async def test_per_use_fee_keeps_projector_without_classifier_or_reference_call(self):
        self.payload.user_prompt='Phí cho mỗi lượt là bao nhiêu?'
        self.fixture.decision={'route':'none'}  # Must never be requested for this recognised property turn.
        await self.run_turn()
        self.assertEqual(self.fixture.requests,[])
        self.references.assert_not_awaited()
        args=self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(args['domain'],'maker_lab_test')
        self.assertEqual(args['active_procedures'],[self.title])
        self.assertIn(self.title,args['search_query'])

    async def test_branch_change_carries_subject_without_pinning_old_record(self):
        self.payload.user_prompt='Nếu chuyển sang Trạm Mây thì thời gian xử lý, hạn giữ và phí thay đổi thế nào?'
        self.fixture.decision={'route':'followup'}
        self.references.side_effect=lambda q,t,c:'CURRENT QUESTION: '+q+'\nREFERENCED SUBJECTS: '+json.dumps(t,ensure_ascii=False)
        await self.run_turn()
        request=json.loads(self.fixture.requests[0]['messages'][1]['content'])
        self.assertEqual(request['active_context'],{'domain':'maker_lab_test','subjects':[self.title]})
        self.assertEqual(list(request)[-1],'question')
        args=self.llm.get_legal_response_stream.call_args.kwargs
        self.assertIn(self.title,args['search_query'])
        self.assertIn('Trạm Mây',args['search_query'])
        self.assertEqual(args['active_procedures'],[])
        self.assertEqual(len(self.fixture.requests),1)
        self.assertEqual(self.references.await_count,1)

    async def test_explicit_new_dataset_overrides_active_projector(self):
        self.payload.user_prompt='Chuyển sang thư viện Demo: được mượn tối đa mấy cuốn sách?'
        self.fixture.decision={'route':'university_test'}
        await self.run_turn()
        args=self.llm.get_legal_response_stream.call_args.kwargs
        self.assertEqual(args['domain'],'university_test')
        self.assertNotIn(self.title,args['search_query'])
        self.assertEqual(args['active_procedures'],[])
        self.references.assert_not_awaited()

    async def test_ambiguous_turn_can_clarify_without_erasing_subject(self):
        self.payload.user_prompt='Vậy thì sao?'
        self.fixture.decision={'route':'clarify'}
        await self.run_turn()
        self.assertEqual(self.conversation.checklist['_domain_state']['name'],'maker_lab_test')
        self.assertEqual(self.module.active_procedures(self.messages,self.title,'maker_lab_test'),[self.title])
        self.llm.get_legal_response_stream.assert_not_awaited()

    async def test_explicit_unrelated_question_still_clears_old_topic(self):
        self.payload.user_prompt='Hướng dẫn tôi nấu trứng chiên.'
        self.fixture.decision={'route':'none'}
        await self.run_turn()
        self.assertEqual(self.saved[-1].sources,['RAG_ROUTE:none'])
        self.assertNotIn('name',self.conversation.checklist['_domain_state'])
        self.llm.get_legal_response_stream.assert_not_awaited()

    async def test_manual_domain_is_not_overridden(self):
        self.payload.domain='maker_lab_test'
        self.payload.user_prompt='Thư viện Demo cho mượn sách trong bao lâu?'
        await self.run_turn()
        self.assertEqual(self.fixture.requests,[])
        self.assertEqual(self.llm.get_legal_response_stream.call_args.kwargs['domain'],'maker_lab_test')

    async def test_pending_question_does_not_mix_in_old_subject(self):
        self.conversation.checklist['_rag_pending_question']='Cần chờ mấy ngày?'
        self.payload.user_prompt='Mình nói đến cấp lại thẻ sinh viên tại Demo.'
        self.fixture.decision={'route':'university_test','resume_pending':True}
        await self.run_turn()
        request=json.loads(self.fixture.requests[0]['messages'][1]['content'])
        self.assertNotIn('active_context',request)
        self.assertIn('pending_question',request)


class ReferenceQualifierTests(unittest.IsolatedAsyncioTestCase):
    setUp=references_fixture.ReferenceTests.setUp

    async def test_existing_reference_call_retains_service_when_only_branch_changes(self):
        question='Nếu chuyển sang Trạm Mây thì thời gian xử lý và phí thay đổi thế nào?'
        resolved=await self.module.resolve_references(question,['Mượn máy chiếu — Trạm Nắng'],self.config)
        self.assertIn(question,resolved)
        self.assertIn('Mượn máy chiếu',resolved)
        self.assertEqual(len(self.requests),1)
        self.assertIn('does not lock its old branch',self.requests[0]['messages'][0]['content'])


if __name__=='__main__':
    unittest.main()
