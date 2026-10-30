"""Mock-only integration regressions for the graph-backed business entry points."""
import copy
import json
from unittest.mock import Mock, patch

import pytest
import agent_graph
from prompt_engineering import PromptBuilder, ItineraryConstraintError
from test_auto_replan import GOOD_PLAN, USER_INPUT, bad_budget


def builder_with(responses):
    builder = PromptBuilder.__new__(PromptBuilder)
    builder._call_model = Mock(side_effect=responses)
    return builder


def response(plan):
    return {"content": json.dumps(plan, ensure_ascii=False)}


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('plans, fails', [([GOOD_PLAN], False),
    ([bad_budget(), GOOD_PLAN], False), ([bad_budget(), bad_budget()], True)])
def test_business_entry_uses_graph(optimize, plans, fails):
    builder = builder_with([response(p) for p in plans])
    with patch('agent_graph.build_graph', wraps=agent_graph.build_graph) as graph:
        def invoke():
            if optimize:
                return builder.optimize_itinerary(USER_INPUT, GOOD_PLAN, '减少费用')
            return builder.generate_itinerary(USER_INPUT)
        if fails:
            with pytest.raises(ItineraryConstraintError):
                invoke()
        else:
            assert invoke()['estimated_total_cost'] == GOOD_PLAN['estimated_total_cost']
        assert graph.call_count == len(plans)


def test_tool_history_survives_replan_without_duplicates():
    tc = {'id': 'weather_1', 'type': 'function', 'function': {
        'name': 'get_weather', 'arguments': '{"city":"北京"}'}}
    seen = []
    responses = iter([{'content': '', 'tool_calls': [tc]}, response(bad_budget()), response(GOOD_PLAN)])
    def model(messages, **kwargs):
        seen.append(copy.deepcopy(messages))
        return next(responses)
    builder = builder_with(model)
    with patch.dict('travel_tools.TOOL_REGISTRY', get_weather=lambda **kw: {'ok': True, 'data': {}}):
        builder.generate_itinerary(USER_INPUT)
    history = seen[-1]
    assert sum(m.get('tool_calls') == [tc] for m in history) == 1
    assert sum(m.get('tool_call_id') == 'weather_1' for m in history) == 1
    assert sum(m.get('content') == response(bad_budget())['content'] for m in history) == 1
    assert any('超过用户预算' in m.get('content', '') for m in history if m['role'] == 'user')


def test_exhaustion_sends_matched_tool_results_and_no_tools():
    tc = {'id': 'poi_1', 'type': 'function', 'function': {
        'name': 'search_poi', 'arguments': '{"keyword":"故宫"}'}}
    def model(messages, tools=None):
        if tools:
            return {'content': '', 'tool_calls': [tc]}
        assert messages[-1]['role'] == 'tool'
        assert messages[-1]['tool_call_id'] == 'poi_1'
        return {'content': '完成'}
    builder = builder_with(model)
    messages = [{'role': 'user', 'content': '北京旅游'}]
    with patch.dict('travel_tools.TOOL_REGISTRY', search_poi=lambda **kw: {'ok': True}):
        assert agent_graph.run_agent(builder, messages, 1) == '完成'
    assert builder._call_model.call_count == 2
    assert [m['role'] for m in messages] == ['user', 'assistant', 'tool', 'assistant']
