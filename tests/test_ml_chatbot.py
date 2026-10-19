import json
from ml_chatbot.agent import route_query


def test_simple_code_query():
    resp = route_query('Where is the visualization HTML for Code-to-Visualizer?')
    assert 'intent' in resp


def test_dashboard_query():
    resp = route_query('Show me prototype performance')
    assert 'intent' in resp
