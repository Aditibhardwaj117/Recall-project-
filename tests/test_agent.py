from types import SimpleNamespace as NS

import agent


def test_sources_come_only_from_citations_and_are_cleaned():
    ann = lambda u, t: NS(type="url_citation", url=u, title=t)
    resp = NS(output=[
        NS(type="web_search_call", action=NS(query="python latest version")),
        NS(type="message", content=[NS(annotations=[
            ann("https://www.python.org/downloads/?utm_source=openai", "Download Python"),
            ann("https://www.python.org/downloads/", "Duplicate"),
        ])]),
    ])
    sources, queries = agent.extract_sources_and_queries(resp)
    assert queries == ["python latest version"]
    assert sources == [{"title": "Download Python", "url": "https://www.python.org/downloads/",
                        "domain": "python.org"}]


def test_no_search_means_no_sources():
    resp = NS(output=[NS(type="message", content=[NS(annotations=[])])])
    assert agent.extract_sources_and_queries(resp) == ([], [])
