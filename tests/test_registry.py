from autoplayus.games.registry import load_catalog


def test_catalog_loads_and_agents_exist():
    catalog = load_catalog()
    assert catalog
    assert all(g.agent for g in catalog.values())
