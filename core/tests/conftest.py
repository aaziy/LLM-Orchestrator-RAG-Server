import pytest
from django.conf import settings

from core.services import llm


@pytest.fixture(autouse=True)
def use_fake_provider():
    """Force the deterministic fake provider so tests never hit the network."""
    original = settings.RAG["PROVIDER"]
    settings.RAG["PROVIDER"] = "fake"
    llm.get_provider.cache_clear()
    yield
    settings.RAG["PROVIDER"] = original
    llm.get_provider.cache_clear()
