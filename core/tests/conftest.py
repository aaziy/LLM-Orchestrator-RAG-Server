import pytest
from django.conf import settings

from core.services import llm, rerank


@pytest.fixture(autouse=True)
def use_fake_backends():
    """Ensure provider/reranker caches are clear and pinned to fake backends.

    config.settings_test already pins PROVIDER/RERANKER to "fake"; this guards
    against cache bleed between tests that toggle settings.
    """
    settings.RAG["PROVIDER"] = "fake"
    settings.RAG["RERANKER"] = "fake"
    llm.get_provider.cache_clear()
    rerank.get_reranker.cache_clear()
    yield
    llm.get_provider.cache_clear()
    rerank.get_reranker.cache_clear()
