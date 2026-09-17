"""Minimal weaviate stubs so retrieval unit tests run without the client package."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def install_weaviate_stubs() -> None:
    if "weaviate" in sys.modules and hasattr(sys.modules["weaviate"], "classes"):
        return

    weaviate = ModuleType("weaviate")
    classes = ModuleType("weaviate.classes")
    query = ModuleType("weaviate.classes.query")
    config = ModuleType("weaviate.classes.config")
    connect = ModuleType("weaviate.connect")
    executor = ModuleType("weaviate.connect.executor")
    auth = ModuleType("weaviate.auth")
    util = ModuleType("weaviate.util")

    class MetadataQuery:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FilterBuilder:
        def __init__(self, name: str):
            self.name = name

        def equal(self, value):
            return ("eq", self.name, value)

    class Filter:
        @staticmethod
        def by_property(name: str):
            return _FilterBuilder(name)

    query.MetadataQuery = MetadataQuery
    query.Filter = Filter

    class _AnyStub:
        """Permissive stub: any attribute access or call returns another stub."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return _AnyStub(*args, **kwargs)

        def __getattr__(self, name):
            return _AnyStub()

    config.Property = _AnyStub()
    config.ReferenceProperty = _AnyStub()
    config.DataType = _AnyStub()
    config.Configure = _AnyStub()
    config.VectorDistances = _AnyStub()
    config.Tokenization = _AnyStub()

    auth.Auth = SimpleNamespace(api_key=lambda key: key)
    util.generate_uuid5 = lambda identifier, namespace="": f"uuid5-{identifier}"

    connect.executor = executor
    classes.query = query
    classes.config = config
    weaviate.classes = classes
    weaviate.connect = connect
    weaviate.auth = auth
    weaviate.util = util
    weaviate.__path__ = []  # mark as package so submodule imports resolve

    sys.modules["weaviate"] = weaviate
    sys.modules["weaviate.classes"] = classes
    sys.modules["weaviate.classes.query"] = query
    sys.modules["weaviate.classes.config"] = config
    sys.modules["weaviate.connect"] = connect
    sys.modules["weaviate.connect.executor"] = executor
    sys.modules["weaviate.auth"] = auth
    sys.modules["weaviate.util"] = util
