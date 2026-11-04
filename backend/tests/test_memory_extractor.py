"""The extractor validates fixed model output without contacting a provider."""

import copy
import importlib

import pytest

from memory_test_support import FACT, ONTOLOGY, SOURCE_TEXT, FixedLLM, extraction_payload


def test_fixed_model_output_becomes_the_planned_extraction_types():
    memory = importlib.import_module("app.memory")
    models = importlib.import_module("app.memory.models")
    fake = FixedLLM(extraction_payload())
    result = memory.GraphExtractor(llm_client=fake).extract(SOURCE_TEXT, ONTOLOGY, ["周宁"])
    assert isinstance(result, models.Extraction)
    assert isinstance(result.entities, tuple)
    assert isinstance(result.relations, tuple)
    assert result.entities[0] == models.ExtractedEntity(name="周宁", type="Person", summary="周宁参与讨论", attributes={"role": "参与者"})
    assert result.relations == (models.ExtractedRelation(source_name="周宁", target_name="林晓", name="KNOWS", fact=FACT),)
    assert len(fake.calls) == 1


def test_extraction_passes_source_ontology_known_names_and_bounded_retry_options():
    memory = importlib.import_module("app.memory")
    fake = FixedLLM(extraction_payload())
    memory.GraphExtractor(llm_client=fake).extract(SOURCE_TEXT, ONTOLOGY, ["周宁", "林晓"])
    messages, options = fake.calls[0]
    assert any(message["role"] == "system" for message in messages)
    prompt = "\n".join(message["content"] for message in messages)
    assert SOURCE_TEXT in prompt
    assert all(value in prompt for value in ("Person", "KNOWS", "周宁", "林晓"))
    assert options == {"temperature": 0.1, "max_tokens": 4096, "max_attempts": 2}


@pytest.mark.parametrize(("field", "bad_value"), [
    ("name", None), ("name", 123), ("name", "   "),
    ("type", []), ("summary", {}), ("attributes", []),
])
def test_entity_entries_with_wrong_types_or_blank_names_are_discarded(field, bad_value):
    memory = importlib.import_module("app.memory")
    payload = extraction_payload()
    invalid = copy.deepcopy(payload["entities"][0])
    invalid[field] = bad_value
    payload["entities"] = [invalid, payload["entities"][1]]
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert [entity.name for entity in result.entities] == ["林晓"]


@pytest.mark.parametrize("field", ["name", "type", "summary", "attributes"])
def test_entity_entries_with_missing_required_fields_are_discarded(field):
    memory = importlib.import_module("app.memory")
    payload = extraction_payload()
    invalid = payload["entities"][0]
    del invalid[field]
    payload["entities"] = [invalid]
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert result.entities == ()


@pytest.mark.parametrize(("field", "bad_value"), [
    ("source", None), ("source", " "), ("target", 123),
    ("type", []), ("fact", {}), ("fact", " "),
])
def test_relation_entries_with_wrong_types_or_blank_values_are_discarded(field, bad_value):
    memory = importlib.import_module("app.memory")
    payload = extraction_payload()
    payload["relations"][0][field] = bad_value
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert result.relations == ()


@pytest.mark.parametrize("field", ["source", "target", "type", "fact"])
def test_relation_entries_with_missing_required_fields_are_discarded(field):
    memory = importlib.import_module("app.memory")
    payload = extraction_payload()
    del payload["relations"][0][field]
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert result.relations == ()


def test_non_object_entries_are_discarded_and_null_entity_type_is_allowed():
    memory = importlib.import_module("app.memory")
    entity = {"name": "周宁", "type": None, "summary": "", "attributes": {}}
    payload = {"entities": [None, [], "wrong", entity], "relations": [None, 123, []]}
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert len(result.entities) == 1
    assert result.entities[0].type is None
    assert result.relations == ()


@pytest.mark.parametrize("response", [None, [], "not a JSON object", 42])
def test_non_object_model_response_raises_instead_of_fabricating_data(response):
    memory = importlib.import_module("app.memory")
    from app.utils.llm_client import LLMResponseError

    with pytest.raises(LLMResponseError):
        memory.GraphExtractor(llm_client=FixedLLM(response)).extract(SOURCE_TEXT, ONTOLOGY, [])


def test_provider_parse_failure_is_propagated_without_a_fallback_extraction():
    memory = importlib.import_module("app.memory")
    from app.utils.llm_client import LLMResponseError

    fake = FixedLLM(error=LLMResponseError("synthetic unusable extraction"))
    with pytest.raises(LLMResponseError, match="synthetic unusable extraction"):
        memory.GraphExtractor(llm_client=fake).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert len(fake.calls) == 1


def test_extraction_caps_entity_and_relation_counts():
    memory = importlib.import_module("app.memory")
    payload = {
        "entities": [{"name": f"人物{index}", "type": "Person", "summary": "", "attributes": {}} for index in range(55)],
        "relations": [{"source": "人物0", "target": "人物1", "type": "KNOWS", "fact": f"事实{index}"} for index in range(105)],
    }
    result = memory.GraphExtractor(llm_client=FixedLLM(payload)).extract(SOURCE_TEXT, ONTOLOGY, [])
    assert len(result.entities) == 50
    assert len(result.relations) == 100
    assert result.entities[-1].name == "人物49"
    assert result.relations[-1].fact == "事实99"
