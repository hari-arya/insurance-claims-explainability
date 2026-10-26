import pytest

from src.text_eval import JUDGE_JSON_SCHEMA, parse_judge


def test_judge_schema_uses_anthropic_supported_integer_keywords():
    properties = JUDGE_JSON_SCHEMA["properties"]

    assert properties["clarity"] == {"type": "integer"}
    assert properties["actionability"] == {"type": "integer"}


def test_parse_judge_reads_json_and_computes_unsupported_rate():
    parsed = parse_judge(
        '```json\n{"claims":[{"text":"A fact","supported":true},'
        '{"text":"Another fact","supported":false}],'
        '"clarity":4,"actionability":3,"rationale":"Clear enough."}\n```'
    )

    assert parsed["clarity"] == 4
    assert parsed["actionability"] == 3
    assert parsed["n_claims"] == 2
    assert parsed["n_unsupported"] == 1
    assert parsed["ucr"] == 0.5


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"claims":[],"actionability":3,"rationale":"x"}', "clarity"),
        ('{"claims":[{"text":"fact","supported":"yes"}],"clarity":4,'
         '"actionability":3,"rationale":"x"}', "claims"),
        ('{"claims":[],"clarity":6,"actionability":3,"rationale":"x"}', "clarity"),
    ],
)
def test_parse_judge_rejects_invalid_schema(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_judge(raw)


def test_parse_judge_handles_json_followed_by_extra_text():
    raw = (
        '{"claims":[],"clarity":4,"actionability":3,"rationale":"x"}'
        '\n{"extra":"malformed trailing response"}'
    )

    assert parse_judge(raw)["clarity"] == 4
