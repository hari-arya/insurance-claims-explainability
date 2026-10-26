from src.explanations import prompt


def sample_claim():
    return {
        "prob": 0.76,
        "threshold": 0.49,
        "row": {
            "policy_state": "OH",
            "policy_csl": "500/1000",
            "policy_deductable": 500,
            "policy_annual_premium": 973.80,
            "umbrella_limit": 0,
            "months_as_customer": 72,
            "policy_tenure_days": 7468,
            "age": 29,
            "insured_sex": "FEMALE",
            "insured_education_level": "Associate",
            "insured_occupation": "craft-repair",
            "insured_hobbies": "exercise",
            "incident_type": "Single Vehicle Collision",
            "collision_type": "Rear Collision",
            "incident_severity": "Major Damage",
            "number_of_vehicles_involved": 1,
            "bodily_injuries": 1,
            "witnesses": 0,
            "authorities_contacted": "Ambulance",
            "police_report_available": "NO",
            "property_damage": "UNKNOWN",
            "incident_hour_of_the_day": 9,
            "incident_state": "WV",
            "total_claim_amount": 66200,
            "injury_claim": 6620,
            "property_claim": 6620,
            "vehicle_claim": 52960,
            "auto_make": "Dodge",
            "auto_year": 2013,
        },
        "drivers": [
            {"label": f"factor {i}", "value": f"value {i}", "shap": 0.1 - i * 0.02,
             "direction": "raises" if i == 1 else "lowers"}
            for i in range(1, 6)
        ],
    }


def test_all_prompt_variants_require_grounded_action_and_driver_directions():
    for prompt_id in ("P0", "P1", "P2", "P3"):
        system, user = prompt(sample_claim(), prompt_id)
        combined = f"{system or ''}\n{user}".lower()
        assert "do not add outside facts" in combined
        assert "infer fraud" in combined
        assert "mention each of the five supplied factors" in combined
        assert "recommended check" in combined
        assert "what to compare or confirm" in combined
        assert "do not invent people, events, records, statements, comparisons, results" in combined
        assert "not a finding" in combined


def test_prompt_designs_retain_distinct_treatments():
    claim = sample_claim()
    p0_system, p0_user = prompt(claim, "P0")
    p1_system, _ = prompt(claim, "P1")
    p2_system, p2_user = prompt(claim, "P2")
    p3_system, p3_user = prompt(claim, "P3")

    assert p0_system is None
    assert "explain in plain english" in p0_user.lower()
    assert "You are a careful insurance-claims explanation writer" in p1_system
    assert "privately verify each supplied factor" in p2_user
    assert "do not show this check" in p2_user.lower()
    assert "REASONING" not in p2_user and "EXPLANATION" not in p2_user
    assert "EXAMPLE 1" in p3_user
    assert p2_system == p1_system == p3_system


def test_all_few_shot_examples_fit_word_limit():
    from src.explanations import EXEMPLARS

    for _, example in EXEMPLARS:
        assert 80 <= len(example.split()) <= 130
        assert example.count("Recommended check:") == 1
