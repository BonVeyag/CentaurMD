from app.guideline_runner import _eval_logic


def test_eval_logic_numeric():
    cond = {"var": "age", "op": "gt", "value": 18}
    assert _eval_logic(cond, {"age": 25}) is True
    assert _eval_logic(cond, {"age": 12}) is False


def test_eval_logic_any_of():
    cond = {
        "any_of": [
            {"var": "sex", "op": "eq", "value": "female"},
            {"var": "age", "op": "lt", "value": 5},
        ]
    }
    assert _eval_logic(cond, {"sex": "female"}) is True
    assert _eval_logic(cond, {"age": 3}) is True
    assert _eval_logic(cond, {"sex": "male", "age": 10}) is False
