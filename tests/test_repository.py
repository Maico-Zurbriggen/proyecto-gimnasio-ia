"""claim_next_pending/complete/fail necesitan Postgres real (FOR UPDATE SKIP LOCKED, jsonb,
enums de ai_integration) -- no hay ese entorno disponible en este repo todavia, asi que solo
se testea la parte pura (el hash de contexto usado para input_hash/context_hash)."""

from gym_engine.persistence.repository import _hash_input


def test_hash_input_is_deterministic() -> None:
    context = {"nivel_experiencia": "intermedio"}
    preferences = {"excluir": ["sentadilla"]}

    assert _hash_input(context, preferences) == _hash_input(context, preferences)


def test_hash_input_differs_on_content() -> None:
    a = _hash_input({"nivel_experiencia": "intermedio"}, {})
    b = _hash_input({"nivel_experiencia": "avanzado"}, {})

    assert a != b


def test_hash_input_ignores_key_order() -> None:
    a = _hash_input({"x": 1, "y": 2}, {})
    b = _hash_input({"y": 2, "x": 1}, {})

    assert a == b
