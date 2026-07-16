from sld.core import hello


def test_hello():
    result = hello()
    assert result
    assert "sld" in result
