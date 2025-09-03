from dam_fs.utils.url_utils import parse_dam_url

def test_parse_dam_url_with_passwords():
    url = "dam://local_cas/some_hash?pwd=p1&pwd2=p2"
    parsed = parse_dam_url(url)
    assert parsed["passwords"] == ["p1", "p2"]
    assert "pwd" not in parsed["query"]
    assert "pwd2" not in parsed["query"]

def test_parse_dam_url_with_single_password():
    url = "dam://local_cas/some_hash?pwd=p1"
    parsed = parse_dam_url(url)
    assert parsed["passwords"] == ["p1"]
    assert "pwd" not in parsed["query"]

def test_parse_dam_url_without_passwords():
    url = "dam://local_cas/some_hash"
    parsed = parse_dam_url(url)
    assert parsed["passwords"] == []
