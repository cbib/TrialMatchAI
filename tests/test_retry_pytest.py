from trialmatchai.utils.retry import with_retries


def test_with_retries_succeeds_after_failures():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert with_retries(flaky, retries=3, base_delay=0.0, max_delay=0.0) == "ok"
