from src.data.paper_faithful import maybe_load_split, resolve_split_path


def test_resolve_split_path_finds_nested_canonical_dataset() -> None:
    path = resolve_split_path("data", "S1", "train")
    assert path is not None
    assert path.name == "s1_train.npz"


def test_maybe_load_split_returns_expected_shapes() -> None:
    loaded = maybe_load_split("data/paper_faithful", "S3", "test")
    assert loaded is not None

    X, y, taus, path = loaded
    assert path.name == "s3_test.npz"
    assert X.shape == (2000, 100)
    assert y.shape == (2000,)
    assert taus.shape == (2000,)
