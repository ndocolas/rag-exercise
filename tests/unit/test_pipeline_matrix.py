from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix


def test_builds_nine_pipelines():
    specs = PipelineMatrix.build()
    assert len(specs) == 9
    ids = [s.pipeline_id for s in specs]
    assert ids == [f"P{i}" for i in range(1, 10)]


def test_unique_collection_names():
    specs = PipelineMatrix.build()
    names = {s.collection_name() for s in specs}
    assert len(names) == 9


def test_filter():
    specs = PipelineMatrix.build(filter_ids=["P1", "P5"])
    assert [s.pipeline_id for s in specs] == ["P1", "P5"]


def test_chunking_axis():
    specs = PipelineMatrix.build()
    chunkings = {s.chunking for s in specs}
    assert chunkings == {"fixed", "semantic", "hierarchical"}


def test_embedder_axis():
    specs = PipelineMatrix.build()
    embedders = {s.embedder for s in specs}
    assert len(embedders) == 3
