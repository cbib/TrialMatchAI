"""Open-vocabulary parsers for the concept store (entities/concept_sources.py)."""

from trialmatchai.entities.concept_sources import (
    OPEN_SOURCES,
    _clean_names,
    parse_gene_info,
    parse_obo,
)


def test_parse_obo_filters_obsolete_and_prefix(tmp_path):
    obo = tmp_path / "t.obo"
    obo.write_text(
        '[Term]\nid: CL:0001\nname: T cell\nsynonym: "T-cell" EXACT []\n\n'
        "[Term]\nid: CL:0002\nname: dead cell\nis_obsolete: true\n\n"
        "[Term]\nid: HP:0001\nname: seizure\n"
    )
    rows = list(parse_obo(obo, "CL:"))
    assert rows == [("CL:0001", ["T cell", "T-cell"])]  # obsolete + non-CL dropped


def test_parse_gene_info_one_row(tmp_path):
    tsv = tmp_path / "g.tsv"
    tsv.write_text(
        "#header\n"
        "9606\t7157\tTP53\t-\tP53|LFS1\t-\t17\t17p13\t"
        "tumor protein p53\tprotein-coding\tTP53\ttumor protein p53\tO\tp53\t-\n"
    )
    rows = list(parse_gene_info(tsv))
    assert len(rows) == 1
    gene_id, names = rows[0]
    assert gene_id == "7157"
    assert names[0] == "TP53" and "P53" in names


def test_parse_gene_info_skips_short_lines(tmp_path):
    tsv = tmp_path / "g.tsv"
    tsv.write_text("9606\t7157\tTP53\n")  # < 14 columns
    assert list(parse_gene_info(tsv)) == []


def test_clean_names_dedupes_casefold_and_strips_whitespace():
    assert _clean_names(["T cell", "t  CELL", "  ", "T-cell"]) == ["T cell", "T-cell"]


def test_open_sources_registry_well_formed():
    assert {"genes", "diseases", "chemicals", "cell_lines"} <= set(OPEN_SOURCES)
    for src in OPEN_SOURCES.values():
        assert src.kind in {"obo", "gene_info"}
        assert src.url.startswith("http")
        assert src.dict_filename.startswith("dict_")
