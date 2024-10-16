import json
from typing import List

from docx2python.depth_collector import Par
from docx2python import docx2python
from docx2python.iterators import is_tbl, iter_at_depth, iter_tables
from langchain_core.documents import Document


def _print_text(tbl: List[List[List[Par]]]) -> str:
    all_cells = iter_at_depth(tbl, 2)
    return "\n\n".join(_print_tc(tc) for tc in all_cells)


def _print_tc(cell: List[Par]) -> str:
    ps = ["".join(p.run_strings).replace("\n", " ") for p in cell]
    return "\n\n".join(ps)


def _print_tc_as_json(cell: List[Par]) -> str:
    ps = ["".join(p.run_strings).replace("\n", " ") for p in cell]
    return " ".join(ps)


def _print_tbl_as_json(tbl: List[List[List[Par]]]) -> List[dict]:
    rows_as_dicts = []
    header = [_print_tc_as_json(tc) for tc in tbl[0]]

    for tr in tbl[1:]:
        row = [_print_tc_as_json(tc) for tc in tr]
        row_dict = dict(zip(header, row))
        rows_as_dicts.append(row_dict)

    return rows_as_dicts


def get_document_list_from_docx(file_path: str) -> List[Document]:
    with docx2python(file_path) as docx_content:
        tables = docx_content.document_pars
        document_list: List[Document] = []
        for possible_table in iter_tables(tables):
            if is_tbl(possible_table):
                content = json.dumps(
                    _print_tbl_as_json(possible_table), ensure_ascii=False
                )
                document_list.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_path},
                    )
                )
            else:
                content = _print_text(possible_table)
                document_list.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_path},
                    )
                )
    return document_list
