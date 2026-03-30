from app.chat_retrieval_flow import (
    _force_company_name_anchor_for_followup,
    _stabilize_followup_merged_query,
)


def test_stabilize_followup_should_prevent_generic_drift():
    stabilized = _stabilize_followup_merged_query(
        merged_query="更多 更多",
        question="更多",
        last_effective_search_query="公司名称 更多",
    )
    assert stabilized == "公司名称 更多"


def test_stabilize_followup_should_keep_non_generic_query():
    stabilized = _stabilize_followup_merged_query(
        merged_query="找下公司名 3月25号",
        question="3月25号",
        last_effective_search_query="公司名称",
    )
    assert stabilized == "找下公司名 3月25号"


def test_force_company_name_anchor_should_fix_company_info_drift_on_more():
    stabilized = _force_company_name_anchor_for_followup(
        search_query="公司信息 更多",
        base_query="公司名 更多",
        question="更多",
        last_answer_type="enumeration_company",
    )
    assert "公司" in stabilized.split()
    assert "名称" in stabilized.split()


def test_force_company_name_anchor_should_not_change_non_company_followup():
    stabilized = _force_company_name_anchor_for_followup(
        search_query="项目进展 更多",
        base_query="项目进展 更多",
        question="更多",
        last_answer_type="enumeration_file",
    )
    assert stabilized == "项目进展 更多"
