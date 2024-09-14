from enum import Enum


class FeishuBlockType(Enum):
    page = 1
    text = 2
    heading1 = 3
    heading2 = 4
    heading3 = 5
    heading4 = 6
    heading5 = 7
    heading6 = 8
    heading7 = 9
    heading8 = 10
    heading9 = 11
    bullet = 12
    ordered = 13
    code = 14
    quote = 15
    equation = 16
    todo = 17
    bitable = 18
    callout = 19
    chat_card = 20
    uml_diagram = 21
    divider = 22
    file = 23
    grid = 24
    grid_column = 25
    iframe = 26
    image = 27
    isv = 28
    mindnote = 29
    sheet = 30
    table = 31
    table_cell = 32
    view = 33
    quote_container = 34
    task = 35
    okr = 36
    okr_objective = 37
    okr_key_result = 38
    okr_progress = 39
    add_ons = 40
    jira_issue = 41
    wiki_catalog = 42
