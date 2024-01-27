def find_contained_string(main_string: str, string_list: list[str]):
    for sub_string in string_list:
        if sub_string in main_string:
            return sub_string
    return None
