def get_description(func):
    if hasattr(func, "description_dictionary"):
        return func.description_dictionary
    else:
        return None


def method_descriptor(description_dict):
    def decorator(func):
        func.description_dictionary = description_dict
        return func

    return decorator
